import glob
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv3d(10, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 256 * 256 * 4, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the model
model = SimpleCNN()

class SimpleCNNWein(nn.Module):
    def __init__(self):
        super(SimpleCNNWein, self).__init__()
        self.conv1 = nn.Conv3d(10, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 256 * 256 * 4, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(1, 3)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # compute wavelength using Wein's Law
        x = 2.898 * 10**-3 / x
        x = self.fc3(x)
        return x

# Create the model
model = SimpleCNNWein()

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Get a list of all active region numbers
all_files = glob.glob("./training/*/*")

# Split all_files into 10 chunks
chunks = np.array_split(all_files, 40)
print(len(chunks[0]))

def generateChunk(idxs):
    wavelengths = ["94","131", "171","193","211","304","335","1700","continuum","magnetogram"]
    x = np.zeros((len(idxs), 4, 256, 256, 10), dtype=np.int64)
    y = np.zeros((len(idxs), 1), dtype=np.int64)  # Updated data type to int64
    df = pd.read_csv('training/meta_data.csv')
    peak_flux_dict = {idx: peak_flux for idx, peak_flux in zip(df['id'], df['peak_flux'])}

    for i, sample in enumerate(idxs):
        images = np.empty((4, 256, 256, 10), dtype=np.int64)
        for j, wave in enumerate(wavelengths):
            path = sample + "/*_{}.jpg".format(wave)
            pics = np.array([np.array(Image.open(i)) for i in glob.glob(path)])
            for _ in range(4 - len(pics)):
                if len(pics) == 0:
                    pics = np.zeros((1, 256, 256))
                else:
                    pics = np.concatenate((pics, np.zeros((1, 256, 256))), axis=0)
            pics = pics.reshape(4, 256, 256, 1)
            images[:, :, :, j] = pics[:, :, :, 0]
        images = images[:, :, :, :]
        x[i] = images

        idx = path.split("/")[2] + "_"+ path.split("/")[3]
        peak_flux = peak_flux_dict[idx]

        if peak_flux < 1e-5:
            y[i] = 0  # Common
        elif peak_flux < 1e-4:
            y[i] = 1  # Moderate
        else:
            y[i] = 2  # Extreme

    return torch.from_numpy(x.transpose(0,4,2,3,1)), torch.from_numpy(y)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import matplotlib.pyplot as plt

def train_model(model, criterion, optimizer, epochs, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the GPU if available

    best_accuracy = 0.0
    loss_curve = []
    accuracy_curve = []

    for epoch in range(epochs):
        for i in range(39):
            train_x, train_y = generateChunk(chunks[i])
            dataset = TensorDataset(train_x, train_y)
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            model.train()  # Set the model to training mode
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device).float(), targets.to(device).long().reshape(-1)  # Move data to GPU
                optimizer.zero_grad()  # Reset the gradients
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

        train_x, train_y = generateChunk(chunks[39])
        dataset = TensorDataset(train_x, train_y)
        val_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No need to track gradients in validation phase
            total_loss = 0
            total_correct = 0
            total_samples = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device).float(), targets.to(device).long().reshape(-1)  # Move data to GPU
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * len(targets)
                _, predicted = torch.max(outputs, 1)
                total_correct += torch.sum(predicted == targets)
                total_samples += len(targets)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Accuracy: {accuracy}')

        loss_curve.append(avg_loss)
        accuracy_curve.append(accuracy)

        # Save checkpoint if accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_path)

    # Plot and save loss curve
    plt.plot(loss_curve)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig('loss_curve.png')
    plt.close()

    # Plot and save accuracy curve
    plt.plot(accuracy_curve)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.savefig('accuracy_curve.png')
    plt.close()


mod = SimpleCNN()
train_model(mod, nn.CrossEntropyLoss(), torch.optim.Adam(mod.parameters()), 10, './')
mod = SimpleCNNWein()
train_model(mod, nn.CrossEntropyLoss(), torch.optim.Adam(mod.parameters()), 10, './')