import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x = 2.898 * 10**-3 / x  # compute wavelength using Wein's Law
        x = self.fc3(x)
        return x

def generate_chunk(idxs, metaPath):
    wavelengths = ["94", "131", "171", "193", "211", "304", "335", "1700", "continuum", "magnetogram"]
    x = np.zeros((len(idxs), 4, 256, 256, 10), dtype=np.int64)
    y = np.zeros((len(idxs), 1), dtype=np.int64)
    df = pd.read_csv(metaPath)
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

        idx = path.split("/")[2] + "_" + path.split("/")[3]
        peak_flux = peak_flux_dict[idx]

        if peak_flux < 1e-5:
            y[i] = 0  # Common
        elif peak_flux < 1e-4:
            y[i] = 1  # Moderate
        else:
            y[i] = 2  # Extreme

    return torch.from_numpy(x.transpose(0, 4, 2, 3, 1)).to(device), torch.from_numpy(y).to(device)

def train_model(model, criterion, optimizer, epochs, checkpoint_path):
    all_files = glob.glob("./training/*/*")
    chunks = np.array_split(all_files, 40)

    model.to(device)

    best_accuracy = 0.0
    loss_curve = []
    accuracy_curve = []

    for epoch in tqdm(range(epochs)):
        for i in range(39):
            train_x, train_y = generate_chunk(chunks[i], 'training/meta_data.csv')
            dataset = TensorDataset(train_x, train_y)
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.float().to(device), targets.long().reshape(-1).to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        train_x, train_y = generate_chunk(chunks[39], 'training/meta_data.csv')
        dataset = TensorDataset(train_x, train_y)
        val_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            total_samples = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.float().to(device), targets.long().reshape(-1).to(device)
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

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_path)

    plt.plot(loss_curve)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig('loss_curve.png')
    plt.close()

    plt.plot(accuracy_curve)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.savefig('accuracy_curve.png')
    plt.close()

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def eval_model(model, criterion, model_path):
    all_files = glob.glob("./test/*/*")
    chunks = np.array_split(all_files, 40)

    # Eval checkpoint model and print accuracy score
    model.load_state_dict(torch.load(model_path))
    model.eval()

    model.to(device)

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(40):
            train_x, train_y = generate_chunk(chunks[i], 'test/meta_data.csv')
            dataset = TensorDataset(train_x, train_y)
            test_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            for data, target in test_loader:
                data, target = data.to(device).float(), target.to(device).long().reshape(-1)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    print(f"Model : {model_path} Accuracy: {accuracy}%")

model = SimpleCNN().to(device)
train_model(model, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters()), 5, './SimpleCNN.mod')
eval_model(model, nn.CrossEntropyLoss(), './SimpleCNN.mod')

model = SimpleCNNWein().to(device)
train_model(model, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters()), 5, './SimpleCNNWein.mod')
eval_model(model, nn.CrossEntropyLoss(), './SimpleCNNWein.mod')
