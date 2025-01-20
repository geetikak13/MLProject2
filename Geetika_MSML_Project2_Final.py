import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import struct

#------------------------------------
# Utilities for loading MNIST data
#------------------------------------
def read_idx_images(filename):
    # IDX format: Magic number (4 bytes) - should be 2051 for images
    # Number of images (4 bytes)
    # Number of rows (4 bytes)
    # Number of cols (4 bytes)
    # Image Data (number_of_images x rows x cols)
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError('Invalid magic number %d in file %s' % (magic, filename))
        data = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return data

def read_idx_labels(filename):
    # IDX format for labels: Magic number (4 bytes) - should be 2049
    # Number of labels (4 bytes)
    # Labels (number_of_labels)
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError('Invalid magic number %d in file %s' % (magic, filename))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)/255.0
        img = np.expand_dims(img, axis=0)  # shape: (1, 28, 28)
        label = self.labels[idx].astype(np.int64)
        
        if self.transform:
            img = self.transform(img)
        
        return torch.tensor(img), torch.tensor(label)

#------------------------------------
# Neural Network Architectures
#------------------------------------

# (a) Feedforward network with at least two hidden layers
class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        # Flatten: input dimension = 28*28 = 784
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# (b) Convolutional neural network with at least two convolutional layers and two fully connected layers
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Input: (N, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # -> (N, 32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> (N, 64, 28, 28)
        
        # Optionally apply a pooling layer to reduce dimension
        self.pool = nn.MaxPool2d(2,2) # -> reduces to (N, 64, 14, 14)
        
        # Fully connected layers
        # After pooling, feature map size: 64 * 14 *14
        self.fc1 = nn.Linear(64*14*14, 256)
        self.fc2 = nn.Linear(256, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#------------------------------------
# Training and Evaluation Functions
#------------------------------------
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(train_loader.dataset)

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

#------------------------------------
# Main Execution
#------------------------------------
if __name__ == "__main__":
    # Adjust these paths as necessary
    train_images_path = "Data/train-images.idx3-ubyte"
    train_labels_path = "Data/train-labels.idx1-ubyte"
    test_images_path =  "Data/t10k-images.idx3-ubyte"
    test_labels_path =  "Data/t10k-labels.idx1-ubyte"
    
    # Load data
    train_images = read_idx_images(train_images_path)
    train_labels = read_idx_labels(train_labels_path)
    test_images = read_idx_images(test_images_path)
    test_labels = read_idx_labels(test_labels_path)
    
    # Create datasets and loaders
    batch_size = 128
    train_dataset = MNISTDataset(train_images, train_labels)
    test_dataset  = MNISTDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We will run the training for 5 runs for each network and compute average test accuracy
    num_runs = 5
    num_epochs = 5  # Increase if needed

    # (a) Feedforward Network
    ff_accuracies = []
    for run in range(num_runs):
        model_ff = FeedForwardNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_ff.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            train_loss = train_model(model_ff, train_loader, criterion, optimizer, device)
        ff_test_acc = evaluate_model(model_ff, test_loader, device)
        ff_accuracies.append(ff_test_acc)
        print(f"Feedforward Run {run+1}: Test Accuracy = {ff_test_acc:.2%}")

    ff_accuracies_avg = np.mean(ff_accuracies)
    print(f"Average Feedforward Test Accuracy over 5 runs: {ff_accuracies_avg:.2%}")

    # (b) Convolutional Network
    cnn_accuracies = []
    for run in range(num_runs):
        model_cnn = ConvNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            train_loss = train_model(model_cnn, train_loader, criterion, optimizer, device)
        cnn_test_acc = evaluate_model(model_cnn, test_loader, device)
        cnn_accuracies.append(cnn_test_acc)
        print(f"CNN Run {run+1}: Test Accuracy = {cnn_test_acc:.2%}")

    cnn_accuracies_avg = np.mean(cnn_accuracies)
    print(f"Average CNN Test Accuracy over 5 runs: {cnn_accuracies_avg:.2%}")
