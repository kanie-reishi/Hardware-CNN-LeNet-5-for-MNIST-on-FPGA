import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# LeNet-5 Architecture
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 32x32 → 32x32
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)              # 14x14 → 10x10
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.pool  = nn.AvgPool2d(2, 2)
        self.act   = nn.Tanh()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))   # → 14x14
        x = self.pool(self.act(self.conv2(x)))   # → 5x5
        x = x.view(-1, 16*5*5)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

# Data
transform = transforms.Compose([transforms.Pad(2),  # MNIST is 28x28, LeNet-5 expects 32x32
                                transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST('.', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('.', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=1000)

# Train
model = LeNet5()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def train(model, epochs=10):
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
        acc = evaluate(model)
        print(f"Epoch {epoch+1}: Accuracy = {acc:.2f}%")

def evaluate(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return 100. * correct / len(test_loader.dataset)

train(model, epochs=15)
torch.save(model.state_dict(), 'lenet5_float.pth')
#Expected Float32 Accuracy: ~98.5~98.9%** — this is your baseline.