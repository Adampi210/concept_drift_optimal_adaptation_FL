import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

# Set random seed for reproducibility
torch.manual_seed(0)
random.seed(0)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Load MNIST dataset (unchanged)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize the model (unchanged)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# Pretraining (unchanged)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Pretrain for 2 epochs
print("Pretraining the model...")
for epoch in range(1, 3):
    train(model, device, train_loader, optimizer, epoch)

print("Pretraining completed.")

# Function to apply cumulative concept drift
def apply_cumulative_concept_drift(original_labels, drifted_labels, drift_probability):
    for i in range(len(original_labels)):
        if drifted_labels[i] == original_labels[i] and random.random() < drift_probability:
            new_label = random.choice([j for j in range(10) if j != original_labels[i]])
            drifted_labels[i] = new_label
    return drifted_labels

# Evaluation function
def evaluate(model, device, test_loader, drifted_labels, drift_probability):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    drifted_labels_tensor = drifted_labels.to(device)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            batch_size = target.size(0)
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            # Apply drift to the batch
            batch_drifted_labels = drifted_labels_tensor[start_idx:end_idx]
            batch_drifted_labels = apply_cumulative_concept_drift(target, batch_drifted_labels, drift_probability)
            drifted_labels_tensor[start_idx:end_idx] = batch_drifted_labels
            
            output = model(data)
            test_loss += criterion(output, batch_drifted_labels).item() * batch_size
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(batch_drifted_labels.view_as(pred)).sum().item()
            total += batch_size

    test_loss /= total
    accuracy = 100. * correct / total
    return test_loss, accuracy, drifted_labels_tensor.cpu()

# Evaluate the model over multiple rounds with cumulative concept drift
num_rounds = 400
drift_probability = 0.01 
losses = []
accuracies = []

# Initialize drifted labels as a copy of the original labels
original_labels = torch.cat([y for _, y in test_loader])
drifted_labels = original_labels.clone()

for round in range(num_rounds):
    loss, accuracy, drifted_labels = evaluate(model, device, test_loader, drifted_labels, drift_probability)
    losses.append(loss)
    accuracies.append(accuracy)
    print(f"Round {round + 1}, Accuracy: {accuracy:.2f}%")

# Plot loss and accuracy vs AoI
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_rounds + 1), losses)
plt.title('Loss vs AoI')
plt.xlabel('Age of Information (AoI)')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_rounds + 1), accuracies)
plt.title('Accuracy vs AoI')
plt.xlabel('Age of Information (AoI)')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig('../data/test_acc_loss_vs_AoI.png')

# Print the percentage of labels that have drifted
total_drifted = sum(original_labels[i] != drifted_labels[i] for i in range(len(original_labels)))
print(f"Percentage of labels drifted after {num_rounds} rounds: {100 * total_drifted / len(original_labels):.2f}%")


# Use finite buffer size, throw away old data
# Either that or sample a constant window of data from all data