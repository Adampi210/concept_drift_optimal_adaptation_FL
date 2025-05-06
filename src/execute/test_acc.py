import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from fl_toolkit import *
import pandas as pd

# Assuming model classes are defined elsewhere
class PACSCNN(BaseModelArchitecture):
    def __init__(self, num_classes=7):
        super(PACSCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out

class PACSCNN_1(BaseModelArchitecture):
    def __init__(self, num_classes=7):
        super(PACSCNN_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(64, num_classes)
        self.apply(self.init_weights)

    @torch.no_grad()
    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class PACSCNN_2(BaseModelArchitecture):
    def __init__(self, num_classes=7):
        super(PACSCNN_2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PACSCNN_3(BaseModelArchitecture):
    def __init__(self, num_classes=7):
        super(PACSCNN_3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PACSCNN_4(BaseModelArchitecture):
    def __init__(self, num_classes=7):
        super(PACSCNN_4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PACSCNN_5(BaseModelArchitecture):
    def __init__(self, num_classes=7):
        super(PACSCNN_5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(512, 512, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PACSCNN_6(BaseModelArchitecture):
    def __init__(self, num_classes=7):
        super(PACSCNN_6, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEResidualBlock(64, 64, stride=1),
            SEResidualBlock(64, 64, stride=1),
            SEResidualBlock(64, 128, stride=2),
            SEResidualBlock(128, 128, stride=1),
            SEResidualBlock(128, 256, stride=2),
            SEResidualBlock(256, 256, stride=1),
            SEResidualBlock(256, 512, stride=2),
            SEResidualBlock(512, 512, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def filter_dataset_by_domain(dataset, domains):
    indices = [i for i, sample in enumerate(dataset) if sample[2] in domains]
    return Subset(dataset, indices)

def calculate_test_accuracy(model_architecture, model_path, domains, img_size, batch_size=256):
    model = model_architecture(num_classes=7).to(device)
    model.eval()

    data_handler = PACSDataHandler()
    data_handler.load_data()
    test_data = data_handler.test_dataset

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_data.transform = transform

    test_subset = filter_dataset_by_domain(test_data, domains)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_path}: {e}")

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels, _ = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def extract_domains_from_filename(filename):
    parts = filename.split('_')
    domains = []
    for part in parts:
        if part in ['photo', 'cartoon', 'sketch', 'art_painting']:
            domains.append(part)
    return domains

if __name__ == "__main__":
    # Define model architectures
    model_architectures = {
        'PACSCNN_1': PACSCNN_1,
        'PACSCNN_2': PACSCNN_2,
        'PACSCNN_3': PACSCNN_3,
        'PACSCNN_4': PACSCNN_4,
        'PACSCNN_5': PACSCNN_5,
        'PACSCNN_6': PACSCNN_6,
    }

    # Base directory for models
    model_dir = '/scratch/gautschi/apiasecz/models/concept_drift_models/'

    # Initialize results list
    results = []

    # Iterate over all model files in the directory
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.pth'):
            model_path = os.path.join(model_dir, model_file)
            model_name = model_file.split('_')[0] + '_' + model_file.split('_')[1]
            domains = extract_domains_from_filename(model_file)
            match = re.search(r'img_size_(\d+)', model_file)
            if match:
                img_size = int(match.group(1))
            else:
                print(f"Warning: Cannot find img_size in filename: {model_file}. Skipping.")
                continue

            try:
                model_architecture = model_architectures[model_name]
                accuracy = calculate_test_accuracy(model_architecture, model_path, domains, img_size)
                print(f"Calculated accuracy for {model_name} on {domains} (size {img_size}): {accuracy:.2f}%")
            except Exception as e:
                print(f"Error processing {model_path}: {e}")
                accuracy = 0.0

            results.append({
                'Model': model_name,
                'Domains': '_'.join(domains),
                'Image_Size': img_size,
                'Accuracy': accuracy
            })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    output_path = '../../data/results/model_accuracies.csv'
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Display the DataFrame sorted by domains and accuracy
    print("\nResults sorted by domains and accuracy:")
    print(df.sort_values(by=['Domains', 'Accuracy'], ascending=[True, False]))