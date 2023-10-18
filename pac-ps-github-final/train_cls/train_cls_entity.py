
from robustness.tools.breeds_helpers import make_entity13, print_dataset_info, ClassHierarchy
from robustness.tools.helpers import get_label_mapping
from robustness.tools import folder

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import random
import pdb
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,2,3'

# %%
# info_dir = './imagenet_class_hierarchy/modified'
info_dir = '/data1/wenwens/imagenet/imagenet_hierarchy/'
num_workers = 8

ret = make_entity13(info_dir)
print(ret)
superclasses, subclass_split, label_map = ret
batch_size = 64

data_dir = '/data1/wenwens/imagenet/imagenetv1/'

def train(model, trainloader, criterion, optimizer, device):
    train_loss = 0.0
    train_total = 0
    train_correct = 0

    # Switch to train mode
    model.train()

    for inputs, labels, _ in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update training loss
        train_loss += loss.item() * inputs.size(0)

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Compute average training loss and accuracy
    train_loss = train_loss / len(trainloader.dataset)
    train_accuracy = 100.0 * train_correct / train_total

    return model, train_loss, train_accuracy

def test(model, testloader, criterion, device):
    test_loss = 0.0
    test_total = 0
    test_correct = 0

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels, _ in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update test loss
            test_loss += loss.item() * inputs.size(0)

            # Compute test accuracy
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Compute average test loss and accuracy
    test_loss = test_loss / len(testloader.dataset)
    test_accuracy = 100.0 * test_correct / test_total

    return test_loss, test_accuracy

def train_epochs(model, trainloader, testloader, criterion, optimizer, device, num_epochs, save_interval=5):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model, train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, testloader, criterion, device)

        # train_losses.append(train_loss)
        # train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # print(f'Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%')
        print()

        if (epoch + 1) % save_interval == 0 or (epoch + 1) > 30:
          # Save the model and variables
          torch.save(model.state_dict(), f'resnet50_cifar10_{epoch+1}.pth')
          checkpoint = {
              'epoch': epoch + 1,
              'train_losses': train_losses,
              'train_accuracies': train_accuracies,
              'test_losses': test_losses,
              'test_accuracies': test_accuracies,
          }
          torch.save(checkpoint, f'resnet50_cifar10_variables_{epoch+1}.pth')

    return model, train_losses, train_accuracies, test_losses, test_accuracies

class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
            dataset (Dataset): The whole Dataset
            indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        # logger.debug(f"IDx recieved {idx}")
        # logger.debug(f"Indices type {type(self.indices[idx])} value {self.indices[idx]}")
        x = self.dataset[self.indices[idx]]

        if self.transform is not None:
            transformed_img = self.transform(x[0])

            return transformed_img, x[1], x[2:]

        else:
            return x

    @property
    def y_array(self):
        return self.dataset.y_array[self.indices]

    def __len__(self):
        return len(self.indices)

def dataset_with_targets(cls):
    """
    Modifies the dataset class to return target
    """

    def y_array(self):
        return np.array(self.targets).astype(int)

    return type(cls.__name__, (cls,), {"y_array": property(y_array)})

def get_entity13(
    root_dir=None,
    transforms=None,
    train_split=True,
):
    return get_breeds(
        root_dir,
        transforms,
        train_split,
    )

def get_breeds(
    root_dir=None,
    transforms=None,
    train_split=True,
):
    root_dir = f"{root_dir}/imagenet/"

    ret = make_entity13(f"{root_dir}/imagenet_hierarchy/")
    ImageFolder = dataset_with_targets(folder.ImageFolder)

    source_label_mapping = get_label_mapping("custom_imagenet", ret[1][0])

    sourceset = ImageFolder(
        f"{root_dir}/imagenetv1/train/", label_mapping=source_label_mapping
    )
    if train_split:
        source_idx = range(0, len(sourceset), 2)
    else:
        source_idx = range(1, len(sourceset)-1, 2)

    source_trainset = Subset(
        sourceset, source_idx, transform=transforms,
    )

    return source_trainset

if __name__ == '__main__':
    # Flag to control whether to run training or use saved fine-tuned model.
    train_model = True
    # Number of classes
    num_classes = 13

    # Import ResNet50 model pretrained on ImageNet
    model = models.resnet50(pretrained=True)
    print("Network before modifying conv1:")
    print(model)

    #Modify conv1 to suit CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Modify the final fully connected layer according to the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    print("Network after modifying conv1:")
    print(model)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    data_transforms = transforms.Compose([
        transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias='warn'),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])]
    )
    # only use even split for training
    full_dataset = get_entity13(
        root_dir='/data1/wenwens/',
        transforms=data_transforms,
        train_split = True,
    )
    train_loader_source = DataLoader(full_dataset, batch_size=64, shuffle=True, pin_memory=True)

    test_dataset = get_entity13(
        root_dir='/data1/wenwens/',
        transforms=data_transforms,
        train_split = False,
    )
    print("data loader generated")
    val_loader_source = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

    if train_model:
      # Train the model for 20 epochs, saving every 5 epochs
      num_epochs = 60
      save_interval = 1
      model, train_losses, train_accuracies, test_losses, test_accuracies = train_epochs(
          model, train_loader_source, val_loader_source, criterion, optimizer, device,
          num_epochs, save_interval)
      # Save the final trained model
      torch.save(model.state_dict(), f'resnet50_cifar10_final_model_epochs_{num_epochs}.pth')
    else:
      # Load the pre-trained model
      model.load_state_dict(torch.load('resnet50_cifar10_final_model_epochs_50.pth'))
      # Load the variables
      checkpoint = torch.load("resnet50_cifar10_variables.pth")
      epoch = checkpoint['epoch']
      train_losses = checkpoint['train_losses']
      train_accuracies = checkpoint['train_accuracies']
      test_losses = checkpoint['test_losses']
      test_accuracies = checkpoint['test_accuracies']
      classes = checkpoint['classes']
      model.to(device)
      model.eval()

