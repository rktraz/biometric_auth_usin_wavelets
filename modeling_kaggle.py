
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# In[ ]:


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In[ ]:


def get_dataloaders(path, batch_size=32, train_size=0.8):
    # Define transformation to be applied to the scaleogram images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Reshape images to 224x224
        transforms.ToTensor(),           # Convert images to tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize image tensors
    ])

    # Create a dataset object
    dataset = ImageFolder(path, transform=transform)
    # Split dataset into training and validation sets
    train_size_ = int(train_size * len(dataset))
    val_size_ = len(dataset) - train_size_
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size_, val_size_])

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return [train_loader, val_loader]

# In[ ]:


import os
import pickle
import torch
import matplotlib.pyplot as plt
import zipfile


def save_model_and_results(method: str, model, results, num_classes):
    folder_name = f'{method}_{model._get_name()}_{num_classes}classes'

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    train_losses, train_accuracies, val_losses, val_accuracies = results

    with open(os.path.join(folder_name, 'train_losses.pickle'), 'wb') as file:
        pickle.dump(train_losses, file)
    with open(os.path.join(folder_name, 'val_losses.pickle'), 'wb') as file:
        pickle.dump(val_losses, file)

    with open(os.path.join(folder_name, 'train_accuracies.pickle'), 'wb') as file:
        pickle.dump(train_accuracies, file)
    with open(os.path.join(folder_name, 'val_accuracies.pickle'), 'wb') as file:
        pickle.dump(val_accuracies, file)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(folder_name, f'{model._get_name()}.pth'))

    # Visualize and save plots
    epochs = range(1, len(val_losses) + 1)

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(folder_name, 'loss_plot.png'))
    plt.show()
    plt.close()

    # Plot training and validation accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Train')
    plt.plot(epochs, val_accuracies, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(folder_name, 'accuracy_plot.png'))
    plt.show()
    plt.close()

    # Create a zip file of the folder
    zip_filename = f'{folder_name}.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_name):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, folder_name))

    print(f"Successfully saved the model, results, and plots in {folder_name}!")
    print(f"Created a zip file: {zip_filename}")


# In[ ]:


def train(model, num_epochs, dataloaders, criterion, optimizer):
    train_loader, val_loader = dataloaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize lists to store training and validation metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100.0 * val_correct / len(val_loader.dataset)

        # Update metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        print(time.strftime("%H:%M:%S", time.localtime()))
    return [train_losses, train_accuracies, val_losses, val_accuracies]

# In[ ]:


train_dataset_path = "/kaggle/input/new-wavelet-datasets-for-my-diploma/train_dataset_mexh_method/train_dataset_mexh_method"
test_dataset_path = "/kaggle/input/new-wavelet-datasets-for-my-diploma/test_dataset_mexh_method 2/test_dataset_mexh_method"

# In[ ]:


import time


NUM_EPOCHS = 25


for dataset_path in [train_dataset_path]: # in case you want to train different models on different datasets
    print("\n\n" + 64*"*")
    print(time.strftime("%H:%M:%S", time.localtime()))
    print("Processing", dataset_path.split("/")[-1], "data")
    print("\n\n" + 64*"*" + "\n\n")
    dataloaders = get_dataloaders(dataset_path, batch_size=32, train_size=0.8)

    # Load the pre-trained efficientnet_b0 model
    model = torchvision.models.efficientnet_b0(pretrained=True)

    # Modify the classifier for the number of classes in your dataset
    num_classes = len(dataloaders[1].dataset.dataset.classes)
    model.classifier[1] = nn.Linear(1280, num_classes)

    # Move the model to the device
    model = model.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    results = train(model=model, num_epochs=NUM_EPOCHS, dataloaders=dataloaders,
                    criterion=criterion, optimizer=optimizer)
    method = dataset_path.split("/")[-1].split("_")[-2]
    save_model_and_results(method, model, results, num_classes)

    print(time.ctime())
    print("This iteration ended successfully.")

# In[ ]:


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Reshape images to 224x224
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize image tensors
])

# Create a dataset object
test_dataset = ImageFolder(test_dataset_path, transform=transform)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Set the model to evaluation mode
model.eval()

# Load the saved model state dictionary
# model.load_state_dict(torch.load("/kaggle/working/mexh_EfficientNet_50classes/EfficientNet.pth"))
model.load_state_dict(torch.load("/kaggle/input/new-wavelet-datasets-for-my-diploma/mexh_EfficientNet_50classes/EfficientNet.pth"))

# Test the model on the new data
total_correct = 0
total_samples = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)  # Move inputs to the appropriate device (e.g., GPU)
        labels = labels.to(device)  # Move labels to the appropriate device

        outputs = model(inputs)  # Forward pass
        _, predicted = torch.max(outputs, dim=1)  # Get the predicted labels

        total_samples += labels.size(0)  # Increment the total number of samples
        total_correct += (predicted == labels).sum().item()  # Increment the correct predictions
        
        all_predictions.extend(predicted.cpu().numpy())  # Collect all predicted labels
        all_labels.extend(labels.cpu().numpy())  # Collect all ground truth labels

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy:.2%}")

# Create confusion matrix
confusion = confusion_matrix(all_labels, all_predictions)

# Display confusion matrix
plt.figure(figsize=(8, 8))
plt.imshow(confusion, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.colorbar()

# Add labels to each cell in the confusion matrix
num_classes = len(class_names)
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=90)
plt.yticks(tick_marks, class_names)

# Add count values to the cells
thresh = confusion.max() / 2.
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(confusion[i, j], 'd'), horizontalalignment="center",
                 color="white" if confusion[i, j] > thresh else "black")

plt.tight_layout()
plt.show()
