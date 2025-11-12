import torch 
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
from PIL import Image 
import pandas as pd 
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"available device: {device}")

# Fixed: Renamed to avoid conflict with transforms module
image_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Define CustomDataset class OUTSIDE if __name__ block
class CustomDataset(Dataset):
    def __init__(self, data_df, label_encoder, transform=None):
        self.dataframe = data_df
        self.transform = transform
        self.labels = torch.tensor(label_encoder.transform(data_df["labels"]), dtype=torch.long)
    
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128*16*16), 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        return x


# Windows multiprocessing fix - all execution code goes here
if __name__ == '__main__':
    image_path = []
    labels = []
     
    # file => label => image
    base_path = r"C:\Users\Benhur Lee\OneDrive\Desktop\pytorch\archive\afhq"#change it to your path
    for i in os.listdir(base_path):
        for label in os.listdir(os.path.join(base_path, i)):
            for image in os.listdir(os.path.join(base_path, i, label)):
                image_path.append(os.path.join(base_path, i, label, image))  
                labels.append(label)

    data_df = pd.DataFrame(zip(image_path, labels), columns=["image_path", "labels"])
    print(f"Total images loaded: {len(data_df)}")

    train = data_df.sample(frac=0.7, random_state=42)
    test = data_df.drop(train.index)
    val = test.sample(frac=0.5, random_state=42)
    test = test.drop(val.index)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    label_encoder = LabelEncoder()
    label_encoder.fit(data_df["labels"])
    print(f"Classes: {label_encoder.classes_}")

    # Create datasets
    train_dataset = CustomDataset(data_df=train, label_encoder=label_encoder, transform=image_transforms)      
    val_dataset = CustomDataset(data_df=val, label_encoder=label_encoder, transform=image_transforms)
    test_dataset = CustomDataset(data_df=test, label_encoder=label_encoder, transform=image_transforms)

    # Visualize some images
    n_rows = 3
    n_cols = 3
    f, axarr = plt.subplots(n_rows, n_cols, figsize=(10, 10))

    for row in range(n_rows):
        for col in range(n_cols):
            sample = data_df.sample(n=1)
            image = Image.open(sample["image_path"].iloc[0]).convert("RGB")
            axarr[row, col].imshow(image)
            axarr[row, col].set_title(sample["labels"].iloc[0])
            axarr[row, col].axis("off")
    plt.tight_layout()
    plt.show()

    LR = 1e-4
    BATCH_SIZE = 64
    EPOCHS = 50

    # Optimized DataLoaders for better GPU utilization
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=2, pin_memory=True)

    model = Net(num_classes=len(data_df['labels'].unique())).to(device)

    from torchsummary import summary
    summary(model, input_size=(3, 128, 128))

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    total_loss_train_plot = []
    total_loss_validation_plot = []
    total_acc_train_plot = []
    total_acc_validation_plot = []

    print("Starting training...")
    for epoch in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0
        total_loss_val = 0
        total_acc_val = 0

        model.train()
        for inputs, labels_batch in train_loader:
            # Move both inputs and labels to device
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = criterion(outputs, labels_batch)
            total_loss_train += train_loss.item()
            train_loss.backward()
            train_acc = (torch.argmax(outputs, dim=1) == labels_batch).sum().item()
            total_acc_train += train_acc
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                # Move both inputs and labels to device
                inputs = inputs.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(inputs)
                val_loss = criterion(outputs, labels_batch)
                total_loss_val += val_loss.item()
                val_acc = (torch.argmax(outputs, dim=1) == labels_batch).sum().item()
                total_acc_val += val_acc
        
        total_loss_train_plot.append(round(total_loss_train/len(train_loader), 4))
        total_loss_validation_plot.append(round(total_loss_val/len(val_loader), 4))
        total_acc_train_plot.append(round((total_acc_train/len(train_dataset))*100, 4))
        total_acc_validation_plot.append(round((total_acc_val/len(val_dataset))*100, 4))

        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss_train_plot[-1]}, Train Acc: {total_acc_train_plot[-1]}%, Val Loss: {total_loss_validation_plot[-1]}, Val Acc: {total_acc_validation_plot[-1]}%')

    print("Training completed!")
    
    # Save the model
    torch.save(model.state_dict(), 'animal_classifier.pth')
    print("Model saved as 'animal_classifier.pth'")