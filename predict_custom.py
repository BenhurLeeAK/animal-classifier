import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define the same model architecture
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

# Image preprocessing
image_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load the trained model
print("Loading model...")
model_path = r"C:\Users\Benhur Lee\OneDrive\Desktop\pytorch\animal_classifier.pth"
class_names = ['cat', 'dog', 'wild']

model = Net(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded successfully!\n")

def predict_image(image_path):
    """Predict the class of a single image"""
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_tensor = image_transforms(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        return img, class_names[predicted_class], confidence, probabilities.cpu().numpy()
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None, None

def visualize_prediction(image_path):
    """Visualize the prediction with confidence scores"""
    img, predicted_class, confidence, probabilities = predict_image(image_path)
    
    if img is None:
        return
    
    # Get the index of the predicted class
    predicted_idx = class_names.index(predicted_class)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Original image with prediction
    ax1.imshow(img)
    ax1.axis('off')
    color = 'green' if confidence > 0.8 else 'orange' if confidence > 0.5 else 'red'
    ax1.set_title(f'Prediction: {predicted_class.upper()}\nConfidence: {confidence*100:.2f}%', 
                  fontsize=16, fontweight='bold', color=color, pad=20)
    
    # Plot 2: Probability bar chart
    colors_map = {'cat': '#FF6B6B', 'dog': '#4ECDC4', 'wild': '#95E1D3'}
    colors = [colors_map[name] for name in class_names]
    bars = ax2.barh(class_names, probabilities * 100, color=colors, edgecolor='black', linewidth=2)
    
    # Highlight the predicted class
    bars[predicted_idx].set_edgecolor('gold')
    bars[predicted_idx].set_linewidth(4)
    
    ax2.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])
    ax2.grid(axis='x', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        label = f'{prob*100:.1f}%'
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2, label,
                ha='left', va='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    filename = image_path.split('\\')[-1]
    print("="*60)
    print(f"IMAGE: {filename}")
    print("="*60)
    print(f"Predicted Class: {predicted_class.upper()}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nAll Class Probabilities:")
    for name, prob in zip(class_names, probabilities):
        bar = "█" * int(prob * 50)
        print(f"  {name:6s}: {prob*100:5.2f}% {bar}")
    print("="*60 + "\n")

def predict_multiple_images(image_paths):
    """Predict multiple images and display in a grid"""
    num_images = len(image_paths)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig = plt.figure(figsize=(6*cols, 6*rows))
    
    for idx, img_path in enumerate(image_paths):
        img, predicted_class, confidence, probabilities = predict_image(img_path)
        
        if img is None:
            continue
        
        # Create subplot for image
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img)
        ax.axis('off')
        
        color = 'green' if confidence > 0.8 else 'orange' if confidence > 0.5 else 'red'
        title = f'{predicted_class.upper()}\n{confidence*100:.1f}%'
        ax.set_title(title, fontsize=14, fontweight='bold', color=color, pad=10)
        
        # Print results
        filename = img_path.split('\\')[-1]
        print(f"{filename:30s} → {predicted_class:6s} ({confidence*100:5.1f}%)")
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("ANIMAL CLASSIFIER - CUSTOM IMAGE PREDICTOR")
    print("="*60)
    print("\nOptions:")
    print("1. Single image prediction")
    print("2. Multiple images prediction")
    print("3. Batch prediction from folder")
    print("="*60)
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        # Single image
        print("\nEnter the full path to your image:")
        print("Example: C:\\Users\\Benhur Lee\\OneDrive\\Desktop\\my_cat.jpg")
        image_path = input("Image path: ").strip().strip('"')
        
        if image_path:
            visualize_prediction(image_path)
        else:
            print("No image path provided!")
    
    elif choice == '2':
        # Multiple specific images
        print("\nEnter image paths (one per line). Press Enter twice when done:")
        image_paths = []
        while True:
            path = input().strip().strip('"')
            if not path:
                break
            image_paths.append(path)
        
        if image_paths:
            print("\nProcessing images...")
            predict_multiple_images(image_paths)
        else:
            print("No images provided!")
    
    elif choice == '3':
        # Batch from folder
        import os
        print("\nEnter the folder path containing images:")
        print("Example: C:\\Users\\Benhur Lee\\OneDrive\\Desktop\\test_images")
        folder_path = input("Folder path: ").strip().strip('"')
        
        if folder_path and os.path.exists(folder_path):
            # Get all image files
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            image_paths = []
            
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_paths.append(os.path.join(folder_path, file))
            
            if image_paths:
                print(f"\nFound {len(image_paths)} images. Processing...")
                predict_multiple_images(image_paths)
            else:
                print("No image files found in the folder!")
        else:
            print("Invalid folder path!")
    
    else:
        print("Invalid choice!")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

# ============================================================================
# DIRECT USAGE (uncomment to use directly in code)
# ============================================================================

# Single image prediction
# visualize_prediction(r"C:\Users\Benhur Lee\OneDrive\Desktop\my_image.jpg")

# Multiple images prediction
# image_list = [
#     r"C:\Users\Benhur Lee\OneDrive\Desktop\image1.jpg",
#     r"C:\Users\Benhur Lee\OneDrive\Desktop\image2.jpg",
#     r"C:\Users\Benhur Lee\OneDrive\Desktop\image3.jpg"
# ]
# predict_multiple_images(image_list)