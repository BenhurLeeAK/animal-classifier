# animal-classifier
CNN-based animal image classifier using PyTorch

📊 Project Overview
This project implements a custom CNN architecture trained on the AFHQ (Animal Faces-HQ) dataset to achieve 97% validation accuracy in classifying animal images.
🎯 Model Performance

Training Accuracy: 100%
Validation Accuracy: 97.02%
Test Accuracy: ~97%
Model Size: 16.4 MB
Parameters: 4.3M trainable parameters

🏗️ Model Architecture
Input (3x128x128 RGB Image)
    ↓
Conv2D (3→32) + MaxPool + ReLU
    ↓
Conv2D (32→64) + MaxPool + ReLU
    ↓
Conv2D (64→128) + MaxPool + ReLU
    ↓
Flatten
    ↓
Linear (32768→128)
    ↓
Linear (128→3)
    ↓
Output (Cat/Dog/Wild)

🚀 Quick Start
Prerequisites
bashPython 3.8+
CUDA-capable GPU (recommended)
Installation

Clone the repository:

bashgit clone https://github.com/yourusername/animal-classifier.git
cd animal-classifier

Install dependencies:

bashpip install -r requirements.txt

Download the AFHQ dataset:

Place the dataset in archive/afhq/ directory
Structure: archive/afhq/{train|val|test}/{cat|dog|wild}/



📥 Download Pretrained Model
Download the pretrained model from [Google Drive/Releases] and place it in the project root directory as animal_classifier.pth.
🎓 Training
Train the model from scratch:
bashpython cnn.py
Training Configuration:

Batch Size: 64
Epochs: 50
Learning Rate: 1e-4
Optimizer: Adam
Loss Function: CrossEntropyLoss

Training Features:

Multi-worker data loading (4 workers)
GPU acceleration with CUDA
Automatic model checkpointing
Real-time training metrics

🧪 Testing & Evaluation
Evaluate the model and generate performance visualizations:
bashpython test.py
Outputs:

Confusion Matrix
Per-Class Accuracy
Loss vs Epochs Graph
Accuracy vs Epochs Graph
Prediction Confidence Distribution
Sample Predictions Grid

Generated Files:

model_performance.png - Comprehensive performance metrics
sample_predictions.png - Visual prediction examples

🔮 Making Predictions
Interactive Mode
bashpython predict_custom.py
Choose from three options:

Single Image - Predict one image at a time
Multiple Images - Predict several specific images
Batch Folder - Predict all images in a folder

Simple Single Image Mode
bashpython predict_single.py
Just enter the image path and get instant results!
Programmatic Usage
pythonfrom predict_custom import predict_image

img, predicted_class, confidence, probabilities = predict_image("path/to/image.jpg")
print(f"Prediction: {predicted_class} with {confidence*100:.2f}% confidence")
📊 Results
Training Progress
EpochTrain LossTrain AccVal LossVal Acc10.663271.82%0.385485.74%100.055198.19%0.115395.91%200.004399.99%0.116296.53%500.0000100.00%0.166897.02%
Per-Class Performance
ClassPrecisionRecallF1-ScoreCat~97%~97%~97%Dog~97%~97%~97%Wild~97%~97%~97%
🛠️ Technologies Used

PyTorch - Deep learning framework
torchvision - Image transformations
scikit-learn - Data preprocessing and metrics
matplotlib - Visualization
seaborn - Statistical visualizations
Pillow - Image processing
pandas - Data manipulation
numpy - Numerical operations

📦 Requirements
torch>=2.0.0
torchvision>=0.15.0
torchsummary>=1.5.1
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
Pillow>=9.0.0
pandas>=1.4.0
numpy>=1.21.0
🎨 Features

✅ Custom CNN architecture from scratch
✅ GPU acceleration support
✅ Multi-worker data loading
✅ Data augmentation ready
✅ Real-time training visualization
✅ Comprehensive model evaluation
✅ Interactive prediction interface
✅ Batch prediction support
✅ Confidence score visualization
✅ Beautiful result plots

📈 Future Improvements

 Add data augmentation (rotation, flip, color jitter)
 Implement dropout layers to reduce overfitting
 Add early stopping based on validation loss
 Experiment with transfer learning (ResNet, EfficientNet)
 Web interface using Gradio/Streamlit
 Model quantization for mobile deployment
 Support for more animal categories
 Real-time webcam prediction

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

AFHQ Dataset creators
PyTorch community
Inspiration from various computer vision tutorials

👤 Author
Benhur Lee.A.K

GitHub: BenhurLeeAK
LinkedIn: Benhur Lee.A.K

📧 Contact
For questions or feedback, please open an issue or contact a.k.benhurlee@gmail.com

⭐ Star this repo if you find it helpful!
Made with ❤️ and PyTorch
