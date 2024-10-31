# Dental Panoramic X-ray Image Segmentation

This project implements a deep learning model for segmenting mandible regions in dental panoramic X-ray images using PyTorch and the Segmentation Models PyTorch library. The goal is to automate the segmentation process, improving diagnostic accuracy and efficiency.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results Visualization](#results-visualization)

## Features
- **Model Architectures**: Supports multiple segmentation architectures including U-Net, U-Net++, and FPN.
- **Custom Dataset Handling**: Efficiently loads and preprocesses images and masks.
- **Performance Metrics**: Utilizes Dice Loss for training and evaluates model performance using Dice scores.
- **Visualization**: Plots training and validation loss and Dice scores over epochs.

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Image Processing Libraries**: OpenCV, NumPy
- **Visualization**: Matplotlib
- **Segmentation Models Library**: segmentation_models_pytorch

## Installation
To set up the project environment, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/dental-xray-segmentation.git
   cd dental-xray-segmentation 
2. **Create a virtual environment (optional but recommended)**:
   ```python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install the required packages**:
   ```pip install -r requirements.txt```

## Usage:
- To run the segmentation model, use the following command:
- python main.py --model <model_type> --epochs <num_epochs> --batch_size <batch_size> --learning_rate <learning_rate> --path_to_images <path_to_images> --path_to_masks <path_to_masks>

## Parameters:
--model: Model architecture to use (unet, unet++, or FPN).
--epochs: Number of training epochs (default: 50).
--batch_size: Number of samples per gradient update (default: 1).
--learning_rate: Learning rate for optimizer (default: 1e-5).
--path_to_images: Path to the directory containing input images.
--path_to_masks: Path to the directory containing segmentation masks.

## Training the Model
The training process involves splitting the dataset into training and validation sets. The model is trained using the specified parameters, and checkpoints are saved based on validation loss.
Example Command:
```python main.py --model unet --epochs 20 --batch_size 2 --learning_rate 1e-5 --path_to_images ./data/images --path_to_masks ./data/masks```
## Evaluation
During training, the model's performance is evaluated on a validation set. The validation loss and Dice score are printed after each epoch.
## Results Visualization
After training, loss and Dice score plots are generated to visualize model performance over epochs. This helps in assessing convergence and identifying potential overfitting.
Example Results:
Training and Validation Loss Plot:
![unet_train_validation_loss](https://github.com/user-attachments/assets/9350c062-707f-410f-93c3-651cb5e09760)
Training and Validation Dice Score Plot:
![unet_train_validtion_dice_score](https://github.com/user-attachments/assets/535bc003-4557-4d12-be86-08fb2bf951e9)



