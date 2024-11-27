# MNIST Digit Recognition with LeNet-5

## Project Overview
A PyTorch implementation of LeNet-5 for handwritten digit recognition using the MNIST dataset, with Flask API for inference.

## Prerequisites
- Python 3.8+
- PyTorch
- Conda (recommended)

## Setup with Conda
```bash
# Create conda environment
conda create -n mnist-lenet python=3.8
conda activate mnist-lenet

# Install dependencies
conda install pytorch torchvision -c pytorch
conda install flask pillow
```

## Project Structure
- `model.py`: LeNet-5 neural network architecture
- `dataset.py`: MNIST data loading and preprocessing
- `train.py`: Training script
- `inference.py`: Model inference utilities
- `main.py`: Training and Flask API endpoint

## Training
Run training to generate model weights:
```bash
python train.py
```
Weights saved to: `./weights/lenet5.pth`

## Inference
### CLI Prediction
```bash
python main.py --image_path /path/to/digit/image.png
```

### Flask API
Start server:
```bash
python main.py
```
Make POST request to `/infer` with JSON payload:
```json
{
  "img_path": "/path/to/digit/image.png"
}
```

## Features
- LeNet-5 CNN architecture
- MNIST dataset preprocessing
- Single/batch image prediction
- Top-k predictions
- Flask inference API

## Device Support
Automatically detects and uses CUDA if available, otherwise falls back to CPU.