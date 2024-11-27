import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from model.model import LeNet5
import os
from typing import Union, List, Tuple


class LeNet5Inferencer:
    def __init__(self, model_path: str, device: str=None):
        """
        Initialize the inference with a trained model
        Args:
            model_path: Path to the saved model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize the model
        self.model = LeNet5().to(self.device)

        # Load model weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            raise FileNotFoundError(f"No model found at {model_path}")
        
        # Set model to evaluation mode
        self.model.eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # Convert to grayscale
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess a single image for inference

        Args: 
            image_path: Path to the image file
        
        Returns:
            Preprocessed image tensor
        """
        # Load and preprocess image
        image = Image.open(image_path)
        image_tensor = self.transform(image)
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor.to(self.device)
    

    def predict_single(self, image_path: str, return_confidence: bool = False) -> Union[int, Tuple[int, float]]:
        """
        Make prediction on a single image

        Args: 
            image_path: Path to the image file
            return_confidence: Whether to return confidence score

        Returns:
            Predicted class (and confidence score if return_confidence True)
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            # Get model prediction
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

            # Get predicted class and confidence
            confidence, predicted = torch.max(probabilities, 1)

            if return_confidence:
                return predicted.item(), confidence.item()
            
            return predicted.item()
        
    def predict_batch(self, image_paths: List[str], return_confidence: bool = False) -> Union[List[int], List[Tuple[int, float]]]:
        """
        Make predictions on a batch of images
        Args:
            image_paths: List of paths to image files
            return_confidence: Whether to return confidence scores

        Returns:
            List of predicted classes (and confidence scores if return_confidence=True)
             
        """
        # Preprocess all images
        batch_tensors = []
        for image_path in image_paths:
            image_tensor = self.preprocess_image(image_path)
            batch_tensors.append(image_tensor)

        # Stack all tensors into a single batch
        batch = torch.cat(batch_tensors, dim=0)

        with torch.no_grad():
            # Get model predictions
            outputs = self.model(batch)
            probabilities = F.softmax(outputs, dim=1)

            # Get predicted classes and confidences
            confidence, predicted = torch.max(probabilities, 1)

            if return_confidence:
                return list(zip(predicted.cpu().numpy(), confidence.cpu().numpy()))
            return predicted.cpu().tolist()
        
    def get_top_k_predictions(self, image_path: str, k: int=3) -> List[Tuple[int, float]]:
        """
        Get top-k predictions for a single image
        Args:
            image_path: Path to the image file
            k: Number of top predicitons to return 

        Returns:
            List of (class, probability) tuples for top-k predictions
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            # Get model prediciton
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

            # Get top-k predicitons
            top_prob, top_class = torch.topk(probabilities, k)

            return list(zip(top_class[0].cpu().numpy(), top_prob[0].cpu().numpy()))
        





