import os
import argparse
from model.train import train
from model.inference import LeNet5Inferencer
from flask import Flask, request, jsonify
import logging

# Default weights path
WEIGHTS_PATH = "/weights/lenet5.pth"

def make_prediction(inferencer, image_path):
    """Make prediction for a single image."""
    if not os.path.exists(image_path):
        print(f"Image path {image_path} doesn't exist")
        return

    prediction, confidence = inferencer.predict_single(image_path, return_confidence=True)
    print(f"Predicted digit: {prediction} with confidence: {confidence:.2f}")
    
    # Get top-3 predictions
    top_predictions = inferencer.get_top_k_predictions(image_path, k=3)
    print("\nTop 3 predictions:")
    for digit, prob in top_predictions:
        print(f"Digit {digit}: {prob:.2f}")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='MNIST Digit Recognition Inference')
    parser.add_argument('--image_path', type=str, help='Path to the image for prediction')
    parser.add_argument('--serve', action='store_true', help='Start Flask server')
    args = parser.parse_args()

    # Check and train if weights don't exist
    if not os.path.exists(os.getcwd() + WEIGHTS_PATH):
        print("Weights do not exist, initializing training")
        train()

    # Initialize inferencer
    inferencer = LeNet5Inferencer("." + WEIGHTS_PATH)

    # Handle prediction if image path provided
    if args.image_path:
        make_prediction(inferencer, args.image_path)

    # Start Flask server if requested
    if args.serve:
        logging.basicConfig(level=logging.DEBUG)
        app = Flask(__name__)

        @app.route("/infer", methods=["POST"])
        def handler():
            if request.method == "POST":
                data = request.get_json()
                img_path = data.get("img_path")

                if not img_path:
                    return jsonify({"error": "img_path is required"}), 400
                
                make_prediction(inferencer, img_path)
                return jsonify({"message": f"Received img_path: {img_path}"})

        app.run(debug=True)

if __name__ == "__main__":
    main()