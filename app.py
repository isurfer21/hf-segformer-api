from flask import Flask, request, send_file, jsonify
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import torch
import numpy as np
import io

app = Flask(__name__)

# Load the model and image processor
image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")

# ADE20K class indices for wall, floor, and ceiling
ADE_CLASSES = {
    "wall": 12,
    "floor": 13,
    "ceiling": 95
}

# Color map for visualization
COLOR_MAP = {
    ADE_CLASSES["wall"]: [255, 0, 0],     # Red
    ADE_CLASSES["floor"]: [0, 255, 0],    # Green
    ADE_CLASSES["ceiling"]: [0, 0, 255]   # Blue
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image URL from the request
        data = request.json
        image_url = data['image_url']

        # Load the image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")

        # Store the original image size
        original_size = image.size

        # Prepare the image for the model
        inputs = image_processor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the logits and predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).squeeze().numpy()

        # Print unique class indices predicted for debugging
        unique_classes = np.unique(predictions)
        print(f"Unique classes predicted: {unique_classes}")  # Debugging line

        # Create a colorized segmentation map
        segmented_image = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)

        # Check if any target classes are present
        target_classes_found = any(cls in unique_classes for cls in COLOR_MAP.keys())

        if target_classes_found:
            # Assign colors for wall, floor, and ceiling
            for class_index, color in COLOR_MAP.items():
                segmented_image[predictions == class_index] = color
        else:
            # Fallback: visualize all detected classes with random colors
            print("Target classes not found. Visualizing all detected classes.")
            np.random.seed(42)
            for cls in unique_classes:
                color = np.random.randint(0, 255, size=3)
                segmented_image[predictions == cls] = color

        # Convert to PIL image and resize
        segmented_image = Image.fromarray(segmented_image, mode="RGB")
        segmented_image = segmented_image.resize(original_size, Image.BILINEAR)

        # Save to BytesIO
        img_byte_arr = io.BytesIO()
        segmented_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return the image
        return send_file(img_byte_arr, mimetype='image/png')

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
