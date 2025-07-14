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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image URL from the request
        data = request.json
        image_url = data['image_url']

        # Load the image
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

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

        # Print unique class indices predicted
        unique_classes = np.unique(predictions)
        print(f"Unique classes predicted: {unique_classes}")  # Debugging line

        # Create a color map for the segmentation
        color_map = np.array([
            [0, 0, 0],        # 0: Background (Black)
            [255, 0, 0],      # 1: Wall (Red)
            [0, 255, 0],      # 2: Floor (Green)
            [0, 0, 255],      # 3: Ceiling (Blue)
        ])

        # Create a colorized segmentation map
        segmented_image = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)

        for class_index in range(len(color_map)):
            segmented_image[predictions == class_index] = color_map[class_index]

        # Resize the segmented image to match the original image size
        segmented_image = Image.fromarray(segmented_image)
        segmented_image = segmented_image.resize(original_size, Image.BILINEAR)

        # Save the segmented image to a BytesIO object
        img_byte_arr = io.BytesIO()
        segmented_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return the image as a response
        return send_file(img_byte_arr, mimetype='image/png')

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
