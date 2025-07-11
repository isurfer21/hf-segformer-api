from flask import Flask, request, jsonify
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import torch

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
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Prepare the image for the model
        inputs = image_processor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the logits
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).squeeze().numpy()

        return jsonify(predictions.tolist())
    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
