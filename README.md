# 🧠 Semantic Segmentation API using NVIDIA SegFormer (ADE20K)

This project provides a Flask-based REST API for performing semantic segmentation on input images using NVIDIA's SegFormer B5 model fine-tuned on the ADE20K dataset. The API highlights specific classes such as **wall**, **floor**, and **ceiling**, and provides a fallback visualization for all detected classes if those are not found.

---

## 🚀 Features

- 🔍 Semantic segmentation using `nvidia/segformer-b5-finetuned-ade-640-640`
- 🎯 Highlights ADE20K classes: wall (12), floor (13), ceiling (95)
- 🖼️ Returns a colorized PNG image with segmented regions
- 🔄 Fallback visualization for all detected classes
- 🤖 Powered by Hugging Face Transformers and PyTorch

---

## 🛠️ Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/isurfer21/hf-segformer-api.git
   cd hf-segformer-api
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app**:
   ```bash
   python app.py
   ```

---

## 📡 API Usage

### Endpoint

```
POST /predict
```

### Request Format

Send a JSON payload with an image URL:

```json
{
  "image_url": "https://example.com/sample.jpg"
}
```

### Response

- Returns a PNG image with segmented regions.
- If wall, floor, or ceiling are not detected, all detected classes are visualized with random colors.

---

## 🧪 Example with `curl`

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://example.com/sample.jpg"}' --output segmented.png
```

---

## 📚 Model Reference

- SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
- Model on Hugging Face

---

## 📝 License

This project is licensed under the MIT License.
```
