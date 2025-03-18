from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import os
import tensorflow as tf

# ✅ Fully disable GPU (Fix cuDNN/cuBLAS errors)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

# ✅ Load models and tokenizer safely
MODEL_PATH = "model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
FEATURE_EXTRACTOR_PATH = "feature_extractor.keras"

try:
    caption_model = load_model(MODEL_PATH)
    feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print("✅ Models and tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    caption_model, feature_extractor, tokenizer = None, None, None

# Constants
MAX_LENGTH = 34
IMG_SIZE = 224

def generate_caption(image):
    """Generates a caption for an uploaded image."""
    if caption_model is None or feature_extractor is None or tokenizer is None:
        return "Error: Models not loaded properly."

    try:
        image = img_to_array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)

        image_features = feature_extractor.predict(image, verbose=0)

        in_text = "startseq"
        for _ in range(MAX_LENGTH):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
            yhat = caption_model.predict([image_features, sequence], verbose=0)
            yhat_index = np.argmax(yhat)
            word = tokenizer.index_word.get(yhat_index, None)
            if word is None:
                break
            in_text += " " + word
            if word == "endseq":
                break

        return in_text.replace("startseq", "").replace("endseq", "").strip()
    except Exception as e:
        return f"Error generating caption: {e}"

@app.route("/generate_caption", methods=["POST"])
def upload_and_generate():
    """Handles image upload and caption generation."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image = request.files["image"]
        image = load_img(image, target_size=(IMG_SIZE, IMG_SIZE))
        caption = generate_caption(image)
        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 500

@app.route("/")
def home():
    return "✅ Image Captioning API is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT variable
    app.run(host="0.0.0.0", port=port)  # Bind to all network interfaces

