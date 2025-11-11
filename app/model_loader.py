import tensorflow as tf
import logging
import h5py
from PIL import Image
import numpy as np
from io import BytesIO

logger = logging.getLogger("model_loader")
model = None
MODEL_PATH = "/app/model/densenet_tinyimagenet_light.h5"

def load_model_once():
    global model
    if model is not None:
        logger.info("✅ Model already loaded.")
        return model
    try:
        logger.info(f"🔹 Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        logger.info("✅ Model loaded successfully!")
    except TypeError as e:
        logger.warning(f"⚠️ TypeError: {e}")
        model = tf.keras.applications.DenseNet121(weights=None, input_shape=(224, 224, 3), classes=200)
        model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
        logger.info("✅ Model weights loaded manually.")
    return model

def predict_from_bytes(image_bytes: bytes):
    global model
    if model is None:
        model = load_model_once()
    img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    preds = model.predict(img_array)
    predicted_class = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return {"predicted_class": predicted_class, "confidence": confidence}
