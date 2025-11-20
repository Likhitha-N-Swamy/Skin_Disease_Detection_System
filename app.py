# app.py
import os
import base64
import io
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from xhtml2pdf import pisa

# Lazy-import TensorFlow only if model is present/needed
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image
    TF_AVAILABLE = True
except Exception:
    # We'll import when/if needed to avoid import-time failures in environments without TF
    TF_AVAILABLE = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Disease classes
CLASSES = [
    'Acne and Rosacea Photos',
    'Eczema Photos',
    'Melanoma Skin Cancer Nevi and Moles',
    'Hair Loss Photos Alopecia and other Hair Diseases',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Normal'
]

# Model path - read from environment (docker_start.sh downloads MODEL_URL -> models/best_model.h5)
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.h5")
model = None

def safe_load_model(path):
    """Attempt to load a Keras model if it exists and looks valid."""
    global model, TF_AVAILABLE
    if not os.path.exists(path) or os.path.getsize(path) < 1024 * 50:
        # file missing or suspiciously small — skip loading
        app.logger.warning("Model file not found or too small at %s. Skipping model load.", path)
        return None

    # If TF not imported earlier, import now
    if not TF_AVAILABLE:
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow.keras.applications.resnet50 import preprocess_input  # noqa: F401
            from tensorflow.keras.preprocessing import image  # noqa: F401
            TF_AVAILABLE = True
        except Exception as e:
            app.logger.exception("TensorFlow import failed at runtime: %s", e)
            return None

    try:
        # try loading the model
        loaded = tf.keras.models.load_model(path)
        app.logger.info("Successfully loaded model from %s", path)
        return loaded
    except Exception as e:
        app.logger.exception("Failed to load model from %s: %s", path, e)
        return None

# Try to load model at startup (if present)
model = safe_load_model(MODEL_PATH)

def generate_gradcam(img_array, model, last_conv_layer_name="conv5_block3_out"):
    """Generate Grad-CAM heatmap for a single image batch `img_array`."""
    # If TF or model aren't available, return a blank heatmap
    try:
        import tensorflow as tf
    except Exception as e:
        app.logger.warning("TensorFlow not available when generating gradcam: %s", e)
        return np.zeros((7, 7))  # small default

    try:
        # Try using the user-specified last conv layer; if it doesn't exist, pick the last conv2d-like layer
        if last_conv_layer_name not in [ly.name for ly in model.layers]:
            # find a likely conv layer name as fallback
            conv_layers = [ly for ly in model.layers if 'conv' in ly.name or 'conv2d' in ly.__class__.__name__.lower()]
            if conv_layers:
                last_conv_layer_name = conv_layers[-1].name
                app.logger.info("Falling back to conv layer: %s", last_conv_layer_name)

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            # choose the predicted class for the first (and only) batch element
            class_idx = tf.argmax(predictions[0])
            loss = tf.gather(predictions, indices=class_idx, axis=1)

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        denom = tf.math.reduce_max(heatmap)
        if denom is not None and denom != 0:
            heatmap = heatmap / denom
        heatmap_np = heatmap.numpy()
        return heatmap_np
    except Exception as e:
        app.logger.exception("Grad-CAM generation failed: %s", e)
        # return a zero heatmap on failure
        return np.zeros((7, 7))

def process_image(image_path):
    """Load image, resize to 224x224 and preprocess for ResNet-style input."""
    # import lazily
    try:
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.resnet50 import preprocess_input
    except Exception:
        # If TF not installed, attempt a pillow/numpy fallback (less ideal)
        from PIL import Image
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        x = np.array(img, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        # simple normalization fallback
        x = (x / 127.5) - 1.0
        return x

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def create_pdf(data, template_name):
    html = render_template(template_name, **data)
    pdf = io.BytesIO()
    pisa.CreatePDF(html, dest=pdf)
    pdf.seek(0)
    return pdf

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if we have a captured webcam image
    captured_image_data = request.form.get('captured_image')

    if captured_image_data and captured_image_data.startswith('data:image'):
        # Handle webcam capture - convert data URL to file
        from io import BytesIO

        # Extract the base64 data from the data URL
        header, encoded = captured_image_data.split(",", 1)
        image_data = base64.b64decode(encoded)

        # Create a file-like object
        file = BytesIO(image_data)
        file.filename = f"webcam_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        # Save the captured image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(image_data)

    elif 'image' in request.files:
        # Handle regular file upload
        file = request.files['image']
        if file.filename == '':
            return 'No selected file', 400

        # Save the uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    else:
        return 'No image uploaded', 400

    # Process image and get prediction
    try:
        img_array = process_image(filepath)
    except Exception as e:
        app.logger.exception("Image processing failed: %s", e)
        return "Image processing failed", 500

    # If model isn't loaded, return a friendly placeholder result
    if model is None:
        predicted_class = "Model not available"
        confidence = 0.0
        heatmap = np.zeros((224, 224), dtype=np.uint8)
        # create a dummy gradcam by resizing the zero matrix to 224x224
    else:
        try:
            predictions = model.predict(img_array)
            predicted_class_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = CLASSES[predicted_class_idx]
            heatmap = generate_gradcam(img_array, model)
        except Exception as e:
            app.logger.exception("Model prediction failed: %s", e)
            predicted_class = "Prediction error"
            confidence = 0.0
            heatmap = np.zeros((224, 224), dtype=np.uint8)

    # Overlay heatmap on original image
    try:
        img = cv2.imread(filepath)
        img = cv2.resize(img, (224, 224))
        # If heatmap is small (e.g., 7x7) resize it up
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        # Normalize and convert to color
        if heatmap_resized.dtype != np.uint8:
            heatmap_resized = np.uint8(255 * (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.ptp() + 1e-8))
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    except Exception as e:
        app.logger.exception("Failed to create gradcam overlay: %s", e)
        # fallback: just read & resize original image
        superimposed_img = cv2.resize(cv2.imread(filepath), (224, 224))

    # Save the Grad-CAM image
    gradcam_filename = f"gradcam_{os.path.basename(filepath)}"
    gradcam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    try:
        cv2.imwrite(gradcam_filepath, superimposed_img)
    except Exception as e:
        app.logger.exception("Failed to write gradcam image: %s", e)

    # Get patient information
    patient_data = {
        'name': request.form.get('name', 'Not provided'),
        'age': request.form.get('age', 'Not provided'),
        'gender': request.form.get('gender', 'Not provided'),
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'predicted_class': predicted_class,
        'confidence': f"{confidence * 100:.2f}%" if isinstance(confidence, float) else str(confidence),
        'image_path': filepath,
        'gradcam_path': gradcam_filepath
    }

    return render_template('result.html', **patient_data)

@app.route('/download_report', methods=['POST'])
def download_report():
    patient_data = {
        'name': request.form.get('name'),
        'age': request.form.get('age'),
        'gender': request.form.get('gender'),
        'date': request.form.get('date'),
        'predicted_class': request.form.get('predicted_class'),
        'confidence': request.form.get('confidence'),
        'image_path': request.form.get('image_path'),
        'gradcam_path': request.form.get('gradcam_path')
    }

    pdf = create_pdf(patient_data, 'pdf_template.html')
    return send_file(
        pdf,
        download_name='skin_disease_report.pdf',
        as_attachment=True,
        mimetype='application/pdf'
    )

@app.route('/healthz', methods=['GET'])
def healthz():
    """Health endpoint returns 200 when model loaded, 503 otherwise."""
    return jsonify(loaded=bool(model)), (200 if model else 503)

if __name__ == '__main__':
    # Run app locally for development. In production, use gunicorn (docker_start.sh).
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 8080))
    app.run(host=host, port=port, debug=True)
