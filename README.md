# Flower Image Classification (Flask + TensorFlow)

A beginner-friendly web application that classifies flower images into five categories using a TensorFlow/Keras model served through a lightweight Flask UI. This document explains every moving part so you can set up the project, understand how it works end-to-end, and adapt it to your own needs.

## Table of Contents
1. [What this project does](#what-this-project-does)
2. [Tech stack](#tech-stack)
3. [Repository layout](#repository-layout)
4. [How the app works](#how-the-app-works)
5. [Prerequisites](#prerequisites)
6. [Setup and installation](#setup-and-installation)
7. [Model artifacts](#model-artifacts)
8. [Running the application](#running-the-application)
9. [Using the web UI](#using-the-web-ui)
10. [Training or updating the model](#training-or-updating-the-model)
11. [Production and deployment tips](#production-and-deployment-tips)
12. [Troubleshooting](#troubleshooting)

## What this project does
- Accepts an uploaded image (PNG, JPG, JPEG) from a browser.
- Validates the file extension and saves it to `static/uploads/`.
- Preprocesses the image to the MobileNetV2 input format (224x224, normalized).
- Uses a TensorFlow/Keras model to predict one of five flower classes: **daisy, dandelion, rose, sunflower, tulip**.
- Displays the predicted class and confidence back in the UI alongside the uploaded image.

## Tech stack
- **Framework:** Flask (Python web microframework)
- **Modeling:** TensorFlow / Keras (MobileNetV2 backbone with a custom classification head)
- **Templating:** Jinja2 (via Flask)
- **Styling:** Simple CSS (see `static/css/styles.css`)
- **Utilities:** Werkzeug (secure file names), NumPy (tensor manipulation), Pillow (via `tensorflow.keras.preprocessing.image`)

## Repository layout
```
.
├── app.py                 # Flask application, request handling, inference logic
├── download_model.py      # Utility to create/save the model file (MobileNetV2-based)
├── models/                # Expected location of model weights (flower_model.h5)
├── static/
│   ├── css/styles.css     # UI styling
│   └── uploads/           # Uploaded images are stored here at runtime
└── templates/
    └── index.html         # HTML template rendered by Flask
```

> Note: `models/flower_model.h5` is required at runtime. If it is missing, see [Model artifacts](#model-artifacts).

## How the app works
High-level request lifecycle in `app.py`:
1. **Upload handling** (`index` route, POST):
   - Ensures a file is present in `request.files['file']`.
   - Validates extension via `allowed_file(...)` (`png`, `jpg`, `jpeg` only).
   - Saves the file to `static/uploads/<secure_filename>`.
2. **Preprocessing** (`predict_image(...)`):
   - Loads the saved image, resizes to `224x224`.
   - Converts to a NumPy array, adds batch dimension, and applies `preprocess_input` from `tensorflow.keras.applications.mobilenet_v2`.
3. **Inference**:
   - Calls `model.predict(...)`.
   - Picks the class with the highest probability and computes a confidence percentage.
4. **Response rendering**:
   - Renders `templates/index.html` with the prediction, confidence, and the saved image path.

Supporting pieces:
- `allowed_file(filename)`: quick extension guard to avoid unexpected file types.
- `UPLOAD_FOLDER` (`static/uploads`): created at startup to ensure the path exists.
- `flower_classes`: hard-coded list that maps model outputs (index order) to human-readable labels.

## Prerequisites
- Python **3.9** or **3.10** (recommended for TensorFlow CPU builds).
- pip >= 22.
- A virtual environment tool (`venv` built into Python, or `conda` if you prefer).
- Sufficient disk space and memory to install TensorFlow (the CPU wheel is typically a few hundred MB).

## Setup and installation
1. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\\Scripts\\activate
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install flask tensorflow numpy pillow werkzeug
   ```
   - If you want GPU support, install a GPU-enabled TensorFlow wheel that matches your CUDA/CUDNN drivers.

3. **Prepare the `models/` directory**  
   Create it if it does not exist:
   ```bash
   mkdir -p models
   ```

## Model artifacts
The application expects a Keras model at `models/flower_model.h5` with output order:
`['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']`.

You have two options:

1. **Generate a baseline model (untrained head) with `download_model.py`**  
   This script builds a MobileNetV2 backbone (ImageNet weights, no top), adds:
   - `GlobalAveragePooling2D`
   - `Dense(1024, activation='relu')`
   - `Dense(5, activation='softmax')`

   It freezes the backbone and saves the compiled model:
   ```bash
   python download_model.py
   # -> creates models/flower_model.h5
   ```
   This produces a model ready for further fine-tuning; out of the box it is **not trained on flower data**, so predictions will be random until you train or replace the weights.

2. **Use your own trained weights**  
   - Train a model that matches the input shape `(224, 224, 3)` and the class order above.
   - Save it as `models/flower_model.h5` so `app.py` can load it at startup.

## Running the application
From the project root (with the virtual environment active and `models/flower_model.h5` present):
```bash
export FLASK_ENV=development   # optional: enables debug reloader
python app.py
```

The app starts on `http://127.0.0.1:5000/` by default. Set `FLASK_RUN_PORT` or `PORT` if you need a different port.

## Using the web UI
1. Open the app in your browser.
2. Click **“Choose an image”** and select a `.png`, `.jpg`, or `.jpeg` file of a flower.
3. Submit the form; the page will reload showing:
   - The uploaded image.
   - The predicted class and confidence percentage (0–100%).
4. If something fails (missing file, invalid type, model error), an error message is shown instead.

## Training or updating the model
While this repository focuses on serving the model, you can train or fine-tune it using any flower dataset (e.g., the “Oxford 102 Flowers” or TensorFlow Flowers dataset). A typical transfer-learning recipe:

1. **Prepare data**: Organize images per class or create a CSV with file paths and labels.
2. **Load the base model**: Same as `download_model.py` (MobileNetV2, `include_top=False`).
3. **Attach the head**: GAP → Dense(1024, relu) → Dense(5, softmax).
4. **Freeze most backbone layers** initially; train the head with a moderate learning rate.
5. **Unfreeze top backbone blocks** for fine-tuning with a lower learning rate.
6. **Augment data** (random flips/rotations/zoom) to improve generalization.
7. **Save weights** to `models/flower_model.h5` and restart the Flask app.

## Production and deployment tips
- **Disable debug**: Run with `FLASK_ENV=production` or set `debug=False` in `app.run`.
- **Use a WSGI server**: e.g., `gunicorn -w 2 -b 0.0.0.0:8000 app:app`.
- **Static file hosting**: Serve `static/` via a CDN or reverse proxy for efficiency.
- **Model loading**: For multi-worker setups, ensure each worker loads the model once at startup to avoid per-request overhead.
- **Uploads hygiene**: Periodically clean `static/uploads/` or store uploads in a temp directory if persistence is not required.

## Troubleshooting
- **`ModuleNotFoundError: No module named 'tensorflow'`**  
  Reinstall dependencies in an active virtual environment: `pip install tensorflow`.
- **`OSError: models/flower_model.h5 not found`**  
  Run `python download_model.py` or place your trained model in `models/flower_model.h5`.
- **GPU conflicts or long install times**  
  Use the CPU wheel of TensorFlow unless you specifically need GPU acceleration.
- **Incorrect predictions**  
  Ensure the class order matches `flower_classes` in `app.py`; retrain or fine-tune the model with labeled flower data.

---

![App UI](https://github.com/user-attachments/assets/b6d2efd7-d5df-437b-94ae-b8ae769b24fe)
![Classification Result](https://github.com/user-attachments/assets/caed4753-ba24-4c69-a99e-c491cab49e11)
