import os
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = load_model('models/flower_model.h5')

# Define flower classes
flower_classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)
    
    return flower_classes[predicted_class], confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    error = None
    confidence = None
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            error = 'No file part'
            return render_template('index.html', error=error)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            error = 'No selected file'
            return render_template('index.html', error=error)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Make prediction
            try:
                prediction, confidence = predict_image(file_path)
            except Exception as e:
                error = f"Error processing image: {str(e)}"
                
    return render_template('index.html', prediction=prediction, filename=filename, error=error, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
