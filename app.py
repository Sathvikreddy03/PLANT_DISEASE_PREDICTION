import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "supersecretkey"

# Define directories and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMAGE_SIZE = (150, 150)
MODEL_PATH = 'plant_disease_model.h5'
DATA_DIR = 'datasets'

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model once when the application starts
model = load_model(MODEL_PATH, compile=False)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess uploaded image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or could not be read.")
    image = cv2.resize(image, IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
    image = image / 255.0  # Normalize pixel values
    return image

# Get the class names based on the dataset directory structure
def get_class_names():
    class_names = os.listdir(os.path.join(DATA_DIR))
    return sorted(class_names)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the uploaded image
            image_data = preprocess_image(filepath)
            input_image = np.expand_dims(image_data, axis=0)

            # Make predictions using the loaded model
            predictions = model.predict(input_image)
            predicted_class = np.argmax(predictions[0])

            # Get the class name from the list of class names
            class_names = get_class_names()
            predicted_class_name = class_names[predicted_class]
            
            flash(f'Prediction: {predicted_class_name}')
            return redirect(url_for('result', prediction=predicted_class_name, filename=filename))
        
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file format. Allowed formats: png, jpg, jpeg, gif')
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    filename = request.args.get('filename')

    if not filename:
        filename = 'default.jpg'  # Provide a default image if not found

    return render_template('result.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
