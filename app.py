import os
from webbrowser import BackgroundBrowser
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from gevent.pywsgi import WSGIServer


ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# Define a flask app
app = Flask(__name__)

# Load your trained model
model = load_model(
    '/Users/yashsingh/Downloads/mynewproject_model.h5')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        if f and allowed_file(f.filename):
            img_path = file_path  # Use file_path instead of filename
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0

            # Make prediction using the loaded model
            pred = model.predict(img)
            # Get top 3 predicted classes
            pred_classes = pred.argsort()[0][-3:][::-1]
            # Get corresponding probabilities
            pred_probs = pred[0][pred_classes]

            # Define a list of classes
            classes = ['Arive-Dantu', 'Basale', 'Betel', 'Curry', 'Drumstick', 'Fenugreek', 'Guava', 'Hibiscus', 'Indian_Beech', 
                       'Indian_Mustard', 'Jackfruit', 'Jamaica_Cherry-Gasagase', 'Jamun', 'Jasmine', 'Karanda', 'Lemon',
                       'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal', 'Rasna', 'Rose_apple','Sandalwood',
                       'Tulsi', 'karanjT', 'neemT', 'peepalT']  # Replace with actual class names

            # Create a list of strings containing the predicted class names and corresponding probabilities
            result = [
                f'{classes[pred_classes[i]]}: {pred_probs[i]:.2%}' for i in range(3)]

            # Join the list elements with newline character
            result = '\n'.join(result)

            return result
        else:
            return "Invalid file format."



if __name__ == '__main__':
    app.run(debug=True, port=5001)
