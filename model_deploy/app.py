'''
Mask application to determine if a uploaded image of a human face is wearing a mask or not.
Model is not 100% accurate, we calculate only a precision of 94% calculating a mask is on
and 100% calculating a mask is off. Overall, weighted average is 99% and macro average 97%.
Model has potential to predict multiple images at a time, but application only allows user to
upload one image at a time for now.

Functions:
model_predict(image_path, model):
    returns array containing an integer that corresponds to mask prediction.
index():
    renders template of index.
help():
    render template of help page.
contributors():
    render template of contributors page.
upload():
    allows user to upload image and calls model_predict on the image. Converts
    prediction to a human-readable string and displays it on the webpage.
'''

# coding=utf-8
import os
import numpy as np

# Tensorflow
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__, template_folder='templates')

# Model saved with Keras model.save()
MODEL_PATH = "models/mask_detect.h5"

# Load your trained model
model = load_model(MODEL_PATH)

print('Model loaded. Start serving...')

def model_predict(img_path, model):
    ''' Predicts if person in image is wearing a mask or not '''
    batch_size = 1
    images = [] # array of images to be inputted into model

    # Preprocessing the image
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # Preprocess the array
    images.append(image)
    images = np.array(images, dtype='float32')

    # Predict whether the image has a mask on or not
    pred = model.predict(images, batch_size=batch_size)

    print(
    '''
        Model is not 100% accurate, we calculate only a precision of 94% calculating a mask is on
        and 100% calculating a mask is off. Overall, weighted average is 99% and macro average
        of 97%. The left side is probability of mask (closer to one the higher the
        probability of mask). The right side is probability of no mask (closer to one the
        higher the probability of no mask).
    '''
    )

    # the first column is likelihood to have a mask on (closer to 1 or 100 the higher the likelihood
    # the person in the image has a mask on), second column is likelihood to not have a mask on (
    # closer to 1 or 100 the higher the likelihood the person in the image does not have a mask on)
    mask_pred = pred * 100 # Multiply by 100 to get percentages

    return mask_pred

@app.route('/')
def index():
    ''' Renders main page'''
    return render_template('index.html') # Main page

@app.route('/predict', methods=['GET'])
def predict():
    ''' Renders page that allows users to use the mask prediction model'''
    return render_template('predict.html')

@app.route('/help')
def helpview():
    ''' Renders a page to explain how to use the mode.'''
    return render_template('help.html')

@app.route('/contributors')
def contributors():
    ''' Renders a page giving information about the contributors.'''
    return render_template('contributors.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    ''' Allows user to upload images for predictions'''
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        # indexes of array are numpy.float32

        # return string as stating percentages of having a mask on or not
        mask_on = str(round(preds[0, 0], 2))
        mask_off = str(round(preds[0, 1], 2))
        print("Mask On: " + mask_on)
        print("Mask Off: " + mask_off)
        result = 'Mask: ' + mask_on + '%\t' + 'No Mask: ' + mask_off + '%'
        return result

    return None

if __name__ == '__main__':
    app.run(debug=True)
