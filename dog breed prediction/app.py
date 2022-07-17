import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Load the Model
model = load_model('dogbreed.h5')

# List of breeds
breeds = ['scottish deerhound',
          'maltese dog',
          'beagle']


# Header of App
st.title('Dog Breed Prediction')
st.markdown('This is a Dog breed Predictor Application '
            'which will take an image of dog and '
            'predict the breed of the dog in '
            'following categories')
st.text('scottish deerhound,'
        ' maltese dog and beagle')


# Uploading the image
dog_img = st.file_uploader('Choose an image of dog', type='jpg')
submit = st.button('Predict')

# On pressing Predict Button
if submit:

    if dog_img is not None:

        # Load the Image & convert to numpy array
        img_bytes = np.array(bytearray(dog_img.read()), dtype = np.uint8)
        img_uploaded = cv2.imdecode(img_bytes, 1)

        # Displaying the Image
        st.image(img_uploaded, channels='BGR')
        # Resizing the Image
        img_uploaded = cv2.resize(img_uploaded, (200, 200))
        # Expanding dimensions of Image
        img_uploaded.shape = (1, 200, 200, 3)
        # Make prediction
        y_pred = model.predict(img_uploaded)

        st.title(str("The Predicted dog breed is "+breeds[np.argmax(y_pred)]))
