# Librarys
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pickle
from PIL import Image

# this cache the model becouse we don't want model loads all the time
@st.cache
def load_model():
    model = pickle.load(open('poc_new.sav', 'rb'))
    return model

st.title("MNIST Digit Recognizer")

# Making two pages for diffrent purposes
purpose = st.selectbox('In version 1 you can uppload your own photos and in version two you can draw your own digits! Which version would you like to use?',("Version1", "Version2"))

if purpose == "Version1":
    model = load_model()
    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image to classify", type=(["jpg", "png"]))

    if uploaded_file is not None:
        
        # Read in the image file
        img = Image.open(uploaded_file)

        # Convert the image to grayscale
        img = img.convert('L')

        # Resize the image to 28x28 pixels
        img = img.resize((28, 28))

        # Flipping the background of the digits and colour of digits as same format as mnist
        img = np.bitwise_not(img)

        img = np.array(img, dtype=np.float64)
        
        # Flatten the image array for knn accept one dim array
        image_array = img.flatten()

        # Use the classifier to make a prediction
        prediction = model.predict([image_array])

        # Display the prediction to the user
        st.write("Predicted digit: ", str(prediction[0]))

if purpose == "Version2":
    
    model = load_model()
    SIZE = 200

    # Discription om canvas details
    canvas_result = st_canvas(
        fill_color="#ffffff",
        stroke_width=20,
        stroke_color='#ffffff',
        background_color="#000000",
        drawing_mode='freedraw',
        key="canvas",
    )

    if st.button('Predict'):
        
        # Resize the image to 28x28 pixels
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))

        # Resize the image again to 200x200 pixels to make the picture more visual for users
        img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        # Convert the image to grayscale
        gray =cv2.cvtColor(img_rescaling, cv2.COLOR_BGR2GRAY)
        
        # finding edge of objects in image
        edged = cv2.Canny(gray, 30, 150)

        # thresholding for having binary images as mnist dataset
        ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

        # finding counters in image for object detecting
        contours,hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        # finding bounding box
        x,y,w,h = cv2.boundingRect(contours[0])

        # cut the digit from canvas and centrlised it as mnist dataset(We want user be able to draw digits where ever they want
        # and that should't efect the model prediction even if we applied image augmentation)
        roi_cropped=img_rescaling[int(y)-20:int(y+h)+20, int(x)-20:int(x+w)+20]
        
        # Showing cropt image as same standard as mnist before fiting to the model
        st.write('Input Image')
        st.image(roi_cropped)

        # now we do the same actions to crop digits as well
        final_image = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2GRAY)
        new_array = cv2.resize(final_image, (28,28))
        image_array = np.array(new_array, dtype=np.float64)
        image_array = image_array.flatten()
        prediction = model.predict([image_array])
        st.write("Predicted digit: ", str(prediction[0]))