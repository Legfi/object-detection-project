import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import cv2
import warnings
warnings.filterwarnings('ignore')

# Loading the model
@st.cache
def load_model():
    model = pickle.load(open("all_data.pkl", "rb"))
    return model
model = load_model()
# Creating the streamlit frame
st.title("Number prediction")
left_column, middle_column, right_column = st.columns(3)

# Creating the canvas
with left_column:
    st.header("Draw a number")
    st.subheader("0-9")
    canvas_result = st_canvas(
        #fill_color="rgb(0,0,0)",
        stroke_width=7,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        width=75,
        height=150,
        drawing_mode="freedraw",
        key="canvas"
    )
predict_button = st.button("Predict digit")
if predict_button:
    im = Image.fromarray(canvas_result.image_data.astype('uint8'), mode="RGBA")
    im.save("user_input.png", "PNG")
    # Getting the values as numpy array from canvas
    input_numpy_array = np.array(canvas_result.image_data)

    # Get the PIL image
    input_image = Image.fromarray(input_numpy_array.astype('uint8'))
    input_image.save('user_input.png')

    # Convert it to grayscale
    input_image_gs = input_image.convert('L')
    input_image_gs_np = np.asarray(input_image_gs.getdata()).reshape(75,150)

    # Create a temporary image for opencv to read it
    input_image_gs.save('temp_for_cv2.jpg')
    image = cv2.imread('temp_for_cv2.jpg', 0)

    # Create a boundingbox, getting the x and y coordinates as well as width and height. 
    height, width = image.shape
    x,y,w,h = cv2.boundingRect(image)

    # Create new blank image and adding a mask of black borders to make it look like mnist. And the adding the "Region of Interest" to the masked image 
    ROI = image[y:y+h, x:x+w]
    mask = np.zeros([ROI.shape[0]+20,ROI.shape[1]+20])
    width, height = mask.shape
    x = width//2 - ROI.shape[0]//2
    y = height//2 - ROI.shape[1]//2
    mask[y:y+h, x:x+w] = ROI
    output_image = Image.fromarray(mask) # mask has values in 0-255

    # Resize and saving the picture being used for prediction
    compressed_output_image = output_image.resize((28,28), Image.BILINEAR)
    plt.imshow(compressed_output_image) 
    plt.savefig("bbox.png")

    # Displaying the picture used for prediction
    with middle_column:
        st.header("Image used for prediction")
        st.image("bbox.png")

    # Reshape and transform the bbox.png to make it work with our scaler and model
    image = np.asarray(compressed_output_image.getdata())
    image = image.reshape(1,-1)

    # Predicting and displaying our prediction
    with right_column:
        prediction = model.predict(image)
        st.header("Predicted digit")
        st.subheader(prediction[0])