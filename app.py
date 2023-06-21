# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# import base64
# st.set_page_config(layout="wide")
# # Path to your logo image file
# st.image("AutoEyeLogo.jpeg", width=200)
#
# # Add custom CSS for background image
# def set_background(png_file):
#     with open(png_file, "rb") as f:
#         image_data = f.read()
#         bin_str = base64.b64encode(image_data).decode("utf-8")
#
#     page_bg_img = '''
#     <style>
#     .stApp {
#         background-image: url("data:image/png;base64,%s");
#         background-size: cover;
#     }
#     </style>
#     ''' % bin_str
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#
#
# set_background("BabyBlue.jpeg")
# model = load_model(r'C:/Users/HANADI/Desktop/FAINALPROJECT/MobileNet_Car_detection.model')
#
# # Load the MobileNet model
# MobileNet = tf.keras.applications.mobilenet.MobileNet()
#
# st.title("Car Damage Detection")
#
# # Add a file uploader to allow the user to upload an image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
#
# # Check if a file has been uploaded
# if uploaded_file is not None:
#     # Load the image using keras.preprocessing
#     img = image.load_img(uploaded_file, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_batch = np.expand_dims(img_array, axis=0)
#     # Make a prediction using the loaded model
#     pred = model.predict(img_batch)
#
#     # Check if the car is damaged or not
# if pred[0][0] < pred[0][1]:
#     print( "The car is damaged")
# else:
#     print( "The car is not damaged")
#
#
#
#
#
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import base64
st.set_page_config(layout="wide")
# Path to your logo image file
st.image("AutoEyeLogo.jpeg", width=200)

# Add custom CSS for background image
def set_background(png_file):
    with open(png_file, "rb") as f:
        image_data = f.read()
        bin_str = base64.b64encode(image_data).decode("utf-8")

    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background("BabyBlue.jpeg")
model = load_model(r'C:/Users/HANADI/Desktop/CapstoneProject/MobileNet_Car_detection.model')

# Load the MobileNet model
MobileNet = tf.keras.applications.mobilenet.MobileNet()

st.title("Car Damage Detection")

# Add a file uploader to allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load the image using keras.preprocessing
    img = image.load_img(uploaded_file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    # Display the uploaded image
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Make a prediction using MobileNet
    pred = model.predict(img_batch)

    # Check if the car is damaged or not
    if pred[0][0] < pred[0][1]:
        st.write("The car is not damaged")
    else:
        st.write("The car is damaged")