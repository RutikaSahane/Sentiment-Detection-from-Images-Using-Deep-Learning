import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time


model = tf.keras.models.load_model('model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(200, 200))  # Resize image
    x = image.img_to_array(img)  # Convert image to array
    x = np.expand_dims(x, axis=0)  # Expand dimensions to match model input
    images = np.vstack([x])
    val = model.predict(images)
    return "I am sad" if val == 1 else "I am happy"


def main():
    st.title("Sentiment Detector App")
    
    # Add an introductory text
    st.markdown("""
    This app detects the mood (happy or sad) based on an image you upload.
    Upload a photo to find out the sentiment!
    """)
    
    # File upload widget
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        
        # Save the uploaded image temporarily to disk for prediction
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        image.save(temp_path)
        
        if st.button("Predict Sentiment"):
            # Show progress bar while processing
            with st.spinner("Processing image..."):
                time.sleep(2)  # Simulate a delay for prediction processing
                
                # Get prediction
                prediction = predict_image(temp_path)
        
                # Show prediction result with customized styling
                st.subheader(f"Prediction: {prediction}")
                
                if prediction == "I am happy":
                    st.success("😊 You seem happy!")
                else:
                    st.error("😢 You seem sad!")
        
        else:
            st.info("Click the button to get the sentiment prediction.")

    else:
        st.warning("Please upload an image to proceed.")

# Run the app
if __name__ == '__main__':
    main()
        
        
        
       