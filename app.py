import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="artifacts/model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detection App",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add a sidebar
st.sidebar.title("Pneumonia Detection App")
st.sidebar.write("Upload a chest X-ray image to detect pneumonia.")

# Main title and description
st.title("🩺 Pneumonia Detection App")
st.markdown(
    """
    Welcome to the Pneumonia Detection App. This app uses a machine learning model to detect pneumonia from chest X-ray images.
    Please upload an image to get started.
    """
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing image...")

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.repeat(img_array, 3, axis=-1)  # Repeat the channel to match the expected input shape
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Check the input shape of the tflite model
    input_shape = input_details[0]['shape']
    if len(img_array.shape) != len(input_shape):
        st.error(f"Input shape mismatch. Expected {input_shape}, but got {img_array.shape}")
    else:
        # Set the tensor
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

        # Run inference
        with st.spinner('Running inference...'):
            interpreter.invoke()

        # Get the output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0][0]

        # Display the prediction
        st.subheader("Prediction Results")
        if prediction > 0.7:
            st.success("Pneumonia detected.")
        else:
            st.success("Normal, No pneumonia detected.")
else:
    st.info("Please upload a chest X-ray image to start the detection.")
