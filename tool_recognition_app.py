import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the model (ensure the correct path and extension are used)
model = load_model('E:/tensorflow website/streamlit/my_model.h5')

# Class labels (update according to your dataset's classes)
class_labels = ['Gasoline Can', 'Hammer', 'Pebble', 'Pliers', 'Rope', 'Screw Driver', 'Tool Box', 'Wrench']  # Add more as needed

# Function to preprocess image
def preprocess_image(image, target_size):
    image = image.resize(target_size)  # Resize image to match model input
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize if required
    return image

# Function to display tool descriptions (extend as needed)
def get_tool_description(predicted_class):
    descriptions = {
        "Gasoline Can": "A container specifically designed to safely store and transport gasoline or other flammable liquids. Typically made of metal or durable plastic, with a spout for pouring fuel into vehicles or machinery.",
        "Hammer": "A hand tool used for driving nails, breaking objects, or shaping materials. It usually has a metal head and a handle made of wood, plastic, or metal.",
        "Pebble": "A small, smooth stone typically found in rivers or on beaches. Pebbles are formed by natural erosion and are commonly used in landscaping or decorative purposes.",
        "Pliers": "A hand tool with two handles and jaws used for gripping, bending, or cutting materials like wires and small objects. Often used in electrical work and mechanical tasks.",
        "Rope": "A strong, thick cord made of twisted fibers or strands, used for tying, lifting, or securing objects. Commonly used in construction, climbing, or boating.",
        "Screw Driver": "A tool used for driving or removing screws. It has a handle and a shaft with a tip designed to fit into the head of a screw, which may be flathead, Phillips, or other types.",
        "Tool Box": "A container designed for organizing, storing, and transporting tools. It can be made of metal or plastic and usually contains compartments or trays for different types of tools.",
        "Wrench": "A tool used for gripping, fastening, or turning objects like nuts and bolts. Wrenches come in various types, such as adjustable, socket, or combination wrenches, and are essential in mechanical tasks.",
    }
    return descriptions.get(predicted_class, "No description available.")

# Streamlit app
st.title("Tool Recognition")
st.header("Upload an image of any tool to recognize it!")

# Upload image from file uploader or take a photo using camera
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp", "gif", "webp", "tiff", "svg"], help="Upload an image of the tool. You can also take a picture using your camera.")

if uploaded_file is not None:
    # Check if the uploaded file is a valid image
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Analyzing...")

        # Process the uploaded image
        processed_image = preprocess_image(image, target_size=(150, 150))  # Adjust size based on your model's input

        # Predict using the CNN model
        predictions = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(predictions)]  # Get the predicted class

        # Display the prediction and description
        st.success(f"Recognized Tool: {predicted_class}")
        st.write("**Tool Description:**")
        st.write(get_tool_description(predicted_class))

        # Additional options
        st.write("### Explore More")
        st.markdown(f"[🔍 Learn more about {predicted_class} on Google](https://www.google.com/search?q={predicted_class.replace(' ', '+')})")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload an image to get started.")
