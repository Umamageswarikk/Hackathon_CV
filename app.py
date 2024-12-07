import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import pandas as pd

# Load the trained YOLO model
model = YOLO(r'C:\MCA\5th trimester\computer vision\hackathon\yolov8n.pt')  # Replace with the correct path to your model

def run_inference(image_path):
    # Perform inference on the uploaded image
    results = model(image_path)
    return results

def resize_image(image, width=600, height=400):
    # Resize image to the given width and height
    return image.resize((width, height))

def main():
    # Page setup and custom title
    st.set_page_config(page_title="Trash Detection with YOLOv8", page_icon=":guardsman:", layout="wide")
    
    # Add custom CSS for styling
    st.markdown(
        """
        <style>
        .big-font {
            font-size: 40px !important;
            font-weight: bold;
            color: #2F4F4F;
        }
        .subtitle {
            font-size: 25px !important;
            color: #555;
        }
        .image-box {
            border: 4px solid #4CAF50;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .image-box:hover {
            transform: scale(1.05);
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.3);
        }
        .button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 20px;
            margin-top: 10px;
        }
        .button:hover {
            background-color: #45a049;
        }
        .header {
            text-align: center;
            padding: 20px;
        }
        .card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }
        .card-title {
            font-size: 22px;
            font-weight: bold;
            color: #2F4F4F;
        }
        .card-content {
            font-size: 18px;
            color: #555;
        }
        .sidebar .sidebar-content {
            padding: 20px;
        }
        .sidebar-title {
            font-size: 26px;
            font-weight: bold;
            color: #4CAF50;
        }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar with about section
    with st.sidebar:
        st.markdown("<h2 class='sidebar-title'>About</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            **Trash Detection with YOLOv8**
            
            This app utilizes the YOLOv8 object detection model to detect trash in images. Simply upload an image, 
            and the model will process it to identify objects that represent trash. 
            
            YOLO (You Only Look Once) is a state-of-the-art object detection algorithm that is fast and efficient, 
            making it suitable for real-time applications.
            
            **How to use:**
            - Upload an image.
            - Click on "Run Inference" to detect trash objects.
            - View the results with bounding boxes around detected trash items.
            """
        )

    # Page title and description
    st.markdown("<h1 class='big-font'>🚮 Trash Detection with YOLOv8</h1>", unsafe_allow_html=True)
    st.markdown("This app uses the YOLOv8 model to detect trash in images. Upload an image to get results!")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to an image
        image = Image.open(uploaded_file)
        
        # Resize the image to a smaller size (e.g., 600x400)
        resized_image = resize_image(image, width=600, height=400)

        # Display the uploaded image in a box with hover effects
        st.markdown("<div class='image-box'>", unsafe_allow_html=True)
        st.image(resized_image, caption="Uploaded Image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Convert the image to OpenCV format for YOLO
        opencv_image = np.array(resized_image)[:, :, ::-1]  # Convert RGB to BGR

        # Save the image temporarily to run inference
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_image_path = temp_file.name
            cv2.imwrite(temp_image_path, opencv_image)

        # Inference button
        if st.button("Run Inference"):
            with st.spinner('Running Inference...'):
                # Run the inference
                results = run_inference(temp_image_path)
            
            # Process results and display in a card
            st.markdown("<div class='card'><div class='card-title'>Processed Image with Bounding Boxes</div>", unsafe_allow_html=True)
            result_image = results[0].plot()  # Annotated image with bounding boxes
            st.image(result_image, caption="Processed Image with Bounding Boxes", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Extract bounding box information
            boxes = results[0].boxes.xywh  # [x_center, y_center, width, height]
            df = pd.DataFrame(boxes.cpu().numpy(), columns=["x_center", "y_center", "width", "height"])

            # Display the detection results in a DataFrame
            st.markdown("<div class='card'><div class='card-title'>Detection Results (Bounding Boxes)</div>", unsafe_allow_html=True)
            st.write(df)  # Show results in a nice table format
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Please upload an image to start detection.")

if __name__ == "__main__":
    main()


