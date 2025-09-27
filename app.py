import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def detect_deepfake(image_array):
    """
    A placeholder function to simulate deepfake detection.
    In a real-world application, this is where you would load and
    run your trained deep learning model (e.g., from TensorFlow or PyTorch).

    Args:
        image_array (np.array): The input image as a NumPy array.

    Returns:
        tuple: A dictionary with detection results and a boolean indicating if a face was detected.
    """
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    face_detected = len(faces) > 0
  
    if face_detected:
        mock_confidence = np.random.uniform(0.65, 0.99)
        is_deepfake = np.random.rand() > 0.8  # 20% chance of being fake
      
        if is_deepfake:
            mock_confidence = np.random.uniform(0.51, 0.7)

        detection_result = {
            "is_deepfake": is_deepfake,
            "confidence": mock_confidence
        }
    else:
        detection_result = {
            "is_deepfake": False,
            "confidence": 0.0
        }
        
    return detection_result, face_detected, faces

def process_image(uploaded_file):
    """
    Processes an uploaded image file, performs deepfake detection,
    and displays the result.
    """
    try:
        # Reading the image file and converting it to PIL Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image_array = np.array(image)
        
        # Displaying the uploaded image
        st.subheader("Uploaded Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Perform detection
        with st.spinner('Analyzing the image...'):
            result, face_detected, faces = detect_deepfake(image_array)
            
        st.success('Analysis Complete!')
        
        # Draw rectangles around detected faces
        if face_detected:
            st.subheader("Detected Faces")
            img_with_faces = image_array.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)
            st.image(img_with_faces, caption="Faces detected", use_column_width=True)

        # Display results based on the placeholder logic
        st.subheader("Detection Results")
        if face_detected:
            if result["is_deepfake"]:
                st.error("🚨 DeepFake Detected!")
                st.write(f"Confidence: {result['confidence']:.2f}")
                st.write("This image exhibits characteristics commonly associated with manipulated media. This may include inconsistencies in facial features, lighting, or subtle artifacts.")
            else:
                st.success("✅ Authentic Media Detected")
                st.write(f"Confidence: {result['confidence']:.2f}")
                st.write("The analysis indicates that this image is likely authentic and has not been synthetically generated or altered.")
        else:
            st.warning("⚠️ No Face Detected")
            st.write("The detector could not find a prominent face in the uploaded image. Please try again with a clear photo of a face.")
            
    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")

def process_video(uploaded_file):
    """
    A placeholder for processing video files.
    In a real app, this would involve frame extraction and analysis.
    """
    st.warning("Video processing is a placeholder in this demo.")
    st.write("To analyze a video, the application would extract frames and run the detection model on a subset of them. This is a resource-intensive task and requires a more complex setup.")
    st.write("Please upload a single image to see the detection functionality.")

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="DeepFake Face Detector", layout="centered")

    st.title("DeepFake Face Detector")
    st.markdown("""
    This application uses a machine learning approach to demonstrate the process of detecting deepfake images.
    Upload a single image (JPG, JPEG, PNG) to analyze it.

    **Disclaimer:** This is a demonstration app. The core detection model is a simplified placeholder and does not
    represent the accuracy of a real-world, production-level deepfake detector.
    """)
    
    st.sidebar.header("Upload Media")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image or video file...",
        type=["jpg", "jpeg", "png", "mp4", "mov"],
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == "image":
            process_image(uploaded_file)
        elif file_type == "video":
            process_video(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload an image or video.")

if __name__ == "__main__":
    main()
