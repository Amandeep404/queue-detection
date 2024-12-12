import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Page configuration
st.set_page_config(page_title="Real-time Object Detection App", page_icon=":movie_camera:", layout="wide")

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO('best-queue-yolo.pt')  # Replace with your YOLO model file
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Real-time video processing function for uploaded video
def process_uploaded_video(uploaded_file, model, selected_objects, min_confidence):
    # Create a placeholder for video display
    stframe = st.empty()
    people_count_placeholder = st.empty()

    # Temporary file for the uploaded video
    with st.spinner("Processing video..."):
        with open("temp_uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.getvalue())

    # Open the uploaded video
    video_stream = cv2.VideoCapture("temp_uploaded_video.mp4")
    
    if not video_stream.isOpened():
        st.error("Error: Could not open uploaded video.")
        return

    # Process video frame by frame
    try:
        while True:
            ret, frame = video_stream.read()
            if not ret:
                break  # End of video
            
            # Run YOLO model for object detection
            results = model(frame, stream=True)
            
            person_count = 0  # Initialize people count for the current frame

            # Draw detections on the frame
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    object_name = model.names[cls]

                    # Filter detections based on selected objects and confidence
                    if object_name in selected_objects and confidence > min_confidence:
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{object_name} {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Count the number of people detected
                        if object_name == 'person':
                            person_count += 1

            # Overlay the people count on the frame
            cv2.putText(frame, f"People Count: {person_count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Stream the updated frame in Streamlit
            stframe.image(frame, channels="BGR", use_column_width=True)
            
            # Update the people count dynamically in Streamlit
            people_count_placeholder.markdown(f"### 👥 People Count: **{person_count}**")

        video_stream.release()
        st.success("Video processing complete!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        video_stream.release()
        cv2.destroyAllWindows()

# Main function
def main():
    # Load YOLO model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check the model path.")
        return

    # Get object names from the model
    object_names = list(model.names.values())

    # Title
    st.title("🎥 Real-time Object Detection for Uploaded Videos")

    # Sidebar options
    st.sidebar.header("Settings")
    default_objects = ['person'] if 'person' in object_names else [object_names[0]]
    selected_objects = st.sidebar.multiselect("Choose objects to detect", object_names, default=default_objects)
    min_confidence = 0.4  # Default minimum confidence score

    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

    # Start detection if the file is uploaded
    if uploaded_file is not None:
        st.info("Playing uploaded video with real-time object detection...")
        process_uploaded_video(uploaded_file, model, selected_objects, min_confidence)

    st.sidebar.info("SIH2024- The Debuggers")

if __name__ == "__main__":
    main()