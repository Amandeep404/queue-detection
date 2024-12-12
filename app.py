import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import tempfile
import time

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO("best-queue-yolo.pt")

model = load_model()

# Function to process frames and detect people
def detect_people(frame):
    result = model(frame)[0]  # Perform inference
    detections = sv.Detections.from_ultralytics(result)  # Extract detections
    return detections

# Streamlit UI
st.title("Real-Time People Detection in Video")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    video = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    # Initialize annotator
    box_annotator = sv.BoxAnnotator()

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Detect people
        detections = detect_people(frame)
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        
        # Display frame
        stframe.image(annotated_frame, channels="BGR", caption=f"People detected: {len(detections)}")
        
        # Update number of people in real-time
        st.write(f"Number of people in queue: {len(detections)}")
        time.sleep(1)  # Delay to simulate real-time processing
    
    video.release()
    st.success("Video processing complete!")
