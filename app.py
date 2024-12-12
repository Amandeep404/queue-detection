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

# Person Tracker Class
class PersonTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.entry_times = {}
        self.exit_times = {}
        self.service_start_time = None
        self.service_time = 0
        self.served_person = None

    def track(self, frame, frame_id):
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        tracked_objects = self.tracker.update_with_detections(detections)

        current_ids = set()
        for track in tracked_objects:
            person_id = track[4]
            current_ids.add(person_id)
            if person_id not in self.entry_times:
                self.entry_times[person_id] = frame_id  # First appearance

            # Detect the front person for service start
            if not self.service_start_time and person_id not in self.exit_times:
                self.service_start_time = frame_id
                self.served_person = person_id

        # Log exit times for people not detected anymore
        for person_id in list(self.entry_times.keys()):
            if person_id not in current_ids and person_id not in self.exit_times:
                self.exit_times[person_id] = frame_id

        # Update service time
        if self.service_start_time and self.served_person in current_ids:
            self.service_time = frame_id - self.service_start_time

        return tracked_objects
    
def resize_to_16_9(frame):
    h, w, _ = frame.shape
    target_aspect_ratio = 10 / 12
    current_aspect_ratio = w / h
    
    if current_aspect_ratio < target_aspect_ratio:  # If the width is smaller compared to the target
        new_width = int(h * target_aspect_ratio)
        resized_frame = cv2.resize(frame, (new_width, h))
        # Padding on left and right to center the image
        pad_left = (new_width - w) // 2
        pad_right = new_width - w - pad_left
        padded_frame = cv2.copyMakeBorder(resized_frame, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:  # If the height is smaller compared to the target
        new_height = int(w / target_aspect_ratio)
        resized_frame = cv2.resize(frame, (w, new_height))
        # Padding on top and bottom to center the image
        pad_top = (new_height - h) // 2
        pad_bottom = new_height - h - pad_top
        padded_frame = cv2.copyMakeBorder(resized_frame, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return padded_frame

# Streamlit App Initialization
st.title("Real-Time People Detection in Video with Timings")

uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
tracker = PersonTracker()
box_annotator = sv.BoxAnnotator()

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    video_path = tfile.name
    video = cv2.VideoCapture(video_path)
    stframe = st.empty()
    st_count = st.empty()
    frame_id = 0

    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_skip = fps // 4  # Process 4 frames per second

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Resize the frame to fit a 16:9 aspect ratio
        frame = resize_to_16_9(frame)
        
        # Only process every frame_skip frames
        if frame_id % frame_skip == 0:
            # Track and annotate people
            tracked_objects = tracker.track(frame, frame_id)
            annotated_frame = box_annotator.annotate(scene=frame, detections=tracked_objects)

            # Add people count to the frame
            cv2.putText(
                annotated_frame,
                f"People Count: {len(tracked_objects)}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            
            # Display frame with updated count
            stframe.image(annotated_frame, channels="BGR")
            st_count.write(f"People Count: {len(tracked_objects)}")
        
        frame_id += 1

    video.release()
    st.success("Video processing complete!")

    # Removed the Final Logs section to stop displaying it
    # st.subheader("Final Logs")
    # for person_id, entry_time in tracker.entry_times.items():
    #     exit_time = tracker.exit_times.get(person_id, "Still in frame")
    #     service_time = tracker.service_time if tracker.served_person == person_id else "N/A"
    #     st.write(f"Person {person_id}: Entry Frame: {entry_time}, Exit Frame: {exit_time}, Service Time: {service_time}")

    st.write("Thank you for using the app!")
