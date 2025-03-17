# # import sys
# # import os
# # import cv2
# # import time
# # import numpy as np
# # import streamlit as st

# # # Add the project root directory to sys.path
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # # Add local ultralytics to sys.path
# # local_ultralytics_path = r"E:\Capstone\yolov10\ultralytics"
# # if local_ultralytics_path not in sys.path:
# #     sys.path.insert(0, local_ultralytics_path)

# # from ultralytics import YOLOv10

# # # Streamlit app configuration
# # st.set_page_config(
# #     page_title="Cocoa Bean Quality Classification",
# #     page_icon="https://github.com/NephCgs/cocoabunch/blob/bbdbdebe0341071e330a907350f57ab5c0425723/logofin.png?raw=true",
# #     layout="wide",
# # )

# # # Custom CSS styling
# # st.markdown(
# #     """
# #     <style>
# #         html, body, .stApp {
# #             background-color: #b7733b;
# #             color: #dbe8ec;
# #         }
# #         [data-testid="stSidebar"] {
# #             background-color: #ffdd8b;
# #             color: #000000;
# #         }
# #         .stButton button {
# #             background-color: #e62954;
# #             color: white !important;
# #             font-weight: bold;
# #         }
# #         .footer {
# #             font-size: 14px;
# #             text-align: center;
# #             margin-top: 20px;
# #             color: #dbe8ec;
# #         }
# #     </style>
# #     """,
# #     unsafe_allow_html=True,
# # )

# # # Sidebar Configuration
# # st.sidebar.image(
# #     "https://github.com/NephCgs/cocoabunch/blob/bbdbdebe0341071e330a907350f57ab5c0425723/logofin.png?raw=true",
# #     use_container_width=True,
# # )
# # st.sidebar.markdown("### Select Your Model")
# # model_choice = st.sidebar.radio("Choose YOLOv10 Model:", ["Original YOLOv10", "YOLOv10-CARAFE"])
# # confidence = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.55, 0.05)

# # # State Management
# # if "run" not in st.session_state:
# #     st.session_state.run = False

# # start_button = st.sidebar.button("Start Classification")
# # stop_button = st.sidebar.button("Stop Classification")

# # st.session_state.run = start_button if start_button else st.session_state.run and not stop_button

# # # Webcam Handling
# # def list_available_cameras(max_cameras=5):
# #     cameras = []
# #     for i in range(max_cameras):
# #         cap = cv2.VideoCapture(i)
# #         if cap.isOpened():
# #             cameras.append(f"Camera {i}")
# #             cap.release()
# #     return cameras

# # available_cameras = list_available_cameras()
# # device_index = int(available_cameras[0].split(" ")[1]) if available_cameras else None

# # st.sidebar.markdown("### Webcam Device Selection")
# # camera_choice = st.sidebar.selectbox("Select Camera", available_cameras or ["No camera found"])

# # # Model Loading
# # model_path = (
# #     r"E:\Capstone\yolov10\final-original.pt"
# #     if model_choice == "Original YOLOv10"
# #     else r"E:\Capstone\yolov10\final-carafe.pt"
# # )

# # try:
# #     model = YOLOv10(model_path)
# #     st.success(f"Model Loaded: {model_choice}")
# # except Exception as e:
# #     st.error(f"Model Loading Error: {e}")
# #     st.stop()

# # # Helper Functions
# # def draw_detections(frame, results, model):
# #     """Draw bounding boxes and labels on the frame."""
# #     for result in results:
# #         if hasattr(result, "boxes"):
# #             boxes = result.boxes.xyxy.cpu().numpy()
# #             labels = result.boxes.cls.cpu().numpy()
# #             scores = result.boxes.conf.cpu().numpy()

# #             for box, label, score in zip(boxes, labels, scores):
# #                 x1, y1, x2, y2 = map(int, box)
# #                 class_name = model.names[int(label)]
# #                 confidence_text = f"{class_name}: {score:.2f}"
# #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #                 cv2.putText(frame, confidence_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# #     return frame

# # def calculate_fps(start_time, frame_count):
# #     """Calculate frames per second (FPS)."""
# #     elapsed_time = time.time() - start_time
# #     if elapsed_time > 0:
# #         return frame_count / elapsed_time
# #     return 0

# # # Real-Time Classification
# # if st.session_state.run and device_index is not None:
# #     cap = cv2.VideoCapture(device_index)
# #     if not cap.isOpened():
# #         st.error("Cannot access the webcam. Please check your camera connection.")
# #         st.session_state.run = False
# #     else:
# #         prev_time = time.time()
# #         frame_count = 0
# #         stframe = st.empty()
# #         fps_display = st.empty()

# #         while st.session_state.run:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 st.error("Failed to capture frame.")
# #                 break

# #             try:
# #                 results = model.predict(source=frame, conf=confidence)
# #                 frame = draw_detections(frame, results, model)
# #                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #                 frame_count += 1
# #                 fps = calculate_fps(prev_time, frame_count)

# #                 stframe.image(frame_rgb, channels="RGB", use_container_width=True)
# #                 fps_display.markdown(f"<h4 style='text-align: center;'>FPS: {fps:.2f}</h4>", unsafe_allow_html=True)

# #                 if time.time() - prev_time >= 1:
# #                     frame_count = 0
# #                     prev_time = time.time()

# #             except Exception as e:
# #                 st.error(f"Prediction Error: {e}")
# #                 break

# #         cap.release()
# # else:
# #     st.warning("Press 'Start Classification' to begin or check your camera settings.")

# import sys
# import os
# import cv2
# import time
# import numpy as np
# import streamlit as st

# # Add the project root directory to sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # Add local ultralytics to sys.path
# local_ultralytics_path = r"E:\Capstone\yolov10\ultralytics"
# if local_ultralytics_path not in sys.path:
#     sys.path.insert(0, local_ultralytics_path)

# from ultralytics import YOLOv10

# # Streamlit app configuration
# st.set_page_config(
#     page_title="Cocoa Bean Quality Classification",
#     page_icon="https://github.com/NephCgs/cocoabunch/blob/bbdbdebe0341071e330a907350f57ab5c0425723/logofin.png?raw=true",
#     layout="wide",
# )

# # Custom CSS styling
# st.markdown(
#     """
#     <style>
#         html, body, .stApp {
#             background-color: #b7733b;
#             color: #dbe8ec;
#         }
#         [data-testid="stSidebar"] {
#             background-color: #ffdd8b;
#             color: #000000;
#         }
#         .stButton button {
#             background-color: #e62954;
#             color: white !important;
#             font-weight: bold;
#         }
#         .footer {
#             font-size: 14px;
#             text-align: center;
#             margin-top: 20px;
#             color: #dbe8ec;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Sidebar Configuration
# st.sidebar.image(
#     "https://github.com/NephCgs/cocoabunch/blob/bbdbdebe0341071e330a907350f57ab5c0425723/logofin.png?raw=true",
#     use_container_width=True,
# )
# st.sidebar.markdown("### Select Your Model")
# model_choice = st.sidebar.radio("Choose YOLOv10 Model:", ["Original YOLOv10", "YOLOv10-CARAFE"])
# confidence = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.55, 0.05)

# # State Management
# if "run" not in st.session_state:
#     st.session_state.run = False

# start_button = st.sidebar.button("Start Classification")
# stop_button = st.sidebar.button("Stop Classification")

# st.session_state.run = start_button if start_button else st.session_state.run and not stop_button

# # Webcam Handling
# def list_available_cameras(max_cameras=5):
#     cameras = []
#     for i in range(max_cameras):
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             cameras.append(f"Camera {i}")
#             cap.release()
#     return cameras

# available_cameras = list_available_cameras()
# device_index = int(available_cameras[0].split(" ")[1]) if available_cameras else None

# st.sidebar.markdown("### Webcam Device Selection")
# camera_choice = st.sidebar.selectbox("Select Camera", available_cameras or ["No camera found"])

# # Model Loading
# model_path = (
#     r"E:\Capstone\yolov10\final-original.pt"
#     if model_choice == "Original YOLOv10"
#     else r"E:\Capstone\yolov10\final-carafe.pt"
# )

# try:
#     model = YOLOv10(model_path)
#     st.success(f"Model Loaded: {model_choice}")
# except Exception as e:
#     st.error(f"Model Loading Error: {e}")
#     st.stop()

# # Helper Functions
# def draw_detections(frame, results, model):
#     """Draw bounding boxes and labels on the frame."""
#     # Define a color map for classes
#     color_map = {
#         0: (0, 255, 0),    # Green for Class A
#         1: (255, 0, 0),    # Blue for Class B
#         2: (0, 0, 255),    # Red for Class C
#     }

#     for result in results:
#         if hasattr(result, "boxes"):
#             boxes = result.boxes.xyxy.cpu().numpy()
#             labels = result.boxes.cls.cpu().numpy()
#             scores = result.boxes.conf.cpu().numpy()

#             for box, label, score in zip(boxes, labels, scores):
#                 x1, y1, x2, y2 = map(int, box)
#                 class_name = model.names[int(label)]
#                 confidence_text = f"{class_name}: {score:.2f}"

#                 # Get color for the class, default to white if not in map
#                 color = color_map.get(int(label), (255, 255, 255))

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(
#                     frame, confidence_text, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
#                 )
#     return frame

# def calculate_fps(start_time, frame_count):
#     """Calculate frames per second (FPS)."""
#     elapsed_time = time.time() - start_time
#     if elapsed_time > 0:
#         return frame_count / elapsed_time
#     return 0

# # Real-Time Classification
# if st.session_state.run and device_index is not None:
#     cap = cv2.VideoCapture(device_index)
#     if not cap.isOpened():
#         st.error("Cannot access the webcam. Please check your camera connection.")
#         st.session_state.run = False
#     else:
#         prev_time = time.time()
#         frame_count = 0
#         stframe = st.empty()
#         fps_display = st.empty()

#         while st.session_state.run:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to capture frame.")
#                 break

#             try:
#                 results = model.predict(source=frame, conf=confidence)
#                 frame = draw_detections(frame, results, model)
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#                 frame_count += 1
#                 fps = calculate_fps(prev_time, frame_count)

#                 stframe.image(frame_rgb, channels="RGB", use_container_width=True)
#                 fps_display.markdown(f"<h4 style='text-align: center;'>FPS: {fps:.2f}</h4>", unsafe_allow_html=True)

#                 if time.time() - prev_time >= 1:
#                     frame_count = 0
#                     prev_time = time.time()

#             except Exception as e:
#                 st.error(f"Prediction Error: {e}")
#                 break

#         cap.release()
# else:
#     st.warning("Press 'Start Classification' to begin or check your camera settings.")


import sys
import os
import cv2
import time
import numpy as np
import streamlit as st
import zipfile  # Change from rarfile to zipfile

# Step 1: Extract ultralytics folder if it's not already extracted
if not os.path.exists("ultralytics"):
    with zipfile.ZipFile("ultralytics.zip", "r") as zip_ref:  # Use zipfile for .zip extraction
        zip_ref.extractall("ultralytics")

# Add the extracted ultralytics folder to sys.path
sys.path.insert(0, os.path.abspath("ultralytics"))

# Import ultralytics and model (after extraction)
from ultralytics import YOLOv10

# Streamlit app configuration
st.set_page_config(
    page_title="Cocoa Bean Quality Classification",
    page_icon="https://github.com/NephCgs/cocoabunch/blob/bbdbdebe0341071e330a907350f57ab5c0425723/logofin.png?raw=true",
    layout="wide",
)
# Custom CSS styling
st.markdown(
    """
    <style>
        html, body, .stApp {
            background-color: #b7733b;
            color: #dbe8ec;
        }
        [data-testid="stSidebar"] {
            background-color: #ffdd8b;
            color: #000000;
        }
        .stButton button {
            background-color: #e62954;
            color: white !important;
            font-weight: bold;
        }
        .footer {
            font-size: 14px;
            text-align: center;
            margin-top: 20px;
            color: #dbe8ec;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.image(
    "https://github.com/NephCgs/cocoabunch/blob/bbdbdebe0341071e330a907350f57ab5c0425723/logofin.png?raw=true",
    use_container_width=True,
)
st.sidebar.markdown("### Select Your Model")
model_choice = st.sidebar.radio("Choose YOLOv10 Model:", ["Original YOLOv10", "YOLOv10-CARAFE"])
confidence = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.55, 0.05)

# State Management
if "run" not in st.session_state:
    st.session_state.run = False

start_button = st.sidebar.button("Start Classification")
stop_button = st.sidebar.button("Stop Classification")

st.session_state.run = start_button if start_button else st.session_state.run and not stop_button

# Webcam Handling
def list_available_cameras(max_cameras=5):
    cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(f"Camera {i}")
            cap.release()
    return cameras

available_cameras = list_available_cameras()
device_index = int(available_cameras[0].split(" ")[1]) if available_cameras else None

st.sidebar.markdown("### Webcam Device Selection")
camera_choice = st.sidebar.selectbox("Select Camera", available_cameras or ["No camera found"])

# Model Loading
model_path = (
    "final-original.pt"
    if model_choice == "Original YOLOv10"
    else "final-carafe.pt"
)

try:
    model = YOLOv10(model_path)
    st.success(f"Model Loaded: {model_choice}")
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# Helper Functions
def draw_detections(frame, results, model):
    """Draw bounding boxes and labels on the frame."""
    color_map = {
        0: (0, 255, 0),    # Green for Class A
        1: (255, 0, 0),    # Blue for Class B
        2: (0, 0, 255),    # Red for Class C
    }

    for result in results:
        if hasattr(result, "boxes"):
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(label)]
                confidence_text = f"{class_name}: {score:.2f}"

                color = color_map.get(int(label), (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, confidence_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
    return frame

def calculate_fps(start_time, frame_count):
    """Calculate frames per second (FPS)."""
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0

# Real-Time Classification
if st.session_state.run and device_index is not None:
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        st.error("Cannot access the webcam. Please check your camera connection.")
        st.session_state.run = False
    else:
        prev_time = time.time()
        frame_count = 0
        stframe = st.empty()
        fps_display = st.empty()

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            try:
                results = model.predict(source=frame, conf=confidence)
                frame = draw_detections(frame, results, model)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_count += 1
                fps = calculate_fps(prev_time, frame_count)

                stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                fps_display.markdown(f"<h4 style='text-align: center;'>FPS: {fps:.2f}</h4>", unsafe_allow_html=True)

                if time.time() - prev_time >= 1:
                    frame_count = 0
                    prev_time = time.time()

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                break

        cap.release()
else:
    st.warning("Press 'Start Classification' to begin or check your camera settings.")
