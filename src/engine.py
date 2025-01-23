import base64
import datetime
import json
import logging
import os
import threading
import time

import cv2
from dotenv import load_dotenv, find_dotenv

from src.utils.llm_client import LLMClient
from src.utils.object_detection_client import ObjectDetectionClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv(find_dotenv())

# Constants
CAMERA_WINDOW_NAME = 'Camera Feed'
SEND_INTERVAL = 2.5  # Interval in seconds
frame_result = None
frame_result_lock = threading.Lock()

def send_frame_to_llm(frame):
    """
    Encodes the frame to base64 and sends it to the LLM for analysis.

    Args:
        frame (numpy.ndarray): The frame to be sent.

    Returns:
        tuple: The response from the LLM and the time the frame was analyzed.
    """
    analyze_time = datetime.datetime.now().time()

    # Encode frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # Send frame to LLM
    logging.info("Sending frame to LLM %s", str(time.time()))
    client = LLMClient().get_client()
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "Qwen2-VL-7B-Instruct-GGUF"),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": '''
        Detect the person mood, reply in the following response in json format:
        {
          "angry_detected": true|false,
        }
        '''},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpg_as_text}"}}]
        }],
    )

    return response.choices[0].message.content, analyze_time

def process_frame(frame):
    """
    Processes the frame by resizing it and sending it to the LLM.

    Args:
        frame (numpy.ndarray): The frame to be processed.
    """
    global frame_result

    # Resize frame to reduce data size
    resized_frame = cv2.resize(frame, (320, 240))

    # Send frame to LLM and get the result
    result, analyze_time = send_frame_to_llm(resized_frame)
    json_result = json.loads(result)
    json_result["analyze_time"] = str(analyze_time)
    json_result["current_time"] = str(datetime.datetime.now().time())

    # Format the result for display
    with frame_result_lock:
        frame_result = json.dumps(json_result, indent=2).replace('.', '.\n')
    logging.info("LLM Response: %s", frame_result)

def start():
    """
    Captures frames from the camera and displays them in a window.
    Sends frames to the LLM at regular intervals.
    """
    # Get video source (RTSP or local camera)
    cap = get_video_source()

    if not cap.isOpened():
        logging.error("Error: Could not receive any video input.")
        return

    # Create a window to display the camera feed
    cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAMERA_WINDOW_NAME, 800, 600)

    last_send_time = time.time()

    # Detect objects in the frame
    is_tracking_enabled = bool(os.getenv("IS_TRACKING_ENABLED", "True"))
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Could not read frame.")
            break

        # Track individuals in the frame
        if is_tracking_enabled:
            frame_with_detections = track_individual(frame)
        else:
            frame_with_detections = frame

        # Send frame to LLM at regular intervals
        current_time = time.time()
        if current_time - last_send_time >= SEND_INTERVAL:
            threading.Thread(target=process_frame, args=(frame_with_detections,)).start()
            last_send_time = current_time

        display_camera_feed(frame_with_detections)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def display_camera_feed(frame_with_detections):
    """
    Displays the camera feed with the processed frame results.

    Args:
        frame_with_detections (numpy.ndarray): The frame with detection results.
    """
    # Display the result on the frame
    with frame_result_lock:
        text = frame_result if frame_result else "Processing..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    font_scale = 1
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 2
    # Split text into lines and display each line separately
    for i, line in enumerate(text.split('\n')):
        y = org[1] + i * 30  # Adjust the vertical position for each line
        frame_with_detections = cv2.putText(frame_with_detections, line, (org[0], y), font, font_scale, color,
                                            thickness, cv2.LINE_AA)
    # Show the frame
    cv2.imshow(CAMERA_WINDOW_NAME, frame_with_detections)

def get_video_source():
    """
    Gets the video source based on environment variables.

    Returns:
        cv2.VideoCapture: The video capture object.
    """
    if bool(os.getenv("USE_LOCAL_CAMERA", "True")):
        cap = cv2.VideoCapture(int(os.getenv("EXTERNAL_VIDEO_SOURCE", 0)))
    else:
        cap = cv2.VideoCapture(os.getenv("EXTERNAL_VIDEO_SOURCE"))
    return cap

def track_individual(frame):
    """
    Tracks individuals in the frame using the object detection client.

    Args:
        frame (numpy.ndarray): The frame to be processed.

    Returns:
        numpy.ndarray: The frame with detection results.
    """
    detection_client = ObjectDetectionClient()
    detections = detection_client.detect_and_track(frame)
    # Draw bounding boxes on the frame
    frame_with_detections = detection_client.draw_tracks(frame, detections)
    return frame_with_detections