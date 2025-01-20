import base64
import datetime
import json
import threading
import time

import cv2
from openai import OpenAI

# Constants
CAMERA_WINDOW_NAME = 'Camera Feed'
SEND_INTERVAL = 2.5  # Interval in seconds
frame_result = None

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

    # Initialize OpenAI client
    client = OpenAI(
        api_key="asdsadd",
        base_url="http://localhost:1234/v1"
    )

    # Send frame to LLM
    print("Sending frame to LLM", str(time.time()))
    response = client.chat.completions.create(
        model="Qwen2-VL-7B-Instruct-GGUF",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": '''
        Detect the person mood, reply in the following response:
        {
          "angry_detected": true|false,
          "mood":  describe the mood in 1 or 2 word
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
    frame_result = json.dumps(json_result, indent=2).replace('.', '.\n')
    print("LLM Response:", frame_result)

def display_camera_feed():
    """
    Captures frames from the camera and displays them in a window.
    Sends frames to the LLM at regular intervals.
    """
    # Open the camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create a window to display the camera feed
    cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAMERA_WINDOW_NAME, 800, 600)

    last_send_time = time.time()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the result on the frame
        text = frame_result if frame_result else "Processing..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        font_scale = 0.75
        color = (255, 0, 0)  # Blue color in BGR
        thickness = 2

        # Split text into lines and display each line separately
        for i, line in enumerate(text.split('\n')):
            y = org[1] + i * 30  # Adjust the vertical position for each line
            frame = cv2.putText(frame, line, (org[0], y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Show the frame
        cv2.imshow(CAMERA_WINDOW_NAME, frame)

        # Send frame to LLM at regular intervals
        current_time = time.time()
        if current_time - last_send_time >= SEND_INTERVAL:
            threading.Thread(target=process_frame, args=(frame,)).start()
            last_send_time = current_time

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera_feed()