import cv2
import numpy as np
from twilio.rest import Client
from twilio.twiml.voice_response import Gather, VoiceResponse
import pygame
import time  # Import the time module

# Load YOLO model and configuration
yolo_weights = "D:\RIC\yolov3.weights"
yolo_cfg = "D:\RIC\yolov3.cfg"
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
layer_names = net.getUnconnectedOutLayersNames()

# Initialize variables
crowd_threshold = 10 # Adjust as needed
alarm_message = "Crowd is within limits."
alarm_triggered = False

# Twilio configuration
account_sid = 'AC72412f3e10d993749cd4c8fed09025e4'
auth_token = 'bdd62544fbaab8f80ef4807780b24646'
twilio_phone_number = '+14105753927'
your_phone_number = '+917701847323'

# Initialize Twilio client
client = Client(account_sid, auth_token)

# Initialize pygame for sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("D:\RIC\Crowd\Alarm Audio.mp3")  # Replace with the path to your alarm sound file

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Function to make phone call with a short delay
def make_phone_call_with_delay():
    # Make a phone call using Twilio Voice API with a short delay
    time.sleep(5)  # Adjust the delay time as needed (5 seconds in this example)
    call = client.calls.create(
        to=your_phone_number,
        from_=twilio_phone_number,
        url="http://demo.twilio.com/docs/voice.xml"  # Replace with your TwiML URL for call instructions
    )
    print(f"Phone call initiated. SID: {call.sid}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(layer_names)

    # Apply Non-Maximum Suppression (NMS)
    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.2:  # Adjusted confidence threshold
                center_x, center_y, width, height = (obj[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype("int")
                x, y = int(center_x - width / 2), int(center_y - height / 2)

                boxes.append([x, y, x + width, y + height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Loop through the remaining boxes after NMS
    detected_boxes = [boxes[i] for i in indices]

    # Count persons in the frame
    persons_count = len(detected_boxes)

    # Display the frame with bounding boxes
    for box in detected_boxes:
        x, y, x_end, y_end = box
        cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 0), 2)

    cv2.putText(frame, f"Number of Persons: {persons_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Crowd Detection", frame)

    # Check if crowd count exceeds the threshold
    if persons_count > crowd_threshold and not alarm_triggered:
        alarm_triggered = True
        alarm_message = "Crowd limit exceeded! Take necessary actions."

        # Send SMS using Twilio
        message = client.messages.create(
            body=alarm_message,
            from_=twilio_phone_number,
            to=your_phone_number
        )

        # Initiate a phone call with a short delay
        make_phone_call_with_delay()

        # Play alarming sound
        pygame.mixer.Sound.play(alarm_sound)

    elif persons_count <= crowd_threshold and alarm_triggered:
        alarm_triggered = False
        alarm_message = "Crowd is within limits."

    # Print the alarm message
    print(alarm_message)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
