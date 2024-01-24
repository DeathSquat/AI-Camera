import cv2
import face_recognition

# Load images of known criminals (provide image file paths)
criminal_image_paths = ["Dawood Ibrahim.jpg", "Osama Bin Laden.jpg"]

# Load and encode the face images of criminals
criminal_face_encodings = []
for image_path in criminal_image_paths:
    criminal_image = face_recognition.load_image_file(image_path)
    criminal_face_encoding = face_recognition.face_encodings(criminal_image)[0]
    criminal_face_encodings.append(criminal_face_encoding)

# Open the webcam (you can change the index to use a different camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the detected face matches any known criminal
        matches = face_recognition.compare_faces(criminal_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            name = "Alert: Possible Criminal Detected"

        # Draw a rectangle around the face and display the result
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Criminal Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
