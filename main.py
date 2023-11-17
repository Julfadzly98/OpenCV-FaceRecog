import cv2
import face_recognition

# Load your images for recognition (multiple images for a single person)
your_images = [
    face_recognition.load_image_file("path_to_your_image_1.jpg"),
    face_recognition.load_image_file("path_to_your_image_2.jpg"),
    # Add more image paths if available
]

your_encodings = [face_recognition.face_encodings(img)[0] for img in your_images]

known_encodings = your_encodings  # All encodings for the same person

def recognize_face(frame):
    # Convert the image from BGR color to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        # See if the face matches any known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, label it with your name
        if True in matches:
            name = "Your Name"

        # Draw a box around the face and label it
        top, right, bottom, left = face_recognition.face_locations(rgb_frame)[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

    return frame

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Recognize faces in the video
    frame = recognize_face(frame)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
