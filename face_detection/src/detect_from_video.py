# PavanMantha's Face detection
import cv2
import face_recognition

scale = 0.2
box_multiplier = 1 / scale

# Take input of the person name
name = input("Enter name:  ")

# Define a videocapture object
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()  # Reading Each frame

    # Resize the frame
    Current_image = cv2.resize(img, (0, 0), None, scale, scale)
    Current_image = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)

    # Find the face location and encodings for the current frame
    # 'cnn' runs on gpu and it is more accurate. change it to 'hog' is you want to run on cpu.
    face_locations = face_recognition.face_locations(Current_image, model='hog')
    face_encodes = face_recognition.face_encodings(Current_image, face_locations)

    # Find the matches for each detection
    for encodeFace, faceLocation in zip(face_encodes, face_locations):
        y1, x2, y2, x1 = faceLocation
        y1, x2, y2, x1 = int(y1 * box_multiplier), int(x2 * box_multiplier), int(y2 * box_multiplier), int(
            x1 * box_multiplier)

        # Draw rectangle around detected face

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 20), (x2, y2), (255, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

    # show the output
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

# release the camera object

cap.release()
cv2.destroyAllWindows()
