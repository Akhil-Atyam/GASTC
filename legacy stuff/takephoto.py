import cv2

# Initialize the camera object (0 is usually the default webcam)
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not camera.isOpened():
    print("No Camera :(")
else:
    # Read a single frame from the camera
    return_value, image = camera.read()
    cv2.imwrite("webcam_photo.jpg", image)
# Release the camera resource
camera.release()