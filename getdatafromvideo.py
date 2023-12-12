import cv2

stream_url = 'http://192.168.1.13:81/stream'
print(f"Attempting to connect to: {stream_url}")

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Couldn't open the video stream.")
else:
    print("Successfully opened the video stream.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from the stream")
        break

    # Process the frame as needed (e.g., display or save it)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
