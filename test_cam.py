
import cv2

print("Starting Webcam Test...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Cannot open Webcam 0")

else:
    print("SUCCESS: Webcam opened!")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame")
            break

        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            "WEBCAM WORKING",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
