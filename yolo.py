from ultralytics import YOLO
import cv2
# from ocrScan import scanOcrOnImage

# car_model = YOLO("yolo11n.pt")
model = YOLO("runs/detect/train15/weights/best.pt")


def camera():
    video(0)


def video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if video_path != 0:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (720, 1080))

        results = model(frame)

        for result in results:
            frame = result.plot()
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "assets/bawara.mp4"
    video(video_path)
    # camera()
