import cv2
import torch
import time
from ultralytics import YOLO

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Załaduj modele YOLO i przenieś je na odpowiednie urządzenie
car_model = YOLO("D:/AplikacjaSamochodowa/sieci/yolo11n.pt").to(DEVICE)
plate_model = YOLO("D:/AplikacjaSamochodowa/sieci/train9/weights/best.pt").to(DEVICE)
car_model.fuse()  # Fuzja warstw (jeśli obsługiwana) dla szybszej inferencji
plate_model.fuse()  # Fuzja warstw (jeśli obsługiwana) dla szybszej inferencji


# Klasy COCO odpowiadające pojazdom
CAR_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# Rysowanie ramki z opcjonalnym confidence
def draw_custom_box(frame, xyxy, label, confidence=None, color=(0, 255, 0), thickness=1, show_conf=False):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if confidence is not None and show_conf:
        label = f"{label} {confidence:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


# Detekcja pojazdów
def detect_cars(frame):
    car_result = car_model(frame, conf=0.5, verbose=False)[0]
    boxes = []
    for box in car_result.boxes:
        cls_id = int(box.cls[0])
        if cls_id in CAR_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0].item())
            boxes.append((x1, y1, x2, y2, CAR_CLASSES[cls_id], conf))
    return boxes


# Detekcja znaków/tablic w obrębie jednego pojazdu
def detect_plates_in_car(frame, car_box):
    x1, y1, x2, y2, _, _ = car_box
    # Pomijamy pojazdy, których szerokość lub wysokość jest mniejsza niż 100 pikseli.
    if (x2 - x1 < 100) or (y2 - y1 < 100):
        return []

    crop = frame[y1:y2, x1:x2]

    # Wymuszenie rozdzielczości wejściowej 640x640 — domyślnie model może oczekiwać 1280x1280
    # Można usunąć imgsz=640, by użyć domyślnej rozdzielczości zapisanej w modelu
    plate_result = plate_model(crop, imgsz=640, verbose=False)[0]

    plates = []
    for plate_box in plate_result.boxes:
        cls_id = int(plate_box.cls[0])
        if cls_id >= 2:
            px1, py1, px2, py2 = plate_box.xyxy[0]
            conf = float(plate_box.conf[0].item())
            abs_x1 = int(px1.item()) + x1
            abs_y1 = int(py1.item()) + y1
            abs_x2 = int(px2.item()) + x1
            abs_y2 = int(py2.item()) + y1
            label = plate_model.names[cls_id]
            plates.append(((abs_x1, abs_y1, abs_x2, abs_y2), label, conf))
    return plates


# Przetwarza pojedynczą klatkę: detekcja + rysowanie + statystyka
def process_frame(frame):
    car_boxes = detect_cars(frame)
    plate_char_count = 0

    for car_box in car_boxes:
        plates = detect_plates_in_car(frame, car_box)
        plate_char_count += len(plates)

        for (x1, y1, x2, y2), label, conf in plates:
            draw_custom_box(frame, (x1, y1, x2, y2), label, confidence=conf,
                            color=(0, 0, 255), thickness=1, show_conf=False)

        # Można odkomentować, aby narysować też pojazdy:
        # x1, y1, x2, y2, label, conf = car_box
        # draw_custom_box(frame, (x1, y1, x2, y2), label, confidence=conf,
        #                 color=(0, 255, 0), thickness=1, show_conf=True)

    return frame, len(car_boxes), plate_char_count


# Tryb testowy: bez przetwarzania, tylko obrót i FPS
def test_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow("ANPR Cascade", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ANPR Cascade",560,1000)
    cv2.moveWindow("ANPR Cascade",0,0)

    frame_count = 0
    start_global = time.perf_counter()

    while cap.isOpened():
        start_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 1)

        fps = 1 / (time.perf_counter() - start_time)
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10)

        cv2.imshow("ANPR Cascade", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    end_global = time.perf_counter()
    total_time = end_global - start_global
    avg_fps = frame_count / total_time if total_time > 0 else 0.0

    print("\n=== TEST FPS – TYLKO WYŚWIETLANIE ===")
    print(f"Liczba klatek: {frame_count}")
    print(f"Czas całkowity: {total_time:.2f} s")
    print(f"Średni FPS: {avg_fps:.2f}")


# Główna pętla przetwarzania wideo z wykrywaniem i pomiarem
def video_stream(video_path):
    cap = cv2.VideoCapture(video_path)

    paused = False
    step_mode = False
    processed_frame = None
    cv2.namedWindow("ANPR Cascade", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ANPR Cascade",560,1000)
    cv2.moveWindow("ANPR Cascade",0,0)

    frame_count = 0
    car_count = 0
    plate_char_count = 0

    start_global = time.perf_counter()

    while cap.isOpened():
        if not paused or step_mode:
            start_time = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 1)

            processed_frame, cars_this_frame, chars_this_frame = process_frame(frame)
            frame_count += 1
            car_count += cars_this_frame
            plate_char_count += chars_this_frame

            fps = 1 / (time.perf_counter() - start_time + 1e-6)
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10)

            if step_mode:
                paused = True
                step_mode = False
        else:
            if processed_frame is not None:
                cv2.putText(processed_frame, "STEP MODE / PAUSED", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        if processed_frame is not None:
            cv2.imshow("ANPR Cascade", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            step_mode = True
            paused = False
        elif key == 13:
            paused = False
            step_mode = False

    cap.release()
    cv2.destroyAllWindows()

    end_global = time.perf_counter()
    total_time = end_global - start_global
    avg_fps = frame_count / total_time if total_time > 0 else 0.0

    print("\n=== STATYSTYKI PRZETWARZANIA ===")
    print(f"Liczba przetworzonych klatek: {frame_count}")
    print(f"Czas całkowity: {total_time:.2f} s")
    print(f"Średni FPS: {avg_fps:.2f}")
    print(f"Łączna liczba samochodów: {car_count}")
    print(f"Łączna liczba wykrytych znaków (tablic): {plate_char_count}")


# Uruchomienie: test i pełne przetwarzanie
if __name__ == "__main__":
    video_path = "D:/AplikacjaSamochodowa/assets/bawara.mp4"
    test_fps(video_path)
    video_stream(video_path)
