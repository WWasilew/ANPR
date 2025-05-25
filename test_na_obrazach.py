import cv2
import torch
import random
import os
import numpy as np
from ultralytics import YOLO

# Ścieżki i konfiguracja
IMG_DIR = "C:/STUDIA/VI_SEM/PIT/ApplikacjaSamochodowa/assets/tablice"
MODEL_PATH = "runs/detect/train16/weights/best.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ścieżka do poprawnych zdjęć
SAVE_IMAGE_DIR_OK = "C:/STUDIA/VI_SEM/PIT/ApplikacjaSamochodowa/assets/saved/poprawne/images"
SAVE_LABEL_DIR_OK = "C:/STUDIA/VI_SEM/PIT/ApplikacjaSamochodowa/assets/saved/poprawne/labels"
os.makedirs(SAVE_IMAGE_DIR_OK, exist_ok=True)
os.makedirs(SAVE_LABEL_DIR_OK, exist_ok=True)

# Ścieżka do zdjęć wymagających poprawy
SAVE_IMAGE_DIR = "C:/STUDIA/VI_SEM/PIT/ApplikacjaSamochodowa/assets/saved/do_poprawy/images"
SAVE_LABEL_DIR = "C:/STUDIA/VI_SEM/PIT/ApplikacjaSamochodowa/assets/saved/do_poprawy/labels"
os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)
os.makedirs(SAVE_LABEL_DIR, exist_ok=True)

# Parametry
conf_thresh = 0.1
nms_thresh = 0.5
imgsz = 1408
scale_factor = 8
label_margin = 80
agnostic_nms = True  # False = NMS klasowy, True = NMS globalny

# Wczytanie modelu
plate_model = YOLO(MODEL_PATH).to(DEVICE)
plate_model.fuse()  # Fuzja warstw (jeśli obsługiwana) dla szybszej inferencji

# Wczytanie obrazów
image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if not image_files:
    raise FileNotFoundError("Brak obrazów w katalogu.")


def load_random_image():
    path = os.path.join(IMG_DIR, random.choice(image_files))
    img = cv2.imread(path)
    return path, img


def detect_plate_and_chars(image):
    results = plate_model(image, imgsz=imgsz, conf=conf_thresh, iou=nms_thresh, agnostic_nms=agnostic_nms, verbose=False)[0]
    plate_box = None
    char_boxes = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = plate_model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if label == "number_plate":
            plate_box = (x1, y1, x2, y2)
        else:
            char_boxes.append((x1, y1, x2, y2, label))

    # Zapamiętaj ostatnie wykrycia
    global last_detected_boxes, last_detected_labels
    last_detected_boxes = results.boxes.xywhn.tolist()
    last_detected_labels = results.boxes.cls.tolist()

    return plate_box, char_boxes


def prepare_plate_view(original_image, plate_box, char_boxes):
    base_image = original_image.copy()  # kopia do rysowania ramki
    x1, y1, x2, y2 = plate_box
    cv2.rectangle(base_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plate_crop = original_image[y1:y2, x1:x2]
    plate_big = cv2.resize(plate_crop, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    h_big, w_big = plate_big.shape[:2]
    final_plate = np.full((h_big + label_margin, w_big, 3), 255, dtype=np.uint8)
    final_plate[label_margin:, :] = plate_big

    for (cx1, cy1, cx2, cy2, label) in char_boxes:
        if cx1 >= x1 and cy1 >= y1 and cx2 <= x2 and cy2 <= y2:
            rel_x1 = int((cx1 - x1) * scale_factor)
            rel_y1 = int((cy1 - y1) * scale_factor) + label_margin
            rel_x2 = int((cx2 - x1) * scale_factor)
            rel_y2 = int((cy2 - y1) * scale_factor) + label_margin

            cv2.rectangle(final_plate, (rel_x1, rel_y1), (rel_x2, rel_y2), (0, 0, 255), 2)
            cv2.putText(final_plate, label, (rel_x1, rel_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

    return final_plate, base_image


def save_labeled_image(path, original, labels, boxes, save_img_dir, save_lbl_dir, message):
    base_name = os.path.splitext(os.path.basename(path))[0]
    img_save_path = os.path.join(save_img_dir, base_name + ".jpg")
    label_save_path = os.path.join(save_lbl_dir, base_name + ".txt")

    # Zapisz obraz
    cv2.imwrite(img_save_path, original)
    if os.path.exists(path):
        os.remove(path)

    # Zapisz etykiety
    with open(label_save_path, "w") as f:
        for cls_id, (x, y, w, h) in zip(labels, boxes):
            f.write(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print(f"{message}: {base_name}")
    return load_random_image()


# Inicjalizacja
image = None
final_plate = None
key = 32
original = None

# Pętla główna
while True:
    if key == 32 or image is None or key in [ord('q'), ord('a'), ord('w'), ord('s'), ord('e'), ord('d'), ord('r'), ord('z'), ord('p'), ord('o')]:
        if key == 32 or image is None:
            path, original = load_random_image()
            if original is None:
                print(f"Nie udało się wczytać obrazu: {path}")
                continue

        if key == ord('q'):
            conf_thresh = min(conf_thresh + 0.05, 1.0)
            print(f"conf_thresh: {conf_thresh:.2f}")
        elif key == ord('a'):
            conf_thresh = max(conf_thresh - 0.05, 0.0)
            print(f"conf_thresh: {conf_thresh:.2f}")
        elif key == ord('w'):
            nms_thresh = min(nms_thresh + 0.05, 1.0)
            print(f"nms_thresh: {nms_thresh:.2f}")
        elif key == ord('s'):
            nms_thresh = max(nms_thresh - 0.05, 0.0)
            print(f"nms_thresh: {nms_thresh:.2f}")
        elif key == ord('e'):
            imgsz = min(imgsz + 64, 2048)
            print(f"imgsz: {imgsz}")
        elif key == ord('d'):
            imgsz = max(imgsz - 64, 64)
            print(f"imgsz: {imgsz}")
        elif key == ord('r'):
            agnostic_nms = not agnostic_nms
            print(f"agnostic_nms: {agnostic_nms}")
        elif key == ord('z'):
            print(f"Img_path: {path}")
        elif key == ord('p'):
            path, original = save_labeled_image(
                path,
                original,
                last_detected_labels,
                last_detected_boxes,
                SAVE_IMAGE_DIR,
                SAVE_LABEL_DIR,
                "Zapisano do poprawy"
                )
        elif key == ord('o'):
            path, original = save_labeled_image(
                path,
                original,
                last_detected_labels,
                last_detected_boxes,
                SAVE_IMAGE_DIR_OK,
                SAVE_LABEL_DIR_OK,
                "Zapisano jako OK"
                )

        plate_box, char_boxes = detect_plate_and_chars(original)

        if plate_box is None:
            print("Brak tablicy na obrazie. Wciśnij SPACJĘ, aby wczytać nowy obraz lub ESC, aby wyjść.")
            final_plate = np.full((200, 600, 3), 240, dtype=np.uint8)
            cv2.putText(final_plate, "Brak tablicy", (50, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 255), 3, cv2.LINE_AA)
            image = original.copy()
        else:
            final_plate, image = prepare_plate_view(original, plate_box, char_boxes)

    if image is not None and final_plate is not None:
        cv2.namedWindow("Tablica", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Obraz zrodlowy", cv2.WINDOW_NORMAL)
        cv2.imshow("Tablica", final_plate)
        cv2.imshow("Obraz zrodlowy", image)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
