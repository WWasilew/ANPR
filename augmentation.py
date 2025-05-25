from datetime import datetime
import glob
import os
import cv2
import albumentations as A
# import numpy as np
from utils import plot_examples


def save_files(base_name, augmented_bboxes, augmented_labels):
    # data i godzina do dodoawania do plikow
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Tworzenie nowej nazwy pliku
    new_filename = f"{base_name}_{timestamp}_{i}.jpg"
    new_b_filename = f"{base_name}_{timestamp}_{i}.txt"

    # Zapisywanie obrazu
    output_img_path = os.path.join(OUTPUT_IMAGE_FOLDER, new_filename)
    cv2.imwrite(output_img_path, cv2.cvtColor(augmented_img,
                                              cv2.COLOR_RGB2BGR))

    # Zapisywanie bounding boxów w formacie YOLO
    output_bbox_path = os.path.join(OUTPUT_BBOX_FOLDER, new_b_filename)
    with open(output_bbox_path, "w") as f:
        for bbox, label in zip(augmented_bboxes, augmented_labels):
            box0 = bbox[0]
            box1 = bbox[1]
            box2 = bbox[2]
            box3 = bbox[3]
            f.write(f"{label} {box0} {box1} {box2} {box3}\n")


def load_yolo_bboxes(txt_file):
    bboxes = list()
    class_labels = list()

    with open(txt_file, "r") as file:
        for line in file:
            values = line.strip().split()
            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)

    return bboxes, class_labels


# foldery z obrazami
IMAGE_FOLDER = "albumentation/assets/images/"
BBOX_FOLDER = "albumentation/assets/labels/"

# foldery z otrzymanymi obrazami
OUTPUT_IMAGE_FOLDER = "albumentation/created/images/"
OUTPUT_BBOX_FOLDER = "albumentation/created/labels/"

# OUTPUT_IMAGE_FOLDER = "training/images/train/"
# OUTPUT_BBOX_FOLDER = "training/labels/train/"

# Tworzenie folderów, jeśli nie istnieją
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_BBOX_FOLDER, exist_ok=True)


image_files = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.[jp][pn]g")))

transform = A.Compose(
    [
        # A.Resize(width=1920, height=1080),
        # A.RandomScale(scale_limit=(0.1, 0.4), p=0.7),
        # A.CropAndPad(percent=(0.0, 1.3), p=0.8),
        # A.RandomCrop(width=1280, height=720),
        A.LongestMaxSize(max_size=1280),
        A.Affine(
            scale=(0.1, 0.4),
            translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
            fit_output=False,  # dostosowywanie rozmiaru do nowej tablicy
            p=1.0
        ),
        A.PadIfNeeded(min_height=720, min_width=1280,
                      border_mode=cv2.BORDER_CONSTANT),
        A.Perspective(scale=(0.05, 0.08), p=0.9),
        A.Rotate(limit=10, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.RGBShift(r_shift_limit=10,
                   g_shift_limit=10,
                   b_shift_limit=10,
                   p=0.9),
        A.Blur(blur_limit=15, p=0.5),
        A.ColorJitter(p=0.3),
    ], bbox_params=A.BboxParams(format="yolo",
                                min_visibility=0.3,
                                label_fields=["labels"])  # min_area=2048,
)

images_list = []
saved_bboxes = []
saved_labels = []

for img_path in image_files:
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pobranie nazwy pliku bez rozszerzenia
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    bbox_file = os.path.join(BBOX_FOLDER, base_name + ".txt")

    if os.path.exists(bbox_file):
        bboxes, labels = load_yolo_bboxes(bbox_file)

        for i in range(1):
            augmentations = transform(image=image,
                                      bboxes=bboxes,
                                      labels=labels)
            augmented_img = augmentations["image"]
            augmented_bboxes = [list(bbox) for bbox in augmentations["bboxes"]]
            augmented_labels = augmentations["labels"]
            for j in range(len(augmented_labels)):
                augmented_labels[j] = int(augmented_labels[j])

            if len(augmented_bboxes) == 0:
                continue

            # save_files(base_name, augmented_bboxes, augmented_labels)

            images_list.append(augmented_img)
            saved_bboxes.append(augmented_bboxes)
            saved_labels.append(augmented_labels)

plot_examples(images_list, saved_labels, saved_bboxes)
