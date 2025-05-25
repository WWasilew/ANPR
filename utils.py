# import random
import cv2
from matplotlib import pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
# import albumentations as A


def visualize(image):
    plt.figure(figsize=(30, 30))
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def plot_examples(images, labels, bboxes=None):
    fig = plt.figure(figsize=(30, 30))
    columns = 8
    rows = 7

    for i in range(1, len(images)+1):
        if bboxes is not None:
            bbox = bboxes[i - 1]
            label = labels[i - 1]
            img = visualize_all_bboxes(images[i - 1], label, bbox)
        else:
            img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


# From https://albumentations.ai/docs/examples/example_bboxes/
def visualize_all_bboxes(img, labels, bboxes, color=(255, 0, 0), thickness=2):
    h, w, _ = img.shape
    i = 0
    for bbox in bboxes:
        label = labels[i]
        i += 1
        # YOLO format: (x_center, y_center, width, height) w skali 0-1
        x_center, y_center, width, height = bbox

        # Zamiana YOLO na wartości pikselowe
        x_center, y_center, width, height = (
            int(x_center * w),
            int(y_center * h),
            int(width * w),
            int(height * h),
        )

        # Obliczenie współrzędnych rogów prostokąta
        x_min = x_center - width // 2
        y_min = y_center - height // 2
        x_max = x_center + width // 2
        y_max = y_center + height // 2

        # Rysowanie prostokąta na obrazie
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

        # Opcjonalnie: Dodanie etykiety klasy nad prostokątem
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(label), (x_min, y_min - 10), font, 1, color, 2)

    return img
