import glob
import os
import matplotlib.pyplot as plt
# import numpy as np

IMAGES_FOLDER = "albumentation/created/images/"
BBOX_FOLDER = "albumentation/created/labels/"

txt_files = sorted(glob.glob(os.path.join(BBOX_FOLDER, "*.txt")))

number_of_entity = dict()


def create_dictionary(dictionary):
    char = "default"
    new_dict = dict()
    for label in dictionary:
        match label:
            case 0:
                char = "car"
            case 1:
                char = "number_plate"
            case 2:
                char = "A"
            case 3:
                char = "B"
            case 4:
                char = "C"
            case 5:
                char = "D"
            case 6:
                char = "E"
            case 7:
                char = "F"
            case 8:
                char = "G"
            case 9:
                char = "H"
            case 10:
                char = "I"
            case 11:
                char = "J"
            case 12:
                char = "K"
            case 13:
                char = "L"
            case 14:
                char = "M"
            case 15:
                char = "N"
            case 16:
                char = "O"
            case 17:
                char = "P"
            case 18:
                char = "Q"
            case 19:
                char = "R"
            case 20:
                char = "S"
            case 21:
                char = "T"
            case 22:
                char = "U"
            case 23:
                char = "V"
            case 24:
                char = "W"
            case 25:
                char = "X"
            case 26:
                char = "Y"
            case 27:
                char = "Z"
            case 28:
                char = "1"
            case 29:
                char = "2"
            case 30:
                char = "3"
            case 31:
                char = "4"
            case 32:
                char = "5"
            case 33:
                char = "6"
            case 34:
                char = "7"
            case 35:
                char = "8"
            case 36:
                char = "9"
            case 37:
                char = "0"
        new_dict[char] = dictionary[label]
    return new_dict


def print_dictionary(dictionary):
    for label in dictionary:
        print(f'{label} : {dictionary[label]}')


def load_yolo_bboxes(txt_file):
    class_labels = list()

    with open(txt_file, "r") as file:
        for line in file:
            values = line.strip().split()
            class_id = int(values[0])

            class_labels.append(class_id)

    return class_labels


for file in txt_files:
    labels = load_yolo_bboxes(file)
    for label in labels:
        if label in number_of_entity.keys():
            number_of_entity[label] += 1
        else:
            number_of_entity[label] = 1

sorted_dict = dict(sorted(number_of_entity.items()))

labels = list()
vals = list()
dict_of_entities = create_dictionary(sorted_dict)
print_dictionary(dict_of_entities)
for label in dict_of_entities:
    labels.append(label)
    vals.append(dict_of_entities[label])
plt.plot(labels, vals, marker="x")
plt.grid()
plt.show()
