from PIL import Image
import ntpath
import os
import glob
from PIL import Image, ImageDraw
import PIL
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
import random
import glob
from torchvision.utils import save_image
import string
import cv2
import ntpath


# convert yolo to pascal voc format
def yolo_convert_pascal_voc(box_yolo, h, w, label):
    a = int((box_yolo[0] * 2 * w - box_yolo[2] * w) / 2)
    c = int((box_yolo[0] * 2 * w + box_yolo[2] * w) / 2)
    b = int((box_yolo[1] * 2 * h - box_yolo[3] * h) / 2)
    d = int((box_yolo[1] * 2 * h + box_yolo[3] * h) / 2)
    return [label, a, b, c, d]


# convert pascal voc to yolo format
def pascal_voc_convert_yolo(list_box, labels, w, h):
    new_list_box = []
    new_boxes = list_box.tolist()
    labels = labels.tolist()
    for i, box in enumerate(list_box):
        a, b, c, d = new_boxes[i]
        new_boxes[i][0] = (a + c) / 2 / w
        new_boxes[i][1] = (b + d) / 2 / h
        new_boxes[i][2] = (c - a) / w
        new_boxes[i][3] = (d - b) / h
        new_boxes[i].insert(0, int(labels[i]))
    return new_boxes


# merge all bouding box
def yolo_format_to_str(list_box):
    for i in range(len(list_box)):
        list_box[i] = [str(e) for e in list_box[i]]
        list_box[i] = " ".join(list_box[i])

    return "\n".join(list_box)


def parse_annot(image_path, folder_annotation):
    image = Image.open(image_path, mode="r")
    image = image.convert("RGB")
    w, h = image.size

    # anno_path = image_path.replace("images", "labels")
    # anno_path = anno_path.replace("jpg", "txt")

    name_file = ntpath.basename(image_path).split(".")[0]
    anno_path = folder_annotation + f"{name_file}.txt"

    with open(anno_path, 'r') as f:
        anno_data = f.read()
    f.close()
    anno_data = anno_data.strip()
    if anno_data == "":
        return []
    else:
        boxes = list()
        labels = list()

        data_test = anno_data.split("\n")
        data_test = [i for i in data_test if i != ""]
        data_test = [i.split(" ") for i in data_test]
        for i, _ in enumerate(data_test):
            tmp = data_test[i]
            tmp = [float(e) for e in tmp]
            box_descrip = tmp[1:]
            new_box = yolo_convert_pascal_voc(box_yolo=box_descrip, h=h, w=w, label=int(tmp[0]))
            boxes.append(new_box[1:])
            labels.append(new_box[0])

        return {"boxes": boxes, "labels": labels}


class MixupAugmentation:
    def __init__(self, anno_dir):
        self.anno_dir = anno_dir
        self.min_lambd = 0.3
        self.max_lambd = 0.7

    @staticmethod
    def mixup(image_info_1, image_info_2, lambd):
        '''
            Mixup 2 image

            image_info_1, image_info_2: Info dict 2 image with keys = {"image", "label", "box"}
            lambd: Mixup ratio

            Out: mix_image (Temsor), mix_boxes, mix_labels
        '''
        img1 = image_info_1["image"]  # Tensor
        img2 = image_info_2["image"]  # Tensor
        mixup_width = max(img1.shape[2], img2.shape[2])
        mix_up_height = max(img1.shape[1], img2.shape[1])

        mix_img = torch.zeros(3, mix_up_height, mixup_width)
        mix_img[:, :img1.shape[1], :img1.shape[2]] = img1 * lambd
        mix_img[:, :img2.shape[1], :img2.shape[2]] += img2 * (1. - lambd)

        mix_labels = torch.cat((image_info_1["label"], image_info_2["label"]), dim=0)
        mix_boxes = torch.cat((image_info_1["box"], image_info_2["box"]), dim=0)

        return mix_img, mix_boxes, mix_labels

    def sample(self, image_path1, image_path2):
        image1 = Image.open(image_path1, mode="r")
        image1 = image1.convert("RGB")

        image2 = Image.open(image_path2, mode= "r")
        image2 = image2.convert("RGB")

        objects1 = parse_annot(image_path1, self.anno_dir)
        objects2 = parse_annot(image_path1, self.anno_dir)

        image_info_1 = {"image": F.to_tensor(image1),
                        "label": torch.Tensor(objects1['labels']), "box": torch.Tensor(objects1['boxes'])}

        image_info_2 = {"image": F.to_tensor(image2),
                        "label": torch.Tensor(objects2['labels']), "box": torch.Tensor(objects2['boxes'])}

        lambd = random.uniform(self.min_lambd, self.max_lambd)
        new_image, new_boxes, new_labels = self.mixup(image_info_1, image_info_2, lambd)

        new_image = torch.permute(new_image, (1, 2, 0))
        height, width, _ = new_image.shape
        ndarray_img = new_image.numpy()

        return ndarray_img
