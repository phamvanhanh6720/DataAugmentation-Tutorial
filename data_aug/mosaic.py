import random

import cv2
import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image

OUTPUT_SIZE = (600, 600)  # Height, Width
SCALE_RANGE = (0.3, 0.7)
FILTER_TINY_SCALE = 1 / 100  # if height or width lower than this scale, drop it.

DATA_DIR = '//data_samples'


def get_dataset(data_dir):

    labels = []
    data_dir = Path(data_dir)

    img_paths = list(data_dir.glob('*.jpg'))

    for img_file in img_paths:
        filename = img_file.stem
        label_file = data_dir / (filename + '.txt')
        if not label_file.is_file():
            raise Exception('label file is not exist')

        bboxes = []
        with open(label_file, 'r') as file:
            lines = file.readlines()

        img = cv2.imread(str(img_file))
        img_height, img_width, _ = img.shape
        del img

        for line in lines:
            line = line.strip()
            if line != '':
                box = list(map(float, line.split(' ')))

                id_class = box[0]
                x_center_normed = box[1]
                y_center_normed = box[2]
                width_normed = box[3]
                height_normed = box[4]

                x_left = (x_center_normed - width_normed / 2)
                x_left = round(x_left, 4)

                x_right = (x_center_normed + width_normed / 2)
                x_right = round(x_right, 4)

                y_left = (y_center_normed - height_normed / 2)
                y_left = round(y_left, 4)

                y_right = (y_center_normed + height_normed / 2)
                y_right = round(y_right, 4)

                bboxes.append([id_class, x_left, y_left, x_right, y_right])

        labels.append(bboxes)

    img_paths = list(map(str, img_paths))

    return img_paths, labels


class MosaicAugmentation:

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def sample(self):
        img_paths, annos = get_dataset(self.data_dir)

        idxs = random.sample(range(len(annos)), 4)

        new_image, new_annos = update_image_and_anno(img_paths, annos,
                                                     idxs,
                                                     OUTPUT_SIZE, SCALE_RANGE,
                                                     filter_scale=FILTER_TINY_SCALE)

        original_img = new_image.copy()
        for anno in new_annos:
            start_point = (int(anno[1] * OUTPUT_SIZE[1]), int(anno[2] * OUTPUT_SIZE[0]))
            end_point = (int(anno[3] * OUTPUT_SIZE[1]), int(anno[4] * OUTPUT_SIZE[0]))
            cv2.rectangle(new_image, start_point, end_point, (0, 255, 0), 1, cv2.LINE_AA)

        return original_img, new_image


def update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale=0.):
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    new_anno = []
    for i, idx in enumerate(idxs):
        path = all_img_list[idx]
        img_annos = all_annos[idx]

        img = cv2.imread(path)
        if i == 0:  # top-left
            img = cv2.resize(img, (divid_point_x, divid_point_y))
            output_img[:divid_point_y, :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = bbox[2] * scale_y
                xmax = bbox[3] * scale_x
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

        elif i == 1:  # top-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y))
            output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = bbox[2] * scale_y
                xmax = scale_x + bbox[3] * (1 - scale_x)
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        elif i == 2:  # bottom-left
            img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = scale_y + bbox[2] * (1 - scale_y)
                xmax = bbox[3] * scale_x
                ymax = scale_y + bbox[4] * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        else:  # bottom-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = scale_y + bbox[2] * (1 - scale_y)
                xmax = scale_x + bbox[3] * (1 - scale_x)
                ymax = scale_y + bbox[4] * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

    if 0 < filter_scale:
        new_anno = [anno for anno in new_anno if
                    filter_scale < (anno[3] - anno[1]) and filter_scale < (anno[4] - anno[2])]

    return output_img, new_anno


if __name__ == '__main__':
    main()