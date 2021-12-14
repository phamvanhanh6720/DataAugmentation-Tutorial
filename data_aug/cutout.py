from PIL import Image
import PIL
import torch
import torchvision.transforms.functional as F
import random
from mixup import parse_annot


def intersect(boxes1, boxes2):
    '''
        Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)

        Out: Intersection each of boxes1 with respect to each of boxes2,
             a tensor of dimensions (n1, n2)
    '''
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy = torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
                        boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))

    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)  # (n1, n2, 2)
    return inter[:, :, 0] * inter[:, :, 1]  #(n1, n2)


class CutoutAugmentation:
    def __init__(self, anno_dir):
        self.anno_dir = anno_dir
        self.fill_val = 0
        self.bbox_remove_thres = 0.4
        self.min_num_of_cutout_box = 8
        self.max_num_of_cutout_box = 11

    def cutout(self, image, boxes, labels, fill_val=0, bbox_remove_thres=0.4):
        '''
            Cutout augmentation
            image: A PIL image
            boxes: bounding boxes, a tensor of dimensions (#objects, 4)
            labels: labels of object, a tensor of dimensions (#objects)

            Out: new image, new_boxes, new_labels
        '''
        if type(image) == PIL.Image.Image:
            image = F.to_tensor(image)
        original_h = image.size(1)
        original_w = image.size(2)
        original_channel = image.size(0)

        new_image = image.copy()
        new_boxes = boxes.copy()
        new_labels = labels.copy()

        for _ in range(50):
            # Random cutout size: [0.15, 0.5] of original dimension
            cutout_size_h = random.uniform(0.05 * original_h, 0.05 * original_h)
            cutout_size_w = random.uniform(0.05 * original_w, 0.05 * original_w)

            # Random position for cutout
            left = random.uniform(0, original_w - cutout_size_w)
            right = left + cutout_size_w
            top = random.uniform(0, original_h - cutout_size_h)
            bottom = top + cutout_size_h
            cutout = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])

            # Calculate intersect between cutout and bounding boxes
            overlap_size = intersect(cutout.unsqueeze(0), boxes)
            area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            ratio = overlap_size / area_boxes
            # If all boxes have Iou greater than bbox_remove_thres, try again
            if ratio.min().item() > bbox_remove_thres:
                continue

            cutout_arr = torch.full((original_channel, int(bottom) - int(top), int(right) - int(left)), fill_val)
            new_image[:, int(top):int(bottom), int(left):int(right)] = cutout_arr

            # Create new boxes and labels
            boolean = ratio < bbox_remove_thres

            new_boxes = boxes[boolean[0], :]

            new_labels = labels[boolean[0]]

        return new_image, new_boxes, new_labels

    def sample(self, img_path):
        image = Image.open(img_path, mode="r")
        image = image.convert("RGB")

        # Get anno file from image path
        objects = parse_annot(img_path, self.anno_dir)
        boxes = torch.FloatTensor(objects['boxes'])
        labels = torch.LongTensor(objects['labels'])

        new_image = image
        new_boxes = boxes
        new_labels = labels

        num_cutout_box = random.randint(self.min_num_of_cutout_box, self.max_num_of_cutout_box)
        for i in range(num_cutout_box):
            new_image, new_boxes, new_labels = self.cutout(image, boxes, labels)

        new_image = torch.permute(new_image, (1, 2, 0))
        # height, width, _ = new_image.shape
        ndarray_img = new_image.numpy()

        return ndarray_img
