import os
import torch
import cv2
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

class ShipDataset(Dataset):
    def __init__(self, img_dir, ann_dir):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()

        xml_path = os.path.join(
            self.ann_dir, img_name.replace(".png", ".xml")
        )

        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # ship class

        target = {
            "boxes": torch.tensor(boxes).float(),
            "labels": torch.tensor(labels).long()
        }

        return image, target
