# -*- coding: utf-8 -*-
#Folder logs_model: https://drive.google.com/drive/folders/1-kcV4qeiR3G3Nf4P8vFeL2fjksoE0YO0?usp=sharing
#Folder weights file coco h5: https://drive.google.com/drive/folders/1JMp8FLCZM5tk4KDWj-1ueS-flG9Ugtww?usp=sharing
"""
Created on Wed Mar  6 19:26:32 2024
@author: pc
"""
import os
import sys
import json
import random
import numpy as np
import skimage.draw
from os import listdir
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from keras.optimizers import Adam
from xml.etree import ElementTree
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from mrcnn import utils


class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        """Load a subset of the custom dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes according to the number of classes required to detect
        class_labels = self.load_class_labels(os.path.join(dataset_dir, "list.txt"))
        for idx, label in enumerate(class_labels, start=1):
            self.add_class("custom", idx, label)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations = json.load(open(os.path.join(dataset_dir, "labels/labels_food.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # Add images
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            custom = [s['region_attributes'] for s in a['regions'].values()]
            num_ids = [class_labels.index(n['label']) + 1 for n in custom if 'label' in n]

            # Load image
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "custom",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_class_labels(self, path):
        """Load class labels from a file."""
        with open(path, 'r') as file:
            class_labels = [line.strip() for line in file if line.strip()]
        return class_labels

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "custom":
            return super().load_mask(image_id)
        num_ids = image_info['num_ids']
        
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "custom":
            return info["path"]
        else:
            super().image_reference(image_id)

# Initialize dataset
dataset_train = CustomDataset()
dataset_train.load_custom("datasets/", "train")
dataset_train.prepare()
print('Train: %d' % len(dataset_train.image_ids))

# Load an image
image_id = random.choice(dataset_train.image_ids)
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)

# Display image with masks and bounding boxes
bbox = extract_bboxes(mask)
display_instances(image, bbox, mask, class_ids, dataset_train.class_names)
print("Number of bounding boxes:", len(bbox))
print("Shape of masks array:", mask.shape)
print("Number of class IDs:", len(class_ids))

# Define model configuration
class FoodsConfig(Config):
    NAME = "foods_cfg"
    NUM_CLASSES = 1 + 32
    STEPS_PER_EPOCH = 507
    LEARNING_RATE = 0.001
    ADAM_DECAY = 0.0
    ADAM_EPSILON = 1e-7
    
    def __init__(self):
        super().__init__()
        self.OPTIMIZER = Adam(lr=self.LEARNING_RATE, decay=self.ADAM_DECAY, epsilon=self.ADAM_EPSILON)

# Prepare config
config = FoodsConfig()
config.display() 

# Model training
ROOT_DIR = os.path.abspath("./")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs_model")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco.h5")

model = MaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


# define checkpoint callback
checkpoint_path = os.path.join(ROOT_DIR, "best.hdf5")
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='auto')
callback_list = [checkpoint]
print(checkpoint_path)

# Train weights (output layers or 'heads')
model.train(dataset_train, dataset_train, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')

history = model.keras_model.history.history
print(history)



# Plot the accuracy history
loss = history['loss']
val_loss = history['val_loss']


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')


plt.tight_layout()
plt.show()



