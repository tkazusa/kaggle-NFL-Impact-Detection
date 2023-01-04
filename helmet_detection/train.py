import os
import random

import cv2
import detectron2
import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_test_loader)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from sklearn.model_selection import train_test_split

setup_logger()
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

INPUT_DIR = "../data/input/"
IMAGE_DIR = INPUT_DIR + "images/"

def get_config():
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("helmet_all",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.OUTPUT_DIR = '.output/all'
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    return cfg


def get_helmet_dicts(img_dir: str, df: pd.DataFrame, image_names: list) -> list:
    dataset_dicts = []
    for image_name in image_names:
        print(image_name)
        record = {}

        df_same_frame = df[df['image']==image_name]
        file_path = img_dir + image_name
        image_height, image_width = cv2.imread(file_path).shape[:2]

        record['file_name'] = file_path
        record['image_id'] = 0
        record['height'] = image_height
        record['width'] = image_width

        objs = []
        for idx, row in df_same_frame.iterrows():
            l, w, t, h = row[['left', 'width', 'top', 'height']]

            obj = {
                "bbox": [l, t, w, h],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0
            }

            objs.append(obj)
        record["annotations"] = objs

        dataset_dicts.append(record)
    return dataset_dicts


if __name__ == '__main__':
    csv_path = "../data/input/image_labels.csv"
    df = pd.read_csv(csv_path)
    image_names = df['image'].unique()

    DatasetCatalog.register(f"helmet_all", lambda: get_helmet_dicts(IMAGE_DIR, df, image_names))
    MetadataCatalog.get(f"helmet_all").set(thing_classes=[f"helmet_all"])
    helmet_metadata = MetadataCatalog.get("helmet_all")

    # Train
    cfg = get_config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()