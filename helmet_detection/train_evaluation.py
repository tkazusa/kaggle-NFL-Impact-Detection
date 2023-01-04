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
    cfg.DATASETS.TRAIN = ("helmet_train",)
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
    
    train, test = train_test_split(image_names, test_size=0.33, random_state=42)
    print(len(train), len(test))

    dataset_names = ['train', 'test']
    image_name_list = [train, test]

    for d, names in zip(dataset_names, image_name_list):
        DatasetCatalog.register(f"helmet_{d}", lambda: get_helmet_dicts(IMAGE_DIR, df, names))
        MetadataCatalog.get(f"helmet_{d}").set(thing_classes=[f"helmet_{d}"])
    helmet_metadata = MetadataCatalog.get("helmet_train")

    # Train
    cfg = get_config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluation
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_helmet_dicts(IMAGE_DIR, df, test)
    for i, dct in enumerate(random.sample(dataset_dicts, 3)):    
        im = cv2.imread(dct["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=helmet_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(f"img_{i}.jpg" , out.get_image()[:, :, ::-1])


    evaluator = COCOEvaluator("helmet_test", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "helmet_test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`