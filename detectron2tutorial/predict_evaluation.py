
import json
import os
import random

import cv2
# Some basic setup:
# Setup detectron2 logger
import detectron2
# import some common libraries
import numpy as np
import torch
from data_preparation import get_balloon_dicts
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_test_loader)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

setup_logger()
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)



def get_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    
    return cfg


if __name__ == '__main__':
    cfg = get_config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)

    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")

    dataset_dicts = get_balloon_dicts("balloon/val")
    for i, dct in enumerate(random.sample(dataset_dicts, 3)):    
        im = cv2.imread(dct["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=balloon_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(f"img_{i}.jpg" , out.get_image()[:, :, ::-1])


    evaluator = COCOEvaluator("balloon_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "balloon_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`