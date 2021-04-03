#Src: https://github.com/vbnmzxc9513/Nuclei-detection_detectron2/blob/master/train_Cascade.py
#Trying to copy same config as maskRCNN nucleus.py (which worked decent for train trees)

import detectron2
from detectron2.utils.logger import setup_logger
# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

setup_logger()

inDir = "/att/gpfsfs/briskfs01/ppl/acartas/git/forest.health/input/"
outDir = "/att/gpfsfs/briskfs01/ppl/acartas/git/forest.health/output"
tmpDir = "/att/gpfsfs/briskfs01/ppl/acartas/git/forest.health/tmp"

flight = 'OR_20190630_Three_Creek'
dsetName = flight+"_train"

register_coco_instances(dsetName, {}, os.path.join(inDir,dsetName+'.json'),os.path.join(inDir, "datasets",flight,"train"))
metadata = MetadataCatalog.get(dsetName)
dataset_dicts = DatasetCatalog.get(dsetName)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = outDir
# cfg.MODEL.WEIGHTS = os.path.join('model', "model_final_Cascade.pkl")
# if you have pre-trained weight.
cfg.DATASETS.TRAIN = (dsetName,)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
# 4999 iterations seems good enough, but you can certainly train longer
cfg.SOLVER.MAX_ITER = 1000
# faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (data, fig, hazelnut)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  # build output folder
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
