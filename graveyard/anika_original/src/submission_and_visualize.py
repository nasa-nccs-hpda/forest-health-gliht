#Src://github.com/vbnmzxc9513/Nuclei-detection_detectron2/blob/master/submission_and_visualize.ipynb
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pdb
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
import random
import cv2
import os
from os import listdir

from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "stage1_train_annotations.json", "./stage1_train")  # registet_coco_dataset

baseDir = "/att/gpfsfs/briskfs01/ppl/acartas/git/forest.health"

inDir = "/att/gpfsfs/briskfs01/ppl/acartas/git/forest.health/input/"
outDir = "/att/gpfsfs/briskfs01/ppl/acartas/git/forest.health/output"
tmpDir = "/att/gpfsfs/briskfs01/ppl/acartas/git/forest.health/tmp"

flight = 'OR_20190630_Three_Creek'
dsetName = flight+"_train"

register_coco_instances(dsetName, {}, os.path.join(inDir,dsetName+'.json'),os.path.join(inDir, "datasets",flight,"train"))
metadata = MetadataCatalog.get(dsetName)
dataset_dicts = DatasetCatalog.get(dsetName)

cfg = get_cfg()
cfg.merge_from_file(os.path.join(baseDir,"model","cascade_mask_rcnn_R_50_FPN_3x.yaml"))  # use cascade_mask_rcnn_R50_FPN for trainging model config
cfg.OUTPUT_DIR = os.path.join(outDir,"cascade")  # output weight directroy path
cfg.MODEL.WEIGHTS = os.path.join(outDir,"cascade","model_0004999.pth")  #  the path for weight save 
cfg.DATASETS.TRAIN = (dsetName,)  # use training data
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300  
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TEST = (dsetName)
predictor = DefaultPredictor(cfg)

toTest = "/att/gpfsfs/briskfs01/ppl/acartas/git/forest.health/input/datasets/OR_20190630_Three_Creek/test/c3r2_c5r12/images/OR_20190630_Three_Creek_c3r2_c5r12.png"

if not os.path.isfile(toTest):
  print('cant find: ' + toTest)

im = cv2.imread(toTest)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
               metadata=metadata,
               scale=1
              )
pdb.set_trace()
cv2.imwrite(os.path.join(outDir,'cascade',os.path.basename(toTest)[:-4]+'_pred.png'),v.get_image())


