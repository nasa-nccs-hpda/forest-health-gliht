# Tracking Southern Alaskan Forest Health 

Deep learning algorithms to map damaged trees in forests.

## Image Data Creation and Locations

The data used for training these models is contained in [the following huggingface directory.](https://huggingface.co/datasets/saking3/alaska_dead_trees) 

The procedure for segmenting data was as follows: 
- Use k-means augmented with Gray Level Cooccurrence Matrix (GLCM) on red band to cluster image into 60 classes.
- Convert k-means classed image into binary mask by using QGIS to reclass each class as either "dead tree" (1) or "non-dead tree" (0)

For reference, failed segmentation attempts include: 
- the segment-geospatial python package
- automatic segmentation with the SAM model.  

## Location Data Creation and Locations

KMZ files containing the flight paths and an Jupyter notebook that uses geopandas to view the flight paths against a map are contained within the flight_path_kmz folder. 

## Original Modeling Methodology

The Segment Anything Model (SAM), DINOv2 Model, ResNet-152 Model pre-trained on ImageNet-1k, ResNet-152 Model with no pretraining, and a 53 layered CNN were trained using transfer learning on our labeled dataset containing labeled dead trees.

## General Workflow description
**Dataset creation/preprocessing:** BinaryMaskNoiseRemoval.ipynb, HuggingFaceDatasetCreation.ipynb

**Models:** Contained within DINOv2Model.ipynb, SAMModel.ipynb, Resnet152model.ipynb, RAWResnet152model.ipynb, and CNN50.ipynb.

**Running predictions on an entire image (vs a single tile):** The huggingface dataset has had all "blank" tiles (tiles with only 0 or 1 values) removed, so if you would like to tile up an whole image (including empty tiles), run predictions on it, then have it stitched back into a single image, you will need to follow the protocols in: DINOv2_PredictEntireImage.ipynb and SAM_PredictEntireImage.ipynb  

## Environment Setup
Can pip install packages as needed. 

## References
[SAM Model Reference](https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb)

[DINOv2 Model Reference](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Train_a_linear_classifier_on_top_of_DINOv2_for_semantic_segmentation.ipynb)

[ResNet-152 Model](https://huggingface.co/microsoft/resnet-152)
