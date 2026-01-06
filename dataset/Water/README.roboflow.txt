
water bottle - v3 2022-08-06 9:31pm
==============================

This dataset was exported via roboflow.com on December 31, 2025 at 7:47 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2057 images.
Waterbottle are annotated in YOLOv11 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 29 percent of the image
* Random shear of between -17° to +17° horizontally and -15° to +15° vertically
* Random brigthness adjustment of between -30 and +30 percent
* Random exposure adjustment of between -26 and +26 percent
* Random Gaussian blur of between 0 and 2.25 pixels
* Salt and pepper noise was applied to 3 percent of pixels


