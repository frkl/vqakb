# VQAKB

* Use VQA and Caption-QA models to extract distributed representations of VQA knowledge.

* Improve image-caption ranking with VQA representations.

## Dependencies

The code is written in Python and Torch. You'll need to install and configure the following packages.

* Python (==2.7)
	* NLTK for tokenization
	* json
	* re

* Torch
	* cutorch
	* cunn
	* cjson
	* npy4th
	* image
	* loadcaffe

##Usage

The code is loosely organized as utility libraries for 

* Extracting VGG-19 fc7 activations. 
* Generating VQA, Caption-QA and image-caption ranking datasets from MSCOCO and the VQA dataset.
* Training and evaluating VQA, Caption-QA and image-caption ranking models
* Extracting image and caption representations as VQA predictions or fc7 activations.
* Training image-caption ranking models with feature fusion.

A more detailed description will be available soon<sup>TM</sup>. 