*captions.py

	Parse MSCOCO captions. Tokenize captions and organize captions by image name.
	
*captions.lua

	Pack up MSCOCO captions into a matrix of tokens.
	
*images.py

	Generate a list of MSCOCO train+val image names for feature extraction.
	
*extract_vgg19.lua

	Extract vgg-19 fc7 features from a list of images (in this case for MSCOCO train+val images).