# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
python evaluate_development_dataset.py --input_tsv_file /workspace/datasets/MS-Celeb-1M/01/development/MsCelebV1-Faces-Aligned-DevSet1.tsv --model_name=inception_resnet_v2 --checkpoint_path=/tensorflow/models/inception_resnet_v2/BEST_MS_100K --dataset_dir=/tensorflow/models/inception_resnet_v2/BEST_MS_100K --output_tsv_file /workspace/datasets/MS-Celeb-1M/01/development/Evaluation-MsCelebV1-Faces-Aligned-DevSet1.tsv

python evaluate_development_dataset.py --input_tsv_file /workspace/datasets/MS-Celeb-1M/01/development/MsCelebV1-Faces-Aligned-DevSet2.tsv --model_name=inception_resnet_v2 --checkpoint_path=/tensorflow/models/inception_resnet_v2/BEST_MS_100K --dataset_dir=/tensorflow/models/inception_resnet_v2/BEST_MS_100K --output_tsv_file /workspace/datasets/MS-Celeb-1M/01/development/Evaluation-MsCelebV1-Faces-Aligned-DevSet2.tsv
"""

########################################################################################################################################################################################################
# Input file format for evaluating performance of face aligned MS-Celeb-1M images.

# Files -
# MsCelebV1-Faces-Aligned-DevSet1.tsv
# MsCelebV1-Faces-Aligned-DevSet2.tsv

# Text files, each line is an image record containing 6 columns, delimited by TAB.
# Column1: Freebase MID (ground truth)
# Column2: EntityNameString
# Column3: ImageURL
# Column4: FaceID
# Column5: Not known
# Column6: FaceData_Base64Encoded d Data
########################################################################################################################################################################################################

########################################################################################################################################################################################################
# Output file format for evaluating performance of face aligned MS-Celeb-1M images.

# Text files, each line is an image record containing 6 columns, delimited by TAB.
# Column1: Freebase MID (ground truth)
# Column2: EntityNameString
# Column3: ImageURL
# Column4: FaceID
# Column5: MID estimated
# Column6: Confidence score
########################################################################################################################################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import base64
import sys
import os
import cv2
from scipy import misc
import argparse
import operator

from tfmtcnn.nets.FaceDetector import FaceDetector
from tfmtcnn.nets.NetworkFactory import NetworkFactory

from datasets import dataset_utils
from nets import nets_factory as network_factory
from preprocessing import preprocessing_factory

from tfface.classifier.Classifier import Classifier

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def main(args):
	if(not args.input_tsv_file):
		raise ValueError('You must supply input TSV file with --input_tsv_file.')
	if(not args.output_tsv_file):
		raise ValueError('You must supply output TSV file with --output_tsv_file.')

	if(not os.path.isfile(args.input_tsv_file)):
		return(False)

	model_root_dir = NetworkFactory.model_deploy_dir()
	last_network='ONet'
	face_detector = FaceDetector(last_network, model_root_dir)

	classifier_object = Classifier()
	if(not classifier_object.load_dataset(args.dataset_dir)):
		return(False)
	if(not classifier_object.load_model(args.checkpoint_path, args.model_name, args.gpu_memory_fraction)):
		return(False)

   	number_of_images = 0
	good_images = 0
	input_tsv_file = open(args.input_tsv_file, 'r')
	output_tsv_file = open(args.output_tsv_file, 'w')
	while( True ):
		
		input_data = input_tsv_file.readline().strip()
		if( not input_data ):
       			break	

		number_of_images += 1

       		fields = input_data.split('\t')
       		class_name = str(fields[0]) 
       		image_string = fields[5]
       		decoded_image_string = base64.b64decode(image_string)
       		image_data = np.fromstring(decoded_image_string, dtype=np.uint8)
       		input_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
		height, width, channels = input_image.shape

		cv2.imwrite('image.png', input_image)
		input_image = misc.imread('image.png')	

		input_clone = np.copy(input_image)
		boxes_c, landmarks = face_detector.detect(input_clone)

		face_probability = 0.0
		found = False
		crop_box = []
       		for index in range(boxes_c.shape[0]):      			
			if(boxes_c[index, 4] > face_probability):
				found = True
      				face_probability = boxes_c[index, 4]
				bounding_box = boxes_c[index, :4]
      				crop_box = [int(max(bounding_box[0],0)), int(max(bounding_box[1],0)), int(min(bounding_box[2], width)), int(min(bounding_box[3], height))]
		if(found):
			cropped_image = input_image[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2],:]			
		else:
			cropped_image = input_image

		#cropped_image = input_image

		class_names_probabilities = classifier_object.classify(cropped_image, print_results=False)
		predicted_name = "Unknown"
		probability = 0.0
		if(len(class_names_probabilities) > 0):
			names = map(operator.itemgetter(0), class_names_probabilities)
			probabilities = map(operator.itemgetter(1), class_names_probabilities)
			predicted_name = str(names[0])
			probability = probabilities[0]
			if( class_name == predicted_name ):
				good_images += 1				

		print(number_of_images, 'OK -', str(class_name == predicted_name), ', class_name -', class_name, ', predicted_name -', predicted_name, ', probability -', probability)

		#cv2.imshow('image', cropped_image)
		#cv2.waitKey();

		output_tsv_file.write(str(fields[0]) + '\t' + str(fields[1]) + '\t' + str(fields[2]) + '\t' + str(fields[3]) + '\t' + str(predicted_name) + '\t' + str(probability) + '\n')		

	print('Accuracy = ', (good_images * 100.0)/number_of_images, ' for ', number_of_images, ' images.')

	return(True)


if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_tsv_file', type=str, help='Input TSV file.')
	parser.add_argument('--output_tsv_file', type=str, help='Output TSV file.')
	parser.add_argument('--model_name', type=str, help='The name of the architecture to evaluate.', default='inception_v3')    
	parser.add_argument('--checkpoint_path', type=str, help='The directory where the model was written to or an absolute path to a checkpoint file.', default='/models/tensorflow/inception_v3/facescrub/all')
	parser.add_argument('--dataset_dir', type=str, help='The directory where the dataset files are stored.', default='/dataset/tensorflow/facescrub')
	parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.6)	
	main(parser.parse_args())


