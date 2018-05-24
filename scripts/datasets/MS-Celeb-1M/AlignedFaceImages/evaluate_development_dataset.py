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
import argparse

def main(args):

   	number_of_images = 0

	input_tsv_file = open(args.input_tsv_file, 'r')
	output_tsv_file = open(args.output_tsv_file, 'w')
	while( True ):
		input_data = input_tsv_file.readline().strip()
		if( not input_data ):
       			break	

       		fields = input_data.split('\t')
       		class_name = fields[0] 
       		image_string = fields[5]
       		decoded_image_string = base64.b64decode(image_string)
       		image_data = np.fromstring(decoded_image_string, dtype=np.uint8)
       		input_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

		class_name = 'name'
		probability = 1.0
		output_tsv_file.write(fields[0] + '\t' + fields[1] + '\t' + fields[2] + '\t' + fields[3] + '\t' + class_name + '\t' + str(probability) + '\n')

		number_of_images += 1

	print('Number of input images - ' + str(number_of_images) + '.')


if __name__ == '__main__':
  
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_tsv_file', type=str, help='Input TSV file.')
	parser.add_argument('--output_tsv_file', type=str, help='Output TSV file.')
	parser.add_argument('--model_name', type=str, help='The name of the architecture to evaluate.', default='inception_v3')    
	parser.add_argument('--checkpoint_path', type=str, help='The directory where the model was written to or an absolute path to a checkpoint file.', default='/models/tensorflow/inception_v3/facescrub/all')
	parser.add_argument('--dataset_dir', type=str, help='The directory where the dataset files are stored.', default='/dataset/tensorflow/facescrub')
	parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.6)	
	main(parser.parse_args())


