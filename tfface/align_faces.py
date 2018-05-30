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

r"""Align face images.

Usage:
```shell

$ python align_faces.py \
	--source_dir=/datasets/images/face_images \
	--target_dir=/datasets/mtcnn_images/face_images \
	--image_format=png \
	--image_size=299 \
	--margin=20.0 \
	--gpu_memory_fraction=0.2

$ python align_faces.py \
	--source_dir=/datasets/images/face_images \
	--target_dir=/datasets/mtcnn_images/face_images \
	--image_format=png \
	--image_size=299 \
	--margin=20.0 \
	--gpu_memory_fraction=0.2 \
	--class_name_file=class-name-file
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import random

import cv2
import numpy as np

from tfmtcnn.networks.FaceDetector import FaceDetector
from tfmtcnn.networks.NetworkFactory import NetworkFactory

from tfface.tools.DatasetAnalyzer import DatasetAnalyzer

def align_faces(args):
	
	class_names = DatasetAnalyzer.read_class_names(args.class_name_file)
	if(len(class_names) == 0):
		class_names = DatasetAnalyzer.get_class_names(args.source_dir)
	
	if(len(class_names) == 0):
		return(False)

	source_path = os.path.expanduser(args.source_dir)

	target_path = os.path.expanduser(args.target_dir)
	if( not os.path.exists(target_path) ):
		os.makedirs(target_path)

	if(args.model_root_dir):
		model_root_dir = args.model_root_dir
	else:
		model_root_dir = NetworkFactory.model_deploy_dir()

	last_network='ONet'
	face_detector = FaceDetector(last_network, model_root_dir)

	total_no_of_images = 0
	successfully_aligned_images = 0

	if(args.class_name_file):
		prefix = args.class_name_file + '-'
	else:
		prefix = ""

	successful_images = open(prefix + 'successful.txt', 'w')
	unsuccessful_images = open(prefix + 'unsuccessful.txt', 'w')	

	for class_name in class_names:

		source_class_dir = os.path.join(source_path, class_name)
		if(not os.path.isdir(source_class_dir)):
			continue

		target_class_dir = os.path.join(target_path, class_name)
		if( not os.path.exists(target_class_dir) ):
			os.makedirs(target_class_dir)

		image_filenames = os.listdir(source_class_dir)
		for image_filename in image_filenames:				

			source_relative_path = os.path.join(class_name, image_filename)
			source_filename = os.path.join(source_class_dir, image_filename)
			if( not os.path.isfile(source_filename) ):
				continue

			total_no_of_images += 1

			target_filename = os.path.join(target_class_dir, image_filename)
	                if( os.path.exists(target_filename) ):
				continue

               		try:
                       		current_image = cv2.imread(source_filename, cv2.IMREAD_COLOR)
               		except (IOError, ValueError, IndexError) as error:
				unsuccessful_images.write(source_relative_path + '\n')
				continue

			if(current_image is None):
				continue

			image_height = current_image.shape[0]
			image_width = current_image.shape[1]

        		boxes_c, landmarks = face_detector.detect(current_image)

			face_probability = 0.0
			found = False
			crop_box = np.zeros(4, dtype=np.int32)
	       		for index in range(boxes_c.shape[0]):      			
				if(boxes_c[index, 4] > face_probability):
					found = True
	      				face_probability = boxes_c[index, 4]
					bounding_box = boxes_c[index, :4]

					bounding_box_width = bounding_box[2] - bounding_box[0]
					bounding_box_height = bounding_box[3] - bounding_box[1]

					width_offset = (bounding_box_width * args.margin) / (2 * 100.0)
					height_offset = (bounding_box_height * args.margin) / (2 * 100.0)
					
                            		crop_box[0] = int(max( (bounding_box[0] - width_offset), 0))
                       			crop_box[1] = int(max( (bounding_box[1] - height_offset), 0))
                       			crop_box[2] = int(min( (bounding_box[2] + width_offset), image_width))
                       			crop_box[3] = int(min( (bounding_box[3] + height_offset), image_height))

			if(found):
				cropped_image = current_image[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2],:]			
				resized_image = cv2.resize(cropped_image, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
	       			
	       			cv2.imwrite(target_filename, resized_image)
				successful_images.write(source_relative_path + '\n')

				successfully_aligned_images += 1
			else:
				unsuccessful_images.write(source_relative_path + '\n')

	print('Total number of images are - %d' % total_no_of_images)
	print('Number of successfully aligned images are - %d' % successfully_aligned_images)
	print('Number of unsuccessful images are - %d' % (total_no_of_images - successfully_aligned_images) )

	return(True)

def main(args):

	if(not args.source_dir):
		raise ValueError('You must supply the input source directory with --source_dir')
	if(not os.path.exists(args.source_dir)):
		print('The input source directory is missing. Error processing the data source without the input source directory.')
		return

	if(not args.target_dir):
		raise ValueError('You must supply the output directory with --target_dir')

	OK = align_faces(args)
	if(OK):
		print("The dataset " + args.source_dir + " is aligned.")
	else:
		print("Error aligning the dataset " + args.source_dir + " .")        

def parse_arguments(argv):

	parser = argparse.ArgumentParser()
	parser.add_argument('--source_dir', type=str, help='Input directory with unaligned face images.', default=None)
	parser.add_argument('--target_dir', type=str, help='Target directory with aligned face images.', default=None)

	parser.add_argument('--model_root_dir', type=str, help='Input model root directory where model weights are saved.', default=None)
	parser.add_argument('--threshold', type=float, help='Lower threshold value for face probability (0 to 1.0).', default=0.6)

	parser.add_argument('--image_format', type=str, help='Output image format.', default='png')
	parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=299)
	parser.add_argument('--margin', type=float, help='Margin for the crop around the bounding box (height, width) in percetage.', default=20.0)

	parser.add_argument('--class_name_file', type=str, help='The class name file where class names to be processed are stored.', default=None)
	parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.2)

	return parser.parse_args(argv)


if __name__ == '__main__':

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	main(parse_arguments(sys.argv[1:]))

