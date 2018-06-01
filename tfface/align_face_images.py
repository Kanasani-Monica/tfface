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

r"""Align face images (old version).

Usage:
```shell

$ python align_face_images.py \
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

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import random
from time import sleep

from tfface.mtcnn.MTCNN import MTCNN
from tfface.tools.DatasetAnalyzer import DatasetAnalyzer

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def process(args):

	classes = DatasetAnalyzer.read_class_names(args.class_name_file)
	if(len(classes) == 0):
		classes = DatasetAnalyzer.get_class_names(args.source_dir)
	
	if(len(classes) == 0):
		return(False)

	no_of_classes = len(classes)
	if(no_of_classes == 0):
		return(False)

	source_path = os.path.expanduser(args.source_dir)
	target_path = os.path.expanduser(args.target_dir)
	if( not os.path.exists(target_path) ):
		os.makedirs(target_path)
       
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			mtcnn_object = MTCNN(sess, None)
    
	minsize = 20 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709 # scale factor
   
	total_no_of_images = 0
	successfully_aligned_images = 0

	if(args.class_name_file):
		prefix = args.class_name_file + '-'
	else:
		prefix = ""

	successful_images = open(prefix + 'successful.txt', 'w')
	unsuccessful_images = open(prefix + 'unsuccessful.txt', 'w')	

	for class_name in classes:

		source_class_dir = os.path.join(source_path, class_name)
		if(not os.path.isdir(source_class_dir)):
			continue

		target_class_dir = os.path.join(target_path, class_name)
		if( not os.path.exists(target_class_dir) ):
			os.makedirs(target_class_dir)

		image_filenames = os.listdir(source_class_dir)
		for image_filename in image_filenames:			

			total_no_of_images += 1

			source_relative_path = os.path.join(class_name, image_filename)
			source_filename = os.path.join(source_class_dir, image_filename)
			if( not os.path.isfile(source_filename) ):
				continue

			target_filename = os.path.join(target_class_dir, image_filename)
	                if( os.path.exists(target_filename) ):
				continue
			else:
                    		try:
                        		image = misc.imread(source_filename)
                    		except (IOError, ValueError, IndexError) as error:
                        		unsuccessful_images.write(source_relative_path + os.linesep)
					continue
                    		else:
                        		if(image.ndim < 2):
                            			unsuccessful_images.write(source_relative_path + os.linesep)
                            			continue
				
                        		if(image.ndim == 2):
                            			image = to_rgb(image)

                        		bounding_boxes, _ = mtcnn_object.detect_face(image, minsize, threshold, factor)
                        		no_of_faces = bounding_boxes.shape[0]

                        		if(no_of_faces > 0):
                            			det = bounding_boxes[:,0:4]
                            			img_size = np.asarray(image.shape)[0:2]

                            			if(no_of_faces > 1):
                                			bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                			image_center = img_size / 2
                                			offsets = np.vstack([ (det[:,0]+det[:,2])/2-image_center[1], (det[:,1]+det[:,3])/2-image_center[0] ])
                                			offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                			index = np.argmax(bounding_box_size-offset_dist_squared*2.0)
                                			det = det[index,:]

                            			det = np.squeeze(det)

						bounding_box_width = det[2] - det[0]
						bounding_box_height = det[3] - det[1]

						width_offset = (bounding_box_width * args.margin) / (2 * 100.0)
						height_offset = (bounding_box_height * args.margin) / (2 * 100.0)

                            			bounding_box = np.zeros(4, dtype=np.int32)
                            			bounding_box[0] = np.maximum(det[0] - width_offset, 0)
                            			bounding_box[1] = np.maximum(det[1] - height_offset, 0)
                            			bounding_box[2] = np.minimum(det[2] + width_offset, img_size[1])
                            			bounding_box[3] = np.minimum(det[3] + height_offset, img_size[0])

                            			cropped_image = image[bounding_box[1]:bounding_box[3],bounding_box[0]:bounding_box[2],:]
                            			scaled_image = misc.imresize(cropped_image, (args.image_size, args.image_size), interp='bilinear')

                            			successfully_aligned_images += 1
                            			misc.imsave(target_filename, scaled_image)
						successful_images.write(source_relative_path + os.linesep)
                        		else:
                            			unsuccessful_images.write(source_relative_path + os.linesep)

                            
	print('Total number of images are - %d' % total_no_of_images)
	print('Number of successfully aligned images are - %d' % successfully_aligned_images)
	print('Number of unsuccessfull images are - %d' % (total_no_of_images - successfully_aligned_images) )

	return(True)

def main(args):

	if(not args.source_dir):
		raise ValueError('You must supply the input source directory with --source_dir')
	if(not os.path.exists(args.source_dir)):
		print('The input source directory is missing. Error processing the data source without the input source directory.')
		return

	if(not args.target_dir):
		raise ValueError('You must supply the output directory with --target_dir')

	OK = process(args)
	if(OK):
		print("The dataset " + args.source_dir + " is aligned using the class name file " + str(args.class_name_file) + ".")
	else:
		print("Error aligning the dataset " + args.source_dir + " .")        

def parse_arguments(argv):

	parser = argparse.ArgumentParser()
	parser.add_argument('--source_dir', type=str, help='Input directory with unaligned images.')
	parser.add_argument('--target_dir', type=str, help='Target directory with aligned face images.')
	parser.add_argument('--image_format', type=str, help='Output image format.', default='png')
	parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=299)
	parser.add_argument('--margin', type=float, help='Margin for the crop around the bounding box (height, width) in percetage.', default=20.0)
	parser.add_argument('--class_name_file', type=str, help='The class name file where class names to be processed are stored.', default=None)
	parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.2)

	return parser.parse_args(argv)

if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	main(parse_arguments(sys.argv[1:]))

