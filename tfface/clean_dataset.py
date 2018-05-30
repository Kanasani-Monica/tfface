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

r"""Clean dataset.

Usage:
```shell

$ python clean_dataset.py \
	--model_name=inception_resnet_v2 \
	--checkpoint_path=/tensorflow/models/inception_resnet_v2/BEST_MS_100K \
	--dataset_dir=/tensorflow/models/inception_resnet_v2/BEST_MS_100K \
	--source_dir=/datasets/MS_100K \
	--target_dir=/datasets/MS_100K_BACKUP \
	--use_top=1 \
	--gpu_memory_fraction=0.1 \
	--class_name_file=class-file 
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from tfface.classifier.Classifier import Classifier

from tfface.tools.read_class_names import read_class_names
from tfface.tools.get_class_names import get_class_names

def process(args):

	if(not os.path.isfile(args.class_name_file)):
		return(False)

	class_names = read_class_names(args.class_name_file)

	if(len(class_names) == 0):
		class_names = get_class_names(args.source_dir)
	no_of_classes = len(class_names)
	if(no_of_classes == 0):
		return(False)
	
	classifier_object = Classifier()
	if(not classifier_object.load_dataset(args.dataset_dir)):
		return(False)
	if(not classifier_object.load_model(args.checkpoint_path, args.model_name, args.gpu_memory_fraction)):
		return(False)

	network_size = classifier_object.network_image_size()

	source_path = os.path.expanduser(args.source_dir)
	target_path = os.path.expanduser(args.target_dir)

	good_files = 0
	bad_files = 0

	for class_name in class_names:

		source_class_dir = os.path.join(source_path, class_name)
		if(not os.path.isdir(source_class_dir)):
			continue

		images = os.listdir(source_class_dir)
		for image in images:
			source_filename = os.path.join(source_class_dir, image)
			if(not os.path.isfile(source_filename)):
				continue

               		try:
                       		current_image = cv2.imread(source_filename, cv2.IMREAD_COLOR)
               		except (IOError, ValueError, IndexError) as error:
				continue

			if(current_image is None):
				continue

			class_names_probabilities = classifier_object.classify(current_image, args.use_top)
			is_good = False
			for predicted_name, probability in class_names_probabilities:
				if(predicted_name == class_name):
					is_good = True
					break

			if(is_good):
				good_files = good_files + 1
			else:
				target_class_dir = os.path.join(target_path, class_name)
				if( not os.path.exists(target_class_dir) ):
					os.makedirs(target_class_dir)

				target_filename = os.path.join(target_class_dir, image)
				os.rename(source_filename, target_filename)

				bad_files = bad_files + 1

	print("Good files are - " + str(good_files) + " and bad files are - " + str(bad_files))
	return(True)

def main(args):  
	if(not args.checkpoint_path):
		raise ValueError('You must supply the checkpoint path with --checkpoint_path')	
	if(not os.path.exists(args.checkpoint_path)):
		print('The checkpoint path is missing. Error processing the data source without the checkpoint path.')
		return

	if(not args.dataset_dir):
		raise ValueError('You must supply the dataset directory with --dataset_dir')		
	if(not os.path.exists(args.dataset_dir)):
		print('The dataset directory is missing. Error processing the data source without the dataset directory.')
		return

	if(not args.source_dir):
		raise ValueError('You must supply the source directory with --source_dir')
	if(not os.path.exists(args.source_dir)):
		print('The source directory is missing. Error cleaning the data source without the source directory.')
		return

	if(not args.target_dir):
		raise ValueError('You must supply the target directory with --target_dir')

	if(not args.class_name_file):
		raise ValueError('You must supply the class name file with --class_name_file')
	if(not os.path.exists(args.class_name_file)):
		print('The class name file is missing. Error processing the data source without the class name file.')
		return

	target_dir = os.path.expanduser(args.target_dir)
	if(not os.path.exists(target_dir)):
		os.makedirs(target_dir)

	OK = process(args)
	if(OK):
		print("The dataset " + args.source_dir + " is cleaned using the class name file " + str(args.class_name_file) + ".")
	else:
		print("Error cleaning the dataset " + args.source_dir + " .")
		
def parse_arguments(argv):

	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str, help='The name of the architecture.', default='inception_v3')    
	parser.add_argument('--checkpoint_path', type=str, help='The directory where the model was written to or an absolute path to a checkpoint file.', default=None)
	parser.add_argument('--dataset_dir', type=str, help='The directory where the dataset files are stored.', default=None)
	
	parser.add_argument('--source_dir', type=str, help='The directory from where input data is read for cleaning.', default=None)
	parser.add_argument('--target_dir', type=str, help='The directory where the output cleaned data is saved.', default=None)

	parser.add_argument('--class_name_file', type=str, help='The class name file where class names to be processed are stored.', default=None)
	parser.add_argument('--use_top', type=int, help='Retain top "top" number of classes.', default=10)
	parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.1)
	return(parser.parse_args(argv))

if __name__ == '__main__':

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	main(parse_arguments(sys.argv[1:]))


