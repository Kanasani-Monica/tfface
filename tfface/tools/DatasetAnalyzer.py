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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import fnmatch
import operator
import random

class DatasetAnalyzer(object):

	default_base_file_name = "class-file-"

	def __init__(self):
		self._patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

	@classmethod
	def check_dataset_dir(cls, data_source_dir):
		OK = False

		if(not os.path.exists(data_source_dir)):
			return(OK)

		if(not os.path.isdir(data_source_dir)):
			return(OK)

		OK = True

		return(OK)

	@classmethod
	def read_class_names(cls, class_name_file):
		class_names = [] 

		if( (class_name_file is None) or (not os.path.isfile(class_name_file)) ):
			return(class_names)
	
		with open(class_name_file) as class_file: 
			class_names = class_file.readlines()

		no_of_classes = len(class_names)
		for index in range(no_of_classes):
			class_names[index] = class_names[index].rstrip()
	
		return(class_names)

	@classmethod
	def get_class_names(cls, source_dir):

		class_names = []
		if(not source_dir) or (not os.path.exists(source_dir)):
			return(class_names)
	
		class_names = [ class_name for class_name in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, class_name)) ]
		return(class_names)

	def analyse(self, data_source_dir, backup_dir=None, min_no_of_images=1, sort_results=True, print_results=False):

		OK = False
		data_source = {}

		if(not DatasetAnalyzer.check_dataset_dir(data_source_dir)):
			return(OK, data_source)

		if(not backup_dir):
			take_backup = False
		else:
			take_backup = True

		if( take_backup and os.path.exists(backup_dir) and (not os.path.isdir(backup_dir)) ):
			return(OK, data_source)
	
		if(take_backup and (not os.path.exists(backup_dir)) ):
			os.makedirs(backup_dir)

		class_names = DatasetAnalyzer.get_class_names(data_source_dir)

		no_of_classes = 0
		total_no_of_images = 0

		max_no_of_images_per_class = 0
		min_no_of_images_per_class = sys.maxint

		for class_name in class_names:	
			class_source_dir = os.path.join(data_source_dir, class_name)	

			if(not os.path.isdir(class_source_dir)):
				continue

			no_of_images = 0			
			for pattern in self._patterns:
				images = fnmatch.filter(os.listdir(class_source_dir), pattern)
				no_of_images = no_of_images + len(images)

			if(no_of_images >= min_no_of_images):
				no_of_classes = no_of_classes + 1
				total_no_of_images = total_no_of_images + no_of_images
				data_source[class_name] = no_of_images

				if(max_no_of_images_per_class < no_of_images):
					max_no_of_images_per_class = no_of_images

				if(min_no_of_images_per_class > no_of_images):
					min_no_of_images_per_class = no_of_images
			else:
				if(take_backup):
					class_backup_dir = os.path.join(backup_dir, class_name)
					os.rename(class_source_dir, class_backup_dir)
			
		if(no_of_classes > 0):
			OK = True
			if(print_results):
				print("Number of classes are - " + str(no_of_classes))
				print("Total number of images are - " + str(total_no_of_images))
				print("Average number of images per class are - " + str((total_no_of_images*1.0)/no_of_classes))
				print("Maximum number of images for a class are - " + str(max_no_of_images_per_class))
				print("Minimum number of images for a class are - " + str(min_no_of_images_per_class))

		if(sort_results):
			sorted_data_source = sorted(data_source.items(), key=operator.itemgetter(1), reverse=True)
		else:
			sorted_data_source = sorted(data_source.items(), key=operator.itemgetter(1))

		return(OK, sorted_data_source)


	def create_class_files(self, data_source_dir, no_of_subsets, base_file_name):
		OK, data_source = self.analyse(data_source_dir, sort_results=False)

		no_of_classes = len(data_source)
		subset_size = int(no_of_classes / no_of_subsets)
		if(subset_size <= 0):
			OK = False

		if(OK):
			classes = map(operator.itemgetter(0), data_source)
			random.shuffle(classes)

			for index in range(no_of_subsets):
				start = index * subset_size				
				end = (index + 1) * subset_size

				if(index == (no_of_subsets - 1) ):
					end = max(end, no_of_classes)					

				subset = classes[start:end]

				file_name = base_file_name + str(index) 
				with open(file_name, 'w') as subset_file:
					for class_name in subset:
						subset_file.write(class_name + os.linesep)
					subset_file.close()
		return(OK)


	def extract_classes(self, source_dir, target_dir, target_number_of_classes):
		OK = False

		if(not DatasetAnalyzer.check_dataset_dir(source_dir)):
			return(OK)

		if( os.path.exists(target_dir) and (not os.path.isdir(target_dir)) ):
			return(OK)

		OK, data_source = self.analyse(source_dir, sort_results=True)
		source_number_of_classes = len(data_source)
		if(source_number_of_classes == 0):
			return(OK)

		if(source_number_of_classes < target_number_of_classes):
			target_number_of_classes = source_number_of_classes

		source_class_names = map(operator.itemgetter(0), data_source)
		target_class_names = source_class_names[:target_number_of_classes]

		if(not os.path.exists(target_dir)):
			os.makedirs(target_dir)
	
		source_path = os.path.expanduser(source_dir)
		target_path = os.path.expanduser(target_dir)

		for class_name in target_class_names:	
			source_class_dir = os.path.join(source_path, class_name)
			if(not os.path.isdir(source_class_dir)):
				continue

			target_class_dir = os.path.join(target_path, class_name)
			os.rename(source_class_dir, target_class_dir)

		OK = True
		return(OK)


	def format(self, data_source_dir):
		OK = False

		if(not DatasetAnalyzer.check_dataset_dir(data_source_dir)):
			return(OK)

		class_names = DatasetAnalyzer.get_class_names(data_source_dir)

		for class_name in class_names:						
			class_source_dir = os.path.join(data_source_dir, class_name)	

			if(not os.path.isdir(class_source_dir)):
				continue

			no_of_images = 0
			for pattern in self._patterns:
				images = fnmatch.filter(os.listdir(class_source_dir), pattern)
				for image in images:
					source_file_name = os.path.join(class_source_dir, image)
					filename, file_extension = os.path.splitext(source_file_name)	
					new_file_name = "Class-" + class_name + "-Image-" + str(no_of_images) + file_extension
					destination_file_name = os.path.join(class_source_dir, new_file_name)
					os.rename(source_file_name, destination_file_name)
					no_of_images = no_of_images + 1

		OK = True
		return(OK)

