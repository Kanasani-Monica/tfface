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

import tensorflow as tf
import os
import sys
import cv2

from datasets import dataset_utils
from nets import nets_factory as network_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

class Classifier(object):

	def __init__(self, model_name='inception_v3'):	
	
		self.model_name = model_name

		self.has_dataset = False
		self.has_model = False
		
	def load_dataset(self, dataset_dir):
		self.has_dataset = False

		self.labels_to_names = dataset_utils.read_label_file(dataset_dir)
		self.number_of_classes = len(self.labels_to_names)

		if(self.number_of_classes > 0):
			self.has_dataset = True

		return(self.has_dataset)

	def load_model(self, checkpoint_path, gpu_memory_fraction):
		return(self.load_model(checkpoint_path, self.model_name, gpu_memory_fraction))

	def load_model(self, checkpoint_path, model_name, gpu_memory_fraction):
		self.has_model = False

		if(not self.has_dataset):
			return(self.has_model)

		self.model_name = model_name

		if tf.gfile.IsDirectory(checkpoint_path):
			self.model_path = tf.train.latest_checkpoint(checkpoint_path)
		else:
			self.model_path = checkpoint_path

		self.graph = tf.Graph()
		with self.graph.as_default():

			image_preprocessing_fn = preprocessing_factory.get_preprocessing(model_name, is_training=False)
			network_fn = network_factory.get_network_fn(model_name, num_classes=self.number_of_classes, is_training=False)
			network_image_size = network_fn.default_image_size		
		
			self.input_tensor = tf.placeholder(tf.uint8, (None, None, 3), 'input')
			processed_image = image_preprocessing_fn(self.input_tensor, network_image_size, network_image_size)
			tensor_image  = tf.expand_dims(processed_image, 0)

			tensor_logits, end_points = network_fn(tensor_image)
			self.tensor_probabilities = tf.nn.softmax(tensor_logits)

			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        		self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

			init_fn = slim.assign_from_checkpoint_fn(self.model_path, slim.get_model_variables(network_factory.model_name_map[model_name]))
			init_fn(self.session) 

		self.has_model = True

		return(self.has_model)

	def classify(self, input_image, use_top=5, print_results=False):

		class_names_probabilities = []

		class_probabilities = []
		try:
	 		feed_dict = {self.input_tensor:input_image}
			class_probabilities = self.session.run(self.tensor_probabilities, feed_dict=feed_dict)
       		except (IOError, ValueError, IndexError) as error:
	       		return(class_names_probabilities)

		if(len(class_probabilities) == 0):		
			return(class_names_probabilities)			

		class_probabilities = class_probabilities[0, 0:]
		sorted_indices = [i[0] for i in sorted(enumerate(-class_probabilities), key=lambda x:x[1])]		

		for index in range(use_top):
			class_name = self.labels_to_names[sorted_indices[index]]
			class_probability = class_probabilities[sorted_indices[index]]
			class_names_probabilities.append([ class_name, class_probability])
			if(print_results):
				print("Class is - " + class_name + " with probability - " + str(class_probability) )
		
		return(class_names_probabilities)

