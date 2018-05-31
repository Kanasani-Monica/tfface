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

r"""Extract classes from the dataset.

Usage:
```shell

$ python extract_classes.py \
	--source_dir=/datasets/mtcnn_images/face_images \
	--target_dir=/datasets/mtcnn_images/extracted_face_images \
	--number_of_classes=100
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from tfface.tools.DatasetAnalyzer import DatasetAnalyzer

def parse_arguments(argv):

	parser = argparse.ArgumentParser()
	parser.add_argument('--source_dir', type=str, help='Input data source directory from where data is read for processing.', default=None)
	parser.add_argument('--target_dir', type=str, help='The directory where the output cleaned data is saved.', default=None)
	parser.add_argument('--number_of_classes', type=int, help='Number of classes to extract from the dataset.', default=100)
	return(parser.parse_args(argv))

def main(args): 
	if(not args.source_dir):
		raise ValueError('You must supply the source directory with --source_dir')
	if(not os.path.exists(args.source_dir)):
		print('The source directory is missing. Error cleaning the data source without the source directory.')
		return

	if(not args.target_dir):
		raise ValueError('You must supply the target directory with --target_dir')

	dataset_analyzer = DatasetAnalyzer()
	OK = dataset_analyzer.extract_classes(args.source_dir, args.target_dir, args.number_of_classes)
	if(OK):
		print(str(args.number_of_classes) + " classes are extracted from the dataset " + args.source_dir + ".")
	else:
		print("Error extracting classes from the dataset " + args.source_dir + " .")

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))


