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

r"""Create class name files for parallel processing.

Usage:
```shell

$ python create_class_files.py \
	--source_dir=/datasets/mtcnn_images/FaceScrub \
	--base_file_name=class-file- 
	--no_of_subsets=4
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
	parser.add_argument('--source_dir', type=str, help='The directory from where input data is read for processing.', default=None)
	parser.add_argument('--no_of_subsets', type=int, help='Number of subsets in which the data source is divided into.', default=4)
	parser.add_argument('--base_file_name', type=str, help='Initial base part of the filename used for storing subset of classes.', default=DatasetAnalyzer.default_base_file_name)
	return(parser.parse_args(argv))


def main(args): 

	if(not args.source_dir):
		raise ValueError('You must supply input source directory with --source_dir.')
	if(not os.path.exists(args.source_dir)):
		print('The data source directory is missing. Error analysing the data source without the source directory.')
		return

	dataset_analyzer = DatasetAnalyzer()
	OK = dataset_analyzer.create_class_files(args.source_dir, args.no_of_subsets, args.base_file_name)
	if(OK):
		print("The dataset " + args.source_dir + " is divided into " + str(args.no_of_subsets) + " number of parts.")
	else:
		print("Error dividing the dataset " + args.source_dir + " .")
	

if __name__ == '__main__':

	main(parse_arguments(sys.argv[1:]))


