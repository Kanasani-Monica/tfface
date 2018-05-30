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

r"""Analyse the dataset.

Usage:
```shell

$ python analyse_dataset.py \
	--source_dir=/datasets/mtcnn_images/face_images

$ python analyse_dataset.py \
	--source_dir=/datasets/mtcnn_images/face_images \
	--min_no_of_images=30 \
	--backup_dir=/datasets/mtcnn_images/face_images_BACKUP
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
	parser.add_argument('--min_no_of_images', type=int, help='Minimum number of images needed.', default=1)
	parser.add_argument('--backup_dir', type=str, help='Output directory whete data source backup is taken.', default=None)

	return(parser.parse_args(argv))

def main(args): 

	if(not args.source_dir):
		raise ValueError('You must supply input source directory with --source_dir.')
	if(not os.path.exists(args.source_dir)):
		print('The data source directory is missing. Error analysing the data source without the source directory.')
		return

	dataset_analyzer = DatasetAnalyzer()
	OK, data_source = dataset_analyzer.analyse(args.source_dir, args.backup_dir, args.min_no_of_images, print_results=True)


if __name__ == '__main__':

	main(parse_arguments(sys.argv[1:]))


