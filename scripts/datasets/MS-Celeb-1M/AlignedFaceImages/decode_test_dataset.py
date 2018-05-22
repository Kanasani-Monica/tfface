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

import numpy as np
import base64
import sys
import os
import cv2
import argparse

"""
python decode_test_dataset.py /workspace/datasets/images/MS-Celeb-1M/test_aligned --tsv_files /workspace/datasets/MS-Celeb-1M/test/aligned_50K.tsv 1>0.1 2>0.2 &
"""

########################################################################################################################################################################################################
# File format for evaluating performance of face aligned MS-Celeb-1M images.

# Files -
# aligned_50K.tsv

# Text files, each line is an image record containing 2 columns, delimited by TAB.
# Column1: Line number
# Column2: FaceData_Base64Encoded d Data
########################################################################################################################################################################################################

def main(args):
	output_dir = os.path.expanduser(args.output_dir)
    	if(not os.path.exists(output_dir)):
        	os.mkdir(output_dir)

   	number_of_images = 0
	celebrity_count = 0
    	for tsv_file in args.tsv_files:
		print('Processing file - ', tsv_file)
        	for line in tsv_file:
            		fields = line.split('\t')
			line_number = fields[0] 
       			img_string = fields[1]

       			image_file_name = str(line_number) + '.' + args.output_format
       			img_dec_string = base64.b64decode(img_string)
       			image_data = np.fromstring(img_dec_string, dtype=np.uint8)
       			input_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

       			if args.size:
               			input_image = cv2.resize(input_image, (args.size, args.size), interpolation = cv2.INTER_CUBIC)

      			full_path = os.path.join(output_dir, image_file_name)
       			cv2.imwrite(full_path, input_image)
       			number_of_images += 1

	print('Number of input images - ' + str(number_of_images) + '.')


if __name__ == '__main__':
  
	parser = argparse.ArgumentParser()
	parser.add_argument('output_dir', type=str, help='Output base directory for the image dataset')
	parser.add_argument('--tsv_files', type=argparse.FileType('r'), nargs='+', help='Input TSV file name(s)')
	parser.add_argument('--size', type=int, help='Images are resized to the given size')
	parser.add_argument('--output_format', type=str, help='Format of the output images', default='png', choices=['png', 'jpg'])
	
	main(parser.parse_args())


