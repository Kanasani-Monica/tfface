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
python decode_train_dataset.py /workspace/datasets/images/ms1m_20K/ /workspace/MS-Celeb-1M/EntityList/MID2Name.tsv /workspace/MS-Celeb-1M/LowShotLearning/BaseSet/20K.tsv 
"""

########################################################################################################################################################################################################
# File format for the entity list file.

# Files -
# MID2Name.tsv 

# Text files, each line is an record containing 2 columns, delimited by TAB.
# Column1: Freebase MID
# Column2: 'Name String'@Language

# Some statistics -
# Number of lines - 3,481,187
# Number of unique MIDs - 1,000,000
# Total file size - 110MB
########################################################################################################################################################################################################

########################################################################################################################################################################################################
# File format for low shot learning MS-Celeb-1M image files.

# Files -
# MID2Name.tsv 

# File format: text files, each line is an image record containing 5 columns, delimited by TAB.
# Column1: Image ID
# Column2: FaceData_Base64Encoded
# Column3: Freebase MID
# Column4: ImageSearchRank
# Column5: ImageURL
########################################################################################################################################################################################################

def main(args):
	output_dir = os.path.expanduser(args.output_dir)
    	if(not os.path.exists(output_dir)):
        	os.mkdir(output_dir)

	mid_name = {}
	mid_count = {}
	for line in args.mid_file:
		fields = line.split('\t')
		freebase_mid = fields[0]
		name_language = fields[1]
		fields = name_language.split('@')
		name = fields[0]
		name = name.replace(' ', '_')
		name = name.strip('"')
		language = fields[1].strip()		
		mid_name[freebase_mid ] = name
		mid_count[freebase_mid] = 0

	print('Number of celebrities = ' + str(len(mid_name)))

    	number_of_images = 0
    	celebrity_count = 0
    	for tsv_file in args.tsv_files:
        	for line in tsv_file:
            		fields = line.split('\t')
            		#class_dir = mid_name[fields[2]]
            		class_dir = fields[2]   
            		image_search_rank = fields[3]             
            		#img_name = class_dir + '-' + str(mid_count[fields[2]]) + '.' + args.output_format
            		img_name = class_dir + '-' + str(image_search_rank) + '.' + args.output_format
                        mid_count[fields[2]] = mid_count[fields[2]] + 1
            		img_string = fields[1]
            		img_dec_string = base64.b64decode(img_string)
            		img_data = np.fromstring(img_dec_string, dtype=np.uint8)
            		img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

            		if args.size:
                			img = misc.imresize(img, (args.size, args.size), interp='bilinear')
                			img = cv2.resize(img, (args.size, args.size), interpolation = cv2.INTER_CUBIC)

            		full_class_dir = os.path.join(output_dir, class_dir)
            		if not os.path.exists(full_class_dir):
                			os.mkdir(full_class_dir)
                			celebrity_count = celebrity_count + 1

            		full_path = os.path.join(full_class_dir, img_name)
            		cv2.imwrite(full_path, img)            		
            		number_of_images += 1

	print('Number of celebrities = ' + str(celebrity_count) + ' with = ' + str(number_of_images) + ' images.')
		
if __name__ == '__main__':
  
	parser = argparse.ArgumentParser()
	parser.add_argument('output_dir', type=str, help='Output base directory for the image dataset')
	parser.add_argument('mid_file', type=argparse.FileType('r'), help='Input MID to celebrity name TSV file name')
	parser.add_argument('tsv_files', type=argparse.FileType('r'), nargs='+', help='Input TSV file name(s)')
	parser.add_argument('--size', type=int, help='Images are resized to the given size')
	parser.add_argument('--output_format', type=str, help='Format of the output images', default='png', choices=['png', 'jpg'])
	
	main(parser.parse_args())


