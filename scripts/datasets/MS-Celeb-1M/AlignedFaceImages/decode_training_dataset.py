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

"""
python decode_training_dataset.py /workspace/datasets/images/ms1m_aligned-00/ --tsv_files /workspace/MS-Celeb-1M/AlignedFaceImages/MsCelebV1-Faces-Aligned.part.00.tsv 1>00.1 2>00.2 &

python decode_training_dataset.py /workspace/datasets/images/ms1m_aligned-01/ --tsv_files /workspace/MS-Celeb-1M/AlignedFaceImages/MsCelebV1-Faces-Aligned.part.01.tsv 1>01.1 2>01.2 &

python decode_training_dataset.py /workspace/datasets/images/ms1m_aligned-02/ --tsv_files /workspace/MS-Celeb-1M/AlignedFaceImages/MsCelebV1-Faces-Aligned.part.02.tsv 1>02.1 2>02.2 &

python decode_training_dataset.py /workspace/datasets/images/ms1m_aligned-03/ --tsv_files /workspace/MS-Celeb-1M/AlignedFaceImages/MsCelebV1-Faces-Aligned.part.03.tsv 1>03.1 2>03.2 &

python decode_training_dataset.py /workspace/datasets/images/ms1m_aligned-04/ --tsv_files /workspace/MS-Celeb-1M/AlignedFaceImages/MsCelebV1-Faces-Aligned.part.04.tsv 1>04.1 2>04.2 &

python decode_training_dataset.py /workspace/datasets/images/ms1m_aligned/ --tsv_files /workspace/MS-Celeb-1M/AlignedFaceImages/MsCelebV1-Faces-Aligned.part.00.tsv /workspace/MS-Celeb-1M/AlignedFaceImages/MsCelebV1-Faces-Aligned.part.01.tsv /workspace/MS-Celeb-1M/AlignedFaceImages/MsCelebV1-Faces-Aligned.part.02.tsv /workspace/MS-Celeb-1M/AlignedFaceImages/MsCelebV1-Faces-Aligned.part.03.tsv /workspace/MS-Celeb-1M/AlignedFaceImages/MsCelebV1-Faces-Aligned.part.04.tsv
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
# File format for face aligned MS-Celeb-1M image files.

# Files -
# MsCelebV1-Faces-Aligned.part.00.tsv 
# MsCelebV1-Faces-Aligned.part.01.tsv 
# MsCelebV1-Faces-Aligned.part.02.tsv 
# MsCelebV1-Faces-Aligned.part.03.tsv 
# MsCelebV1-Faces-Aligned.part.04.tsv 

# Text files, each line is an image record containing 7 columns, delimited by TAB.
# Column1: Freebase MID
# Column2: ImageSearchRank
# Column3: ImageURL
# Column4: PageURL
# Column5: FaceID
# Column6: FaceRectangle_Base64Encoded (four floats, relative coordinates of UpperLeft and BottomRight corner)
# Column7: FaceData_Base64Encoded d Data

# Some statistics -
# Number of entities - 99,892
# Number of Lines - 8,456,240
# Image resolution - up to 300*300
# Average images per entity - 85
# Total file size (uncompressed) - 89GB
########################################################################################################################################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import base64
import sys
import os
import cv2
import argparse

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
            		class_dir = fields[0] 
            		image_search_rank = fields[1] 
       			img_name = class_dir + '-' + str(image_search_rank) + '.' + args.output_format
       			img_string = fields[6]
       			img_dec_string = base64.b64decode(img_string)
       			img_data = np.fromstring(img_dec_string, dtype=np.uint8)
       			img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

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
	parser.add_argument('--tsv_files', type=argparse.FileType('r'), nargs='+', help='Input TSV file name(s)')
	parser.add_argument('--output_format', type=str, help='Format of the output images', default='png', choices=['png', 'jpg'])
	
	main(parser.parse_args())


