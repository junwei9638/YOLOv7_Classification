import os
import math
import cv2
import sys
import glob
import numpy as np
from pathlib import Path
import imutils
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

parser = argparse.ArgumentParser(description="Remove_Pedestrians")
parser.add_argument("--path", type=str, required=True)
opt = parser.parse_args()

path = opt.path

if os.path.exists( path ):
    r_path = str( path )+'_rotated'
    if not os.path.exists( r_path ):
        os.mkdir( r_path )
    jpg_files  = glob.glob(path  + '/*.jpg')
    png_files  = glob.glob(path  + '/*.png')
    img_files = jpg_files + png_files
    
    for img_file in img_files :
        img = cv2.imread( img_file )
        txt_file = img_file[:-3] + 'txt'
        w = img.shape[1]
        h = img.shape[0]
        rw = int((w + h) * math.cos( math.radians( 45 ) ))
        rh = rw
        img = imutils.rotate_bound(img, -45)
        
        cv2.imwrite( os.path.join(r_path, img_file.split('/')[-1]), img)
else:
    assert 'Wrong Path for Rotated Img'