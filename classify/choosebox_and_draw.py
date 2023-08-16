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

from utils.general import increment_path


parser = argparse.ArgumentParser(description="Remove_Pedestrians")
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--ori_img", type=str, required=True, help='inference imgz')
parser.add_argument("--predict_label", type=str, required=True, help='inference label')
parser.add_argument("--rlabel", type=str, required=True, help='rotated 45 label which inferenced by another model')
opt = parser.parse_args()


def rotate_point(x, y, w, h, angle):
    # Convert the angle to radians
    angle_rad = math.radians(angle)
    rw = int((w + h) * math.cos( math.radians( angle ) ))
    rh = rw
    x -= w/2
    y -= h/2

    # Calculate the new coordinates after rotation
    x_rotated = x * math.cos(angle_rad) + y * math.sin(angle_rad)
    y_rotated = -x * math.sin(angle_rad) + y * math.cos(angle_rad)
    
    x_rotated += rw/2
    y_rotated += rh/2 

    return int(x_rotated), int(y_rotated)

def rotate_coor(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)

def check_neareat_coord( rx, ry, img, file):
    with open(file, 'r') as txt:
        dis_list = []
        lines = txt.readlines()
        for i, line in enumerate( lines ):
            angle = float(int(line.split( ' ' )[0]) )
            x = float(line.split( ' ' )[1]) * img.shape[1]
            y = float(line.split( ' ' )[2]) * img.shape[0]
            w = float(line.split( ' ' )[3]) * img.shape[1]
            h = float(line.split( ' ' )[4]) * img.shape[0]
            dis_list.append( (rx-x)**2 + (ry-y)**2 )
        min_value = dis_list.index(min(dis_list))
        w,h = float(lines[min_value].split( ' ' )[3]), float(lines[min_value].split( ' ' )[4])
        return w,h
            

ori_img_path = opt.ori_img
predict_label_path = opt.predict_label
rlabel_path = opt.rlabel
rimg_path = ori_img_path + '_rotated'

txt_files  = glob.glob(predict_label_path  + '/*.txt')
rtxt_files = glob.glob(rlabel_path  + '/*.txt')
txt_files = sorted(txt_files)

# Directories
project=ROOT / 'runs/inference'
opt.name = opt.name if opt.name != '' else 'exp'
save_dir = increment_path(Path(project) / opt.name, exist_ok=False)  # increment run
save_dir.mkdir(parents=True, exist_ok=True)  # make dir

for txtfile in txt_files :
    with open( txtfile, 'r') as txt:
        ori_filename = os.path.join( ori_img_path, str(txtfile).split('/')[-1][:-4] )
        img_file = ori_filename + '.jpg' if os.path.exists( ori_filename + '.jpg' ) else ori_filename + '.png'
        img = cv2.imread( img_file )
        
        for i, line in enumerate( txt.readlines() ):
            angle = float(int(line.split( ' ' )[0]) )
            x = float(line.split( ' ' )[1]) * img.shape[1]
            y = float(line.split( ' ' )[2]) * img.shape[0]
            w = float(line.split( ' ' )[3]) * img.shape[1]
            h = float(line.split( ' ' )[4]) * img.shape[0]
            if w > h :
                w, h = h, w
                xmin = int( x - h/2  ) 
                xmax = int( x + h/2  ) 
                ymin = int( y - w/2  ) 
                ymax = int( y + w/2  ) 
            else :
                xmin = int( x - w/2  ) 
                xmax = int( x + w/2  ) 
                ymin = int( y - h/2  ) 
                ymax = int( y + h/2  ) 
            
            # restrict angle to 0-90
            restrict_angle180 = angle if angle < 180 else angle - 180
            restrict_angle90 = restrict_angle180 if restrict_angle180 <= 90 else restrict_angle180 - 90
            restrict_angle45 = restrict_angle90 if restrict_angle90 <= 45 else 90 - restrict_angle90
            
            if restrict_angle90 != 0 and restrict_angle90 != 90:
                cos = math.cos( math.radians( restrict_angle45 ) )
                sin = math.sin( math.radians( restrict_angle45 ) )
                
                if restrict_angle90 >= 35 and restrict_angle90 <= 55:
                    rfile = os.path.join( rlabel_path, str(txtfile).split('/')[-1] )
                    r_img = cv2.imread( os.path.join( rimg_path,img_file.split('/')[-1] ) )
                    rx, ry = rotate_point(x, y, img.shape[1], img.shape[0], 45)
                    rw, rh = check_neareat_coord( rx, ry, r_img, rfile)
                    rw *= img.shape[1]
                    rh *= img.shape[1]
                else:
                    rw = ( w*cos - h*sin ) / ( cos**2 - sin**2)
                    rh = ( h - rw*sin ) / cos 
                
                if rw > rh :
                    rw, rh = rh, rw
                    
                half_width = rw / 2
                half_height = rh / 2
                
                ori_pt1 =  int(x + half_width), int(y - half_height)
                ori_pt2 =  int(x + half_width), int(y + half_height)
                ori_pt3 =  int(x - half_width), int(y + half_height)
                ori_pt4 =  int(x - half_width), int(y - half_height)
                
                r_cos = math.cos( math.radians(90-angle) )
                r_sin = math.sin( math.radians(90-angle) )
                pt1 = rotate_coor((x,y), ori_pt1, math.radians(360-angle+90))
                pt2 = rotate_coor((x,y), ori_pt2, math.radians(360-angle+90))
                pt3 = rotate_coor((x,y), ori_pt3, math.radians(360-angle+90))
                pt4 = rotate_coor((x,y), ori_pt4, math.radians(360-angle+90))
        
                cv2.line(img, pt1, pt2, (0, 255, 255), 2)
                cv2.line(img, pt2, pt3, (0, 255, 255), 2)
                cv2.line(img, pt3, pt4, (0, 255, 255), 2)
                cv2.line(img, pt4, pt1, (0, 255, 255), 2)
            else:
                cv2.line(img, (xmin, ymin), (xmin, ymax), (0, 255, 255), 2)
                cv2.line(img, (xmin, ymax), (xmax, ymax), (0, 255, 255), 2)
                cv2.line(img, (xmax, ymax), (xmax, ymin), (0, 255, 255), 2) 
                cv2.line(img, (xmax, ymin), (xmin, ymin), (0, 255, 255), 2)
        cv2.imwrite( os.path.join(save_dir, str(txtfile).split('/')[-1][:-4] + '.png'), img)
        print( str(txtfile).split('/')[-1] )
