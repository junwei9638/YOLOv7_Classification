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
opt = parser.parse_args()


def create_rotated45_img(path):

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
            # img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_CUBIC)
            
            cv2.imwrite( os.path.join(r_path, img_file.split('/')[-1]), img)
    else:
        assert 'Wrong Path for Rotated Img'

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
            
            
ori_img_path = '/home/lab602.11077006/.pipeline/11077006/gt_whole/val'
predict_label_path = '/home/lab602.11077006/.pipeline/11077006/yolov5_cls/runs/predict-cls/test/labels'
rlabel_path = '/home/lab602.11077006/.pipeline/11077006/yolov7/runs/detect/not_only_car/labels'
rimg_path = ori_img_path + '_rotated'

txt_files  = glob.glob(predict_label_path  + '/*.txt')
rtxt_files = glob.glob(rlabel_path  + '/*.txt')
txt_files = sorted(txt_files)

# Directories
project=ROOT / 'runs/infere'
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
            # cv2.line(img, (xmin, ymin), (xmin, ymax), (0, 255, 255), 2)
            # cv2.line(img, (xmin, ymax), (xmax, ymax), (0, 255, 255), 2)
            # cv2.line(img, (xmax, ymax), (xmax, ymin), (0, 255, 255), 2) 
            # cv2.line(img, (xmax, ymin), (xmin, ymin), (0, 255, 255), 2) 
            
            # 旋轉的正框
            # if ( angle >= 45 and angle <= 135) or ( angle >= 225 and angle <= 315 ):
            #     cos = math.cos( math.radians( angle ) + math.pi/2 )
            #     sin = math.sin( math.radians( angle ) + math.pi/2 )
            # else :
            #     cos = math.cos( math.radians( angle ) )
            #     sin = math.sin( math.radians( angle ) )
            # print( '------------')
            # print( math.radians( angle ), angle  )
            # print( cos, sin )
            # print( '------------')
            # (x1, y1) = ( int((-w/2*cos - h/2*sin)+x), int((w/2*sin - h/2*cos)+y) )
            # (x2, y2) = ( int((w/2*cos - h/2*sin)+x), int((-w/2*sin - h/2*cos)+y) )
            # (x3, y3) = ( int((w/2*cos + h/2*sin)+x), int((-w/2*sin + h/2*cos)+y) )
            # (x4, y4) = ( int((-w/2*cos + h/2*sin)+x), int((w/2*sin + h/2*cos)+y) )
            # cv2.putText(img, str(angle), (int(x),int(y)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1) 
            # cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 1) 
            # cv2.line(img, (x3, y3), (x4, y4), (0, 0, 255), 1)
            # cv2.line(img, (x4, y4), (x1, y1), (0, 0, 255), 1)
            
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
                # 正的旋轉框
                half_width = rw / 2
                half_height = rh / 2
                ori_pt1 =  int(x + half_width), int(y - half_height)
                ori_pt2 =  int(x + half_width), int(y + half_height)
                ori_pt3 =  int(x - half_width), int(y + half_height)
                ori_pt4 =  int(x - half_width), int(y - half_height)
                
                # cv2.line(img, ori_pt1, ori_pt2, (255, 0, 0), 2)
                # cv2.line(img, ori_pt2, ori_pt3, (255, 0, 0), 2)
                # cv2.line(img, ori_pt3, ori_pt4, (255, 0, 0), 2)
                # cv2.line(img, ori_pt4, ori_pt1, (255, 0, 0), 2)
                r_cos = math.cos( math.radians(90-angle) )
                r_sin = math.sin( math.radians(90-angle) )
                pt1 = rotate_coor((x,y), ori_pt1, math.radians(360-angle+90))
                pt2 = rotate_coor((x,y), ori_pt2, math.radians(360-angle+90))
                pt3 = rotate_coor((x,y), ori_pt3, math.radians(360-angle+90))
                pt4 = rotate_coor((x,y), ori_pt4, math.radians(360-angle+90))
                
                # pt1 = int((-rw/2*r_cos - rh/2*r_sin)+x), int((rw/2*r_sin - rh/2*r_cos)+y)
                # pt2 = int((rw/2*r_cos - rh/2*r_sin)+x), int((-rw/2*r_sin - rh/2*r_cos)+y)
                # pt3 = int((rw/2*r_cos + rh/2*r_sin)+x), int((-rw/2*r_sin + rh/2*r_cos)+y)
                # pt4 = int((-rw/2*r_cos + rh/2*r_sin)+x), int((rw/2*r_sin + rh/2*r_cos)+y)   
                cv2.line(img, pt1, pt2, (0, 255, 255), 2)
                cv2.line(img, pt2, pt3, (0, 255, 255), 2)
                cv2.line(img, pt3, pt4, (0, 255, 255), 2)
                cv2.line(img, pt4, pt1, (0, 255, 255), 2)
                # cv2.putText(img, str((int(angle))), (int(x),int(y)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.line(img, (xmin, ymin), (xmin, ymax), (0, 255, 255), 2)
                cv2.line(img, (xmin, ymax), (xmax, ymax), (0, 255, 255), 2)
                cv2.line(img, (xmax, ymax), (xmax, ymin), (0, 255, 255), 2) 
                cv2.line(img, (xmax, ymin), (xmin, ymin), (0, 255, 255), 2)
        # if r_img is not None:
        #     cv2.imwrite( os.path.join(save_dir, str(txtfile).split('/')[-1][:-4]+'_rotat' + '.png'), r_img)
        cv2.imwrite( os.path.join(save_dir, str(txtfile).split('/')[-1][:-4] + '.png'), img)
        print( str(txtfile).split('/')[-1] )
