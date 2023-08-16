# YOLOv7_Backbone_Angle_Classification


## Step1: Prepare Your Dataset .
HBB Dataset Format: angle(0-359), x, y, w, h

## Step2: Train an Angle Classification Model .
You can customize the parameters.
```
python classify/train.py --data data/rotate.yaml --epochs 40 --img 224 --cfg models/yolov7_backbone_cspElan.yaml --hyp data/hyps/hyp_rotate.yaml --csl 5 --name ${your_name} --workers 6 --batch
-size 32 --optimizer AdamW --device 0 --thresh 5
```

## Step3: Model Inference .
```
python classify/predict.py --weights ${your_model_path} --source ${your_images_txt} --name ${your_name} --thresh 5 --data data/rotate.yaml
```
## Step4: Generate 45-degree-rotation pics . 
After inference we got angle info and need to turn HBB into OBB, but there is a flaw in my our formula of calculating height and width of OBB.
So you need to make original pics rotate 45 degree and send them into any object detection model which can identify vehicles and generate new 45-degree-rotation label.
```
python classify/create_45_img.py --path  ${your_imgz}
```

## Step5: Choose Box and Draw
```
python classify/choosebox_and_draw.py --name ${your_name}  --ori_img ${your_imgz} --pred_label ${label_you_inferenced} --rlabel ${label_you_inferenced_45rotated}
```

## Angle Classification Model
[angle_cls](https://github.com/junwei9638/YOLOv7_Classification/blob/0b643dff766a03f2714dcc3541a75a525de49486/anlge_cls.pt)
