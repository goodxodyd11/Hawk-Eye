from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import argparse
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# load model and put into eval mode

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()
args = arg_parse()
img_size=416
conf_thres=0.8
nms_thres=0.4
config_path='config/yolov3.cfg'
weights_path=args.weightsfile
# weights_path='yolov3.weights'
class_path='config/coco.names'

model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

if __name__ == '__main__':
    
    videopath = args.video
    # videopath = "video.mp4"

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # initialize Sort object and video capture
    from sort import *
    vid = cv2.VideoCapture(videopath)
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    mot_tracker = Sort() 
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    # while(vid.isOpened()):
    pre_tracker = None
    cnt = 0;
    for ii in range(1000):
        ret, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)
        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]
                cls = classes[int(cls_pred)]
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 2)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
                result = ""
                if pre_tracker is not None:
                    for pre_x1, pre_y1, pre_x2, pre_y2, pre_obj_id, pre_cls_pred in pre_tracker:
                        if obj_id == pre_obj_id:
                            pre_y1 = int(((pre_y1 - pad_y // 2) / unpad_h) * img.shape[0])
                            pre_x1 = int(((pre_x1 - pad_x // 2) / unpad_w) * img.shape[1])
                            # print("id",obj_id)
                            # print("cur",y1)
                            # print("pre",pre_y1)
                            # print(ii)
                            if y1 < pre_y1:
                                result = "out"
                            else:
                                result = "come"
                    cv2.putText(frame, cls + "-" + str(box_h * box_w) + " " + result, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            pre_tracker = tracked_objects
        out.write(frame)
        print(ii)
    vid.release()
    out.release()
