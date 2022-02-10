
#########################################################################

from selectivesearch import selective_search as sch
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms
from torchvision import transforms, models
from torch_snippets import *
import numpy as np
import pandas as pd
import cv2

ROOT = 'C:/Users/Epsilon/Desktop/images/open-images-bus-trucks/'
chart_raw = pd.read_csv(ROOT+'df.csv')

#########################################################################

class OpenImages(Dataset): # Process image+box data
    def __init__(self, chart, folder): # added directory option
        self.root = folder
        self.chart = chart
        self.data = chart['ImageID'].unique()  # list of image ids
    
    def __getitem__(self, i): # object detection dataset : image + box + label 
        img_id = self.data[i]
        img_path = f'{self.root}/{img_id}.jpg'
        image = cv2.imread(img_path,1)[...,::-1]    # BGR2RGB
        h, w, _ = image.shape
        row = self.chart[self.chart['ImageID']==img_id]
        box = row['XMin,YMin,XMax,YMax'.split(',')].values    # relative position of the box 
        box = (box*np.array([w,h,w,h])).astype(np.uint16).tolist()    # absolute position of the box
        labels = row['LabelName'].values.tolist()    # label of the object
        return image, box, labels, img_path

    def __len__(self):
        return len(self.images)
    

def get_candidates(img): # get candidate boxes from image
    label, regions = sch(img, scale=200, min_size=100)
    area = np.prod(img.shape[:2])
    candidates = []
    for r in regions:
        if r['rect'] in candidates : continue
        if r['size'] < 0.05*area : continue
        if r['size'] > area : continue    # if candidate is too big or small, eliminated
        candidates.append(list(r['rect']))   # append appropriate box
    return candidates

def get_iou(boxA, boxB, eps=1e-5): # IoU = score( overlap area / combined area )
    x1, y1 = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    x2, y2 = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    w, h = (x2-x1), (y2-y1)
    if (w<0 or h<0) : return 0.0
    areaAnB = w*h
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxB[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    areaAoB = areaA+areaB-areaAnB+eps
    return areaAnB/areaAoB


