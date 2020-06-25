import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np

import cv2
import random
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="./output/faster-rcnn-beagle.pt",
                help="path to the model")
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.7, 
                help="confidence to keep predictions")
args = vars(ap.parse_args())

CLASS_NAMES = ["__background__", "beagle"]
def get_prediction(img_path, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.
    
    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img).to(device)
    pred = model([img])
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]

    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]
    return pred_boxes, pred_class, pred_score
   
def detect_object(img_path, confidence=0.5, rect_th=2, text_size=1, text_th=1):
    """
    object_detection_api
      parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
        - rect_th - thickness of bounding box
        - text_size - size of the class label text
        - text_th - thichness of the text
      method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written 
          with opencv
        - the final image is displayed
    """
    boxes, pred_cls, pred_score = get_prediction(img_path, confidence)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(len(boxes))
    for i in range(len(boxes)):
      cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
      cv2.putText(img,pred_cls[i]+": "+str(round(pred_score[i],3)), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
  
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(args["model"])
    img_path = args["image"]
    detect_object(img_path, confidence=args["confidence"])
    