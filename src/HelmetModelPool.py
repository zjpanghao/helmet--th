
import torch
from PIL import Image
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.nn as nn
import os
import shutil
import time
import sys
import base64
from io import BytesIO
import json
import io
from LogFactory import logger
import sys, glob
import queue
import threading

import Helmet

imsize=299
chars = ["hat", "landmark", "none", "with"]
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
loader = transforms.Compose([
    transforms.Resize(int(imsize*1)),  # scale imported image
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])  # transform it into a torch tensor

def image_toRGB(image):
  if image.mode == 'RGBA':
    logger.info("process rgba")
    r,g,b,a = image.split()
    image = Image.merge("RGB", (r, g, b))
  elif image.mode != 'RGB':
    image = image.convert("RGB") 
  return image

def image_loader(bufferimage):
  image = loader(bufferimage).unsqueeze(0)
  return image.to(device, torch.float)

class HelmetModelPool:
  def __init__(self, poolSize=5):
    self.max_ = poolSize
    self.queue_ = queue.Queue()
    for i in range (self.max_):
      tmp = HelmetModel()
      self.queue_.put(tmp)
  def get(self):
    return self.queue_.get()
  def put(self, obj):
    self.queue_.put(obj) 

class HelmetResult:
  def __init__(self):
    self.code = -1
    self.name = "none"
    self.score = 0
    self.index = 0

class HelmetModel:
  index = 0
  def __init__(self):
    self.index_ = HelmetModel.index
    HelmetModel.index = HelmetModel.index + 1
    self.model_ft = models.inception_v3(pretrained=False)
    num_ftrs = self.model_ft.fc.in_features
    aux_ftrs = self.model_ft.AuxLogits.fc.in_features
    print(num_ftrs, aux_ftrs)
    self.model_ft.fc = nn.Linear(num_ftrs, 4)
    self.model_ft.AuxLogits.fc = nn.Linear(aux_ftrs, 4)
    #print  (image_datasets['train'].classes)
    #print (model_ft)
    self.model_ft = self.model_ft.to(device)
    self.model_ft.load_state_dict(torch.load("helmet/helmet_inception_v3_stat.ft", map_location=device))
    self.model_ft.eval()   # Set model to evaluate mode

  def checkHelmet(self, image):
    result = HelmetResult()
    try:
      bufferImage = base64.b64decode(image)
    except Exception as e:
      logger.error("base64 error" + str(e))
      return result
    imageData = Image.open(io.BytesIO(bufferImage))
    imageData = image_toRGB(imageData)
    data=image_loader(imageData)
    try:
      pred=self.model_ft(data)
      p= F.softmax(pred, dim=1)
      va,inx = torch.max(p, 1)
      index = inx.item()
      score = round(va.item(), 2)
      result.name = chars[index]
      result.score = score
      result.code = 0
      result.index = index
      return result
    except Exception as  e:
      logger.error('exeption' + str(e))
      return result

