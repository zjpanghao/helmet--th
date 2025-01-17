#!/usr/bin/env python
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
from concurrent.futures import ThreadPoolExecutor
import logging
import LogFactory
logger = LogFactory.Logger("helmet").getLogger()
import sys, glob

import Helmet

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

logger = LogFactory.Logger("helmet").getLogger()
imsize=299
chars = ["hat", "landmark", "none", "with"]
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model_ft = models.inception_v3(pretrained=False)
num_ftrs = model_ft.fc.in_features
aux_ftrs = model_ft.AuxLogits.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(chars))
model_ft.AuxLogits.fc = nn.Linear(aux_ftrs, len(chars))
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load("helmet/helmet_inception_v3_stat.ft", map_location=device))
model_ft.eval()   # Set model to evaluate mode
executor = ThreadPoolExecutor(max_workers=10)
loader = transforms.Compose([
    transforms.Resize(int(imsize*1)),  # scale imported image
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])  # transform it into a torch tensor

def image_toRGB(image):
  if image.mode == 'RGBA':
    r,g,b,a = image.split()
    image = Image.merge("RGB", (r, g, b))
  elif image.mode != 'RGB':
    image = image.convert("RGB") 
  return image

def image_loader(bufferimage):
  image = loader(bufferimage).unsqueeze(0)
  return image.to(device, torch.float)

class HelmetHandler:
    def __init__(self):
      pass

    def checkHelmet(self, image):
       #result = HelmetResult(-1, False, 0) 
      logger.info("recv helmet request" + str(len(image)))
      try :
        bufferImage = base64.b64decode(image)
      except Exception as e:
        logger.error(str(e))
        return -1
      imageData = Image.open(io.BytesIO(bufferImage))
      imageData = image_toRGB(imageData)
      data=image_loader(imageData)
      try:
        pred=model_ft(data)
        p= F.softmax(pred, dim=1)
        va,inx = torch.max(p, 1)
        index = inx.item()
        score = round(va.item(), 2)
        if score < 0.5:
          return -2
        park = chars[index]
        res = {"error_code" : 0, "results" : [{"name": park, "score": score}]}
        return index
      except Exception as  e:
        logger.error('exeption' + str(e))
        return -1
      finally:
        result = json.dumps(res)
        logger.info(result)

handler = HelmetHandler()
processor = Helmet.Processor(handler)
transport = TSocket.TServerSocket(port=9090)
tfactory = TTransport.TBufferedTransportFactory()
pfactory = TBinaryProtocol.TBinaryProtocolFactory()

server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)

print ('Starting the server...')
server.serve()
print ('done.')
