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
from LogFactory import logger
import sys, glob

import Helmet
from ttypes import HelmetCheckResult
import HelmetModelPool

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import threading

imsize=299
chars = ["hat", "landmark", "none", "with"]
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
loader = transforms.Compose([
    transforms.Resize(int(imsize*1)),  # scale imported image
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])  # transform it into a torch tensor


class HelmetHandler:
    def __init__(self, pool):
      logger.info("init pool")
      self.helmetPool = pool
    def pingHelmet(self):
      logger.info(str(threading.current_thread().getName()) + "__" + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ) + "ping")

    def checkHelmet(self, image):
       #result = HelmetResult(-1, False, 0) 
      time_1 = time.time()
      model_ft = self.helmetPool.get()
      try:
        result = model_ft.checkHelmet(image)
        helmetResult = HelmetCheckResult(result.code, result.index, result.name, result.score)
        dure = time.time() - time_1
        logger.info(str(threading.current_thread().getName()) + "__" + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ) + "___" + str(result.code) + ":" 
            + result.name + ":" + str(result.index) + ":" +  str(result.score) +"_____" + str(dure) + "s" )
        return helmetResult 
      except Exception as  e:
        helmetResult = HelmetCheckResult(-1)
        logger.error('exeption' + str(e))
        return  helmetResult
      finally:
        self.helmetPool.put(model_ft)
if __name__ == '__main__':
  pool = HelmetModelPool.HelmetModelPool(10)
  handler = HelmetHandler(pool)
  processor = Helmet.Processor(handler)
  transport = TSocket.TServerSocket(port=9090)
  tfactory = TTransport.TBufferedTransportFactory()
  pfactory = TBinaryProtocol.TBinaryProtocolFactory()
  server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
  print ('Starting the server...')
  server.serve()
  print ('done.')
