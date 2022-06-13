# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

#

# NVIDIA CORPORATION and its licensors retain all intellectual property

# and proprietary rights in and to this software, related documentation

# and any modifications thereto.  Any use, reproduction, disclosure or

# distribution of this software and related documentation without an express

# license agreement from NVIDIA CORPORATION is strictly prohibited.



"""Generate images using pretrained network pickle."""



import os

import re

from typing import List, Optional

import torchvision

import torch.nn as nn 

import torchvision.transforms as transforms

import click

import dnnlib

import numpy as np

import PIL.Image

import torch

from torch import linalg as LA

#import clip

from PIL import Image

import legacy

import torch.nn.functional as F

#import cv2

import pandas as pd 

import matplotlib.pyplot as plt

from torch_utils import misc

from torch_utils import persistence

from torch_utils.ops import conv2d_resample

from torch_utils.ops import upfirdn2d

from torch_utils.ops import bias_act

from torch_utils.ops import fma

import random

import math

import time

import pickle
from tqdm import tqdm
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

import copy
#import tensorflow as tf
import numpy as np
import requests
import collections
import os
import zipfile
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from io import BytesIO
from e4e_utils import image_to_s, load_net
from generate_utils import gen_im
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

attribute = "Male"

def get_path_from_bucket(bucket_url, path):
  r = requests.get(bucket_url)
  filename = os.path.split(bucket_url)[-1].replace('.zip', '')
  zip_ref = zipfile.ZipFile(BytesIO(r.content))
  zip_ref.extractall(path)
  return os.path.join(path, filename)


def block_forward(self, x, img, ws, shapes, force_fp32=False, fused_modconv=None, **layer_kwargs):

        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])

        w_iter = iter(ws.unbind(dim=1))

        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32

        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        if fused_modconv is None:

            with misc.suppress_tracer_warnings(): # this value will be treated as a constant

                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        

        # Input.

        if self.in_channels == 0:

            x = self.const.to(dtype=dtype, memory_format=memory_format)

            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])

        else:

            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])

            x = x.to(dtype=dtype, memory_format=memory_format)



        # Main layers.

        if self.in_channels == 0:

            x = self.conv1(x, next(w_iter)[...,:shapes[0]], fused_modconv=fused_modconv, **layer_kwargs)

        elif self.architecture == 'resnet':

            y = self.skip(x, gain=np.sqrt(0.5))

            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)

            x = y.add_(x)

        else:

            x = self.conv0(x, next(w_iter)[...,:shapes[0]], fused_modconv=fused_modconv, **layer_kwargs)

            x = self.conv1(x, next(w_iter)[...,:shapes[1]], fused_modconv=fused_modconv, **layer_kwargs)



        # ToRGB.

        if img is not None:

            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])

            img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.is_last or self.architecture == 'skip':

            y = self.torgb(x, next(w_iter)[...,:shapes[2]], fused_modconv=fused_modconv)

            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)

            img = img.add_(y) if img is not None else y



        assert x.dtype == dtype

        assert img is None or img.dtype == torch.float32

        return x, img



def unravel_index(index, shape):

    out = []

    for dim in reversed(shape):

        out.append(index % dim)

        index = index // dim

    return tuple(reversed(out))





#----------------------------------------------------------------------------



def num_range(s: str) -> List[int]:

    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''



    range_re = re.compile(r'^(\d+)-(\d+)$')

    m = range_re.match(s)

    if m:

        return list(range(int(m.group(1)), int(m.group(2))+1))

    vals = s.split(',')

    return [int(x) for x in vals]


from torchvision.models.mobilenet import MobileNetV2

def load_classifier(path, load_state = True):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = MobileNetV2()

    model.classifier[-1] = torch.nn.Sequential(torch.nn.Linear(in_features=1280, out_features=2))

    model = model.to(device)
    
    if load_state:
      model.load_state_dict(torch.load(path))
    
    return model

#----------------------------------------------------------------------------

class EnsembleModel(nn.Module):
    def __init__(self, model_path, channels):
        super().__init__()
        self.model = load_classifier(model_path, load_state = True)
        self.channels = channels
        self.e4e = load_net()
              
        for i in self.model.parameters():
            i.requires_grad = False  
        for i in self.e4e.parameters():
            i.requires_grad = False  
            

    def forward(self, img_name):
          
          batch = len(img_name)
          img_encodeds = []
          for name in   img_name:
              path = os.path.join("/home/fair/fair/celeba_data/celeba_s/",name +".pt")
              img_path = os.path.join("/home/fair/fair/celeba_data/img_align_celeba/img_align_celeba/", name + ".jpg")
              if os.path.exists(path):
                img_encoded = torch.load(path)
              else:
                img_encoded = image_to_s(self.e4e,img_path)
                torch.save(img_encoded, path)
              
              img_encodeds.append(img_encoded)
            
          img_encoded = torch.cat(img_encodeds)
          img_arr = []
          
          manipulated = img_encoded.detach().clone()
          img_arr.append(gen_im(manipulated))

          for channel in self.channels:
                manipulated = img_encoded.detach().clone()
                manipulated[0,channel[0],channel[1]] +=10
                img_arr.append(gen_im(manipulated))
                
                manipulated = img_encoded.detach().clone()
                manipulated[0,channel[0],channel[1]] -=10
                img_arr.append(gen_im(manipulated))
         
          img = torch.cat(img_arr)
          
          """
          invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    
          
          debiased = [invTrans(image) for image in img_arr]
          debiased = torch.cat(debiased,3)[0]
          debiased = Image.fromarray((debiased.permute(1,2,0)*255).to(torch.uint8).cpu().numpy(), 'RGB')
          debiased.save("./ensemble.png")
          """
          results = torch.nn.functional.softmax(self.model(img), dim=-1)#[:,0]
          results = results.view(-1,batch,2)
          
          
          return torch.mean(results,axis=0)