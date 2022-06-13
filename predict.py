# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

#

# NVIDIA CORPORATION and its licensors retain all intellectual property

# and proprietary rights in and to this software, related documentation

# and any modifications thereto.  Any use, reproduction, disclosure or

# distribution of this software and related documentation without an express

# license agreement from NVIDIA CORPORATION is strictly prohibited.



"""Generate images using pretrained network pickle."""
from ensemble_model import EnsembleModel


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

import debias_matrix

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

class FairDataset(Dataset):
    def __init__(self, df, num_of_im, start_im, folder):

        self.num_of_im = num_of_im
        self.folder = folder
        self.df = df
        self.length = num_of_im
        self.start_im = start_im
        """
        if train_type == "e4e":
          self.length = num_of_im
        elif train_type == "real":
          self.length = num_of_im
        else:
          self.length = num_of_im * 3
        """
        
        self.e4e = load_net() 
          
        self.custom_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.folder, self.df.iloc[idx + self.start_im].name)
        young_label = (float(self.df.iloc[idx + self.start_im]["Young"]) + 1)/2
        glasses_label = (float(self.df.iloc[idx + self.start_im]["Eyeglasses"]) + 1)/2

        path = os.path.join("/home/fair/fair/celeba_data/celeba_s/", self.df.iloc[idx + self.start_im].name.split(".")[0] + ".pt")
        if os.path.exists(path):
          img_encoded = torch.load(path)
        else:
          img_encoded = image_to_s(self.e4e,img_path)
          torch.save(img_encoded, path) 
        img_encoded = gen_im(img_encoded)[0]

        img = Image.open(img_path).convert("RGB")
        img_real = self.custom_transform(img)
          
        return img_encoded, img_real, young_label, glasses_label, self.df.iloc[idx + self.start_im].name 


def get_accuracy(y_test, y_pred):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc 

def save_sample_image(img, epoch, label, train_type):

    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                         std = [ 1., 1., 1. ]),
                                   ])
    
    img = invTrans(img)
        
    after2 = Image.fromarray((img.permute(1,2,0)*255).to(torch.uint8).cpu().numpy(), 'RGB')
    ImageDraw.Draw(after2).text((0, 0),str(label),(255, 0, 0))

    after2.save("./classifier_results/"+train_type +"_"+str(epoch)+".png")

def manual_loss(y_pred, y_test, weight_pos) :
    return torch.sum(-(weight_pos*y_test*torch.log(y_pred) + (1-y_test)*torch.log(1-y_pred)))

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--batch_size', help='Batch size', type=int, required=True)
@click.option('--num_of_im', type=int)
#@click.option('--unsup_direction', help='Unsupervised direction path', type=str, required=True)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    batch_size: int,
    num_of_im: int

):
    """
    python predict.py --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl --batch_size=4 --num_of_im=100000 

    
    """

 
    #optimizer = torch.optim.Adam([styles_wanted_direction], lr=0.01)
    
    celeba_data = pd.read_csv("/home/fair/fair/celeba_data/list_attr_celeba.txt",sep='\s+')
    #print(celeba_data)
    #celeba_data = celeba_data.replace(-1, 0)
    
    print(celeba_data.columns)
    
    weight_young = celeba_data[0:num_of_im]["Young"].value_counts()[1] / celeba_data[0:num_of_im]["Young"].value_counts()[-1]
    
    
    celeba_data = celeba_data[num_of_im:]
    celeba_data_young = celeba_data[celeba_data["Eyeglasses"] == 1][0:(num_of_im//200)]
    celeba_data_old = celeba_data[celeba_data["Eyeglasses"] != 1][0:(num_of_im//200)]

    celeba_data_balanced = pd.concat([celeba_data_young,celeba_data_old])
    celeba_data_balanced = celeba_data_balanced.sample(len(celeba_data_balanced),random_state=61)
    
    
   
    val_dataset = FairDataset(celeba_data_balanced, num_of_im//100, 0, "/home/fair/fair/celeba_data/img_align_celeba/img_align_celeba/") # changed
    
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    models = ["paint_mobilenet-v2-age-celeba-best-1-debiased_ensemble_balanced.pt"]#["paint_mobilenet-v2-age-celeba-best-1-debiased_real_balanced.pt","paint_mobilenet-v2-age-celeba-best-1-debiased_fair_balanced2.pt"]#,"paint_mobilenet-v2-age-celeba-best-1-debiased_e4e.pt",
    model_names = [model.split("_")[-2] for model in models]
    
    #model = load_classifier('mobilenet-v2-age-celeba-best-1.pt', load_state = False)#copy.deepcopy(model)
    
    val_df = celeba_data_balanced[0:num_of_im//100]
    print(len(val_df))
    
    test_glasses_young = len(val_df[(val_df["Young"] == 1) & (val_df["Eyeglasses"] == 1)])
    test_glasses_old = len(val_df[(val_df["Young"] == -1) & (val_df["Eyeglasses"] == 1)])
    test_no_glasses_young = len(val_df[(val_df["Young"] == 1) & (val_df["Eyeglasses"] == -1)])
    test_no_glasses_old = len(val_df[(val_df["Young"] == -1) & (val_df["Eyeglasses"] == -1)])
     
    debias_matrix.main(test_glasses_young, test_glasses_old, test_no_glasses_young, test_no_glasses_old, "test_set")
    
    data_dict = dict()

    for model_path, model_name in zip(models, model_names):
        
        model = EnsembleModel("paint_mobilenet-v2-age-celeba-best-1-debiased_real_balanced.pt", [[3,120], [3,228]])#load_classifier(model_path, load_state = True)
        
        real_acc = 0
        encoded_acc = 0
         
        glasses_positive_predicted_young = 0
        glasses_positive_predicted_old = 0
        glasses_negative_predicted_young = 0
        glasses_negative_predicted_old = 0


        for step, (img_encoded, img_real, young_label, glasses_label, names) in enumerate(val_dataloader):  
         
            img_encoded, img_real, young_label, glasses_label = img_encoded.cuda(), img_real.cuda(), young_label.cuda(), glasses_label.cuda()
            
            with torch.no_grad():
            
              results_encoded = torch.nn.functional.softmax(model(img_encoded), dim=-1)[:,0]
              results_real = torch.nn.functional.softmax(model(img_real), dim=-1)[:,0]
              
              
              for result_encoded, result_real, young, glasses, name in zip(results_encoded, results_real, young_label, glasses_label, names):
              
                  if name in data_dict.keys():
                       dd = data_dict[name]
                  else:
                      dd = dict()
                  
                  dd["Eyeglasses True Label"] = int(glasses.item())
                  dd["Young True Label"] = int(young.item())
                  
                  dd[f"Young Prediction Score Encoded {model_name}"] = result_encoded.item()
                  dd[f"Young Prediction Encoded {model_name}"] = 1 if result_encoded.item() > 0.5 else 0
                  
                  dd[f"Young Prediction Score Real {model_name}"] = result_real.item()
                  dd[f"Young Prediction Real {model_name}"] = 1 if result_real.item() > 0.5 else 0
                  
                  
                  if model_name == "real":
                      if dd["Eyeglasses True Label"] == 1:
                      
                          if dd[f"Young Prediction Real {model_name}"] == 1:
                              glasses_positive_predicted_young += 1
                          else:
                              glasses_positive_predicted_old += 1
                      else:
                          if dd[f"Young Prediction Real {model_name}"] == 1:
                              glasses_negative_predicted_young += 1
                          else:
                              glasses_negative_predicted_old += 1
                      
                  else:
                  
                      if dd["Eyeglasses True Label"] == 1:
                      
                          if dd[f"Young Prediction Encoded {model_name}"] == 1:
                              glasses_positive_predicted_young += 1
                          else:
                              glasses_positive_predicted_old += 1
                      else:
                          if dd[f"Young Prediction Encoded {model_name}"] == 1:
                              glasses_negative_predicted_young += 1
                          else:
                              glasses_negative_predicted_old += 1
                  
                  data_dict[name] = dd

              
              
              encoded_acc += get_accuracy(young_label, results_encoded)
              real_acc += get_accuracy(young_label, results_real)
              
    
              
        print(f"For model {model_name}")
        print("Real Accuracy:", real_acc/len(val_dataloader))
        print("Encoded Accuracy:", encoded_acc/len(val_dataloader))
        debias_matrix.main(glasses_positive_predicted_young, glasses_positive_predicted_old, glasses_negative_predicted_young, glasses_negative_predicted_old, model_name)
          
    
    predict_df = pd.DataFrame.from_dict(data_dict, orient='index')
    predict_df.to_csv("./prediction_result.csv")
    
#----------------------------------------------------------------------------



if __name__ == "__main__":

    generate_images() # pylint: disable=no-value-for-parameter



#----------------------------------------------------------------------------

