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
    def __init__(self, df, train_type, manipulation_channel, alpha, num_of_im, start_im, folder):
        self.train_type = train_type
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
         
        self.manipulation_channel = manipulation_channel
        self.alpha = alpha
          
        if train_type == "e4e" or train_type == "fair":
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
        label = (float(self.df.iloc[idx + self.start_im]["Young"]) + 1)/2
        
        if self.train_type == "e4e":
          path = os.path.join("/home/fair/fair/celeba_data/celeba_s/", self.df.iloc[idx + self.start_im].name.split(".")[0] + ".pt")
          if os.path.exists(path):
            img_encoded = torch.load(path)
          else:
            img_encoded = image_to_s(self.e4e,img_path)
            torch.save(img_encoded, path) 
          img = gen_im(img_encoded)[0]
        elif self.train_type == "real":
          img = Image.open(img_path).convert("RGB")
          img = self.custom_transform(img)
        else:
          path = os.path.join("/home/fair/fair/celeba_data/celeba_s/", self.df.iloc[idx + self.start_im].name.split(".")[0] + ".pt")
          if os.path.exists(path):
            img_encoded = torch.load(path)
          else: 
            img_encoded = image_to_s(self.e4e,img_path)
            torch.save(img_encoded, path)
            
          img_arr = []
          img_arr.append(gen_im(img_encoded))
          has_eyeglasses = self.df.iloc[idx + self.start_im]["Eyeglasses"]
          if has_eyeglasses==1:
            img_encoded_positive = img_encoded.clone()
            img_encoded_positive[:,int(self.manipulation_channel[0]),int(self.manipulation_channel[1])] += 30#self.alpha
            img_arr.append(gen_im(img_encoded_positive))
          else:
           img_encoded_negative = img_encoded.clone()
           img_encoded_negative[:,int(self.manipulation_channel[0]),int(self.manipulation_channel[1])] -= 30#self.alpha
           img_arr.append(gen_im(img_encoded_negative))     
#          img_encoded_positive = img_encoded.clone()
#          img_encoded_positive[:,int(self.manipulation_channel[0]),int(self.manipulation_channel[1])] += self.alpha
#          img_arr.append(gen_im(img_encoded_positive))
#          img_encoded_negative = img_encoded.clone()
#          img_encoded_negative[:,int(self.manipulation_channel[0]),int(self.manipulation_channel[1])] -= self.alpha
#          img_arr.append(gen_im(img_encoded_negative))
          img = torch.cat(img_arr)
          debiased = torch.cat(img_arr,3)[0]
          invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                         std = [ 1., 1., 1. ]), 
                                   ])  
    
          debiased = invTrans(debiased)
          debiased = Image.fromarray((debiased.permute(1,2,0)*255).to(torch.uint8).cpu().numpy(), 'RGB')
          if has_eyeglasses==1:
            #if not os.path.exists("./debiased_examples/glasses_"+str(self.df.iloc[idx + self.start_im].name)):
            debiased.save("./debiased_examples/glasses_"+str(self.df.iloc[idx + self.start_im].name))
          else:
            #if not os.path.exists("./debiased_examples/2noglasses_"+str(self.df.iloc[idx + self.start_im].name)):
            debiased.save("./debiased_examples/noglasses_"+str(self.df.iloc[idx + self.start_im].name))
          label = torch.Tensor([label,label])
          
          
        return img,label, self.df.iloc[idx + self.start_im].name 


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
@click.option('--train_type', type=click.Choice(['e4e', 'real', 'fair']),  required=True)
@click.option('--manipulation_layer', type=int, help='List of random seeds') 
@click.option('--manipulation_channel', type=int, help='List of random seeds')
@click.option('--alpha', type=float)
@click.option('--num_of_im', type=int)
@click.option('--epochs', type=int)
@click.option('--continue_train', type=int)
#@click.option('--unsup_direction', help='Unsupervised direction path', type=str, required=True)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    batch_size: int,
    num_of_im: int,
    epochs: int,
    train_type: str,
    alpha: float,
    manipulation_layer: int,
    manipulation_channel: int,
    continue_train: int
):
    """
    python debias.py --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl --batch_size=4 --train_type=fair --manipulation_layer=3 --manipulation_channel=228 --alpha=20 --num_of_im=100 --epochs=5 --continue_train=1

    
    """
    
    
    if continue_train == 0:
        with open(f'./out_training/balanced2_{train_type}.txt','w') as f:
            f.write("alpha: "+str(alpha)+"\n")
            f.write("batch_size: "+str(batch_size)+"\n")
            f.write("num_of_im: "+str(num_of_im)+"\n")
            f.write("manipulation_layer: "+str(manipulation_layer)+"\n")
            f.write("manipulation_channel: "+str(manipulation_channel)+"\n")
            f.write("Started training"+"\n")
            
    else:
        with open(f'./out_training/balanced2_{train_type}.txt','a') as f:
            f.write("alpha: "+str(alpha)+"\n")
            f.write("batch_size: "+str(batch_size)+"\n")
            f.write("num_of_im: "+str(num_of_im)+"\n")
            f.write("manipulation_layer: "+str(manipulation_layer)+"\n")
            f.write("manipulation_channel: "+str(manipulation_channel)+"\n")
            f.write("Started training"+"\n")
    
    manipulation_channel = [manipulation_layer, manipulation_channel]

 
    #optimizer = torch.optim.Adam([styles_wanted_direction], lr=0.01)
    
    celeba_data = pd.read_csv("/home/fair/fair/celeba_data/list_attr_celeba.txt",sep='\s+')
    #print(celeba_data)
    #celeba_data = celeba_data.replace(-1, 0)
    
    celeba_data_young = celeba_data[celeba_data["Young"] == 1][0:(num_of_im//2 + num_of_im//10)]
    celeba_data_old = celeba_data[celeba_data["Young"] != 1][0:(num_of_im//2 + num_of_im//10)]

    celeba_data_balanced = pd.concat([celeba_data_young,celeba_data_old])
    celeba_data_balanced = celeba_data_balanced.sample(len(celeba_data_balanced))
    
    weight_young = celeba_data[0:num_of_im]["Young"].value_counts()[1] / celeba_data[0:num_of_im]["Young"].value_counts()[-1]


    
    dataset = FairDataset(celeba_data_balanced, train_type, manipulation_channel, alpha, num_of_im, 0, "/home/fair/fair/celeba_data/img_align_celeba/img_align_celeba/")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    
    val_dataset = FairDataset(celeba_data_balanced, train_type, manipulation_channel, alpha, num_of_im//5, num_of_im, "/home/fair/fair/celeba_data/img_align_celeba/img_align_celeba/")
    
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
     
    #model = load_classifier('paint_mobilenet-v2-age-celeba-best-1-debiased2.pt')
    
    if continue_train == 0:
        model = load_classifier('mobilenet-v2-age-celeba-best-1.pt', load_state = False)#copy.deepcopy(model)
    else:
        model = load_classifier(f"paint_mobilenet-v2-age-celeba-best-1-debiased_{train_type}_balanced2.pt", load_state = True)
    
    
    BCE = nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    
     
    val_loss = 0
    
    for epoch in range(epochs):
        print("############","Epoch ",epoch)
        
        model.train()
        running_loss = 0
        for step, (img, label, name) in enumerate(tqdm(dataloader)): 
            img, label = img.cuda(), label.cuda().to(torch.float32)
            
            
            if train_type == "fair":
                label = label.view(-1)
                img = img.view(-1,3,224,224)
                

            results = torch.nn.functional.softmax(model(img), dim=-1)[:,0]
            
            loss = BCE(results, label)#manual_loss(results, label, weight_young) #BCE(results, label)

            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            running_loss += loss.item()
  
            if step%100==0:
                torch.save(model.state_dict(), f"paint_mobilenet-v2-age-celeba-best-1-debiased_{train_type}_balanced2.pt") 

        print("Train loss:", running_loss/len(dataloader))
            
        acc = 0
        val_running_loss = 0
        #model.eval()
        for step, (img, label, name) in enumerate(val_dataloader):  
         
            img, label = img.cuda(), label.cuda().to(torch.float32)
            
            with torch.no_grad():
            
              if train_type == "fair":
                label = label.view(-1)
                img = img.view(-1,3,224,224)

              results = torch.nn.functional.softmax(model(img), dim=-1)[:,0]

              
              loss = BCE(results, label)#loss = manual_loss(results, label, weight_young)# BCE(results, label)
          
              val_running_loss += loss.item()
              
              acc += get_accuracy(label, results)
              
        
              
        print("Val loss:", val_running_loss/len(val_dataloader))
        print("Val accuracy:", acc/len(val_dataloader))
        
        with open(f"./out_training/balanced2_{train_type}.txt", "a") as myfile:
              myfile.write("Epoch: "+ str(epoch)+"\n")
              myfile.write("Train loss: "+str( running_loss/len(dataloader))+"\n")
              myfile.write("Val loss: "+ str(val_running_loss/len(val_dataloader))+"\n")
              myfile.write("Val accuracy:"+ str( acc.item()/len(val_dataloader))+"\n")
              myfile.write("\n")
        
        save_sample_image(img[0], epoch, label[0], train_type)
        print(name[0])
         
        
                

    torch.save(model.state_dict(), f"paint_mobilenet-v2-age-celeba-best-1-debiased_{train_type}_balanced2.pt")        
      


#----------------------------------------------------------------------------



if __name__ == "__main__":

    generate_images() # pylint: disable=no-value-for-parameter



#----------------------------------------------------------------------------

