import os
import re
from typing import List, Optional
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import click
import dnnlib
import numpy as np
#import PIL.Image
from PIL import Image, ImageOps
import torch
from torch import linalg as LA
import legacy
import torch.nn.functional as F
#import cv2
import matplotlib.pyplot as plt
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
#from encoders.psp_encoders import GradualStyleEncoder
import random
import math
import time
import dlib
from utils.alignment import align_face
from numpy import dot
from numpy.linalg import norm
from sklearn.cluster import AgglomerativeClustering
import sys
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import pickle
from generate_fromS4 import generate_images
from torch.autograd.functional import jacobian
from sklearn.metrics.pairwise import pairwise_distances
import tqdm


stylegan2_model_path =  "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
noise_mode = 'const'
truncation_psi= 0.7
image_name = "a"
dim=1024
network_pkl = stylegan2_model_path
#from segmentation.face_parsing.test import evaluate
def load_generator():
 
  
  print('Loading networks from "%s"...' % network_pkl)
  device = torch.device('cuda')
  with dnnlib.util.open_url(network_pkl) as f:
      G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
      
  for i in G.parameters():
      i.requires_grad = False  
  
  temp_shapes = []
      
  for res in G.synthesis.block_resolutions:
      block = getattr(G.synthesis, f'b{res}') 
      if res == 4:
          temp_shape = (block.conv1.affine.weight.shape[0], block.conv1.affine.weight.shape[0], block.torgb.affine.weight.shape[0])
          block.conv1.affine = torch.nn.Identity()
          block.torgb.affine = torch.nn.Identity()
      
      else:
          temp_shape = (block.conv0.affine.weight.shape[0], block.conv1.affine.weight.shape[0], block.torgb.affine.weight.shape[0])
          block.conv0.affine = torch.nn.Identity()
          block.conv1.affine = torch.nn.Identity()
          block.torgb.affine = torch.nn.Identity()
        
      temp_shapes.append(temp_shape)
  return G, temp_shapes

G, temp_shapes = load_generator()
 
def run_alignment(image_path):
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor) 
  return aligned_image 
  
def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
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
            #print([None, self.in_channels, self.resolution // 2, self.resolution // 2])
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

def gen_im(styles):
    styles_idx = 0
    x2 = img2 = None
    #with torch.no_grad():
    for k , res in enumerate(G.synthesis.block_resolutions):
        block = getattr(G.synthesis, f'b{res}')

        if res == 4:
          x2, img2 = block_forward(block, x2, img2, styles[:, styles_idx:styles_idx+2, :], temp_shapes[k], noise_mode=noise_mode)
          styles_idx += 2
        else:
          x2, img2 = block_forward(block, x2, img2, styles[:, styles_idx:styles_idx+3, :], temp_shapes[k], noise_mode=noise_mode)
          styles_idx += 3

    #img2 =  F.interpolate(img2,size=256)
    device = torch.device('cuda')
    resizer = Resize((dim,dim))
    img2  = resizer(img2)
    
    mean = torch.as_tensor((0.485, 0.456, 0.406), dtype=torch.float, device=device)
    std = torch.as_tensor((0.229, 0.224, 0.225), dtype=torch.float, device=device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
        
    custom_transform = Compose([
                        Resize((256, 256)),
                        CenterCrop((224, 224))])    

    temp_img = (img2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
    #temp_img = custom_transform(temp_img.permute(0, 3, 1, 2))/255
    temp_img = (custom_transform(temp_img.permute(0, 3, 1, 2))/255).sub_(mean).div_(std)
                
    #img2 = img2.clamp(-1,1)
    #print(img2.shape)   
    #image1 = (img2 * 127.5 + 128).clamp(0, 255)
    

    return temp_img
   

image_array = []  

def generate_random(seeds):

    device = torch.device('cuda')

    network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        
    for i in G.parameters():
        i.requires_grad = False  

    truncation_psi = 0.7

    # Labels.
    class_idx = 1
    label = torch.zeros([1, G.c_dim], device=device).requires_grad_()
    if G.c_dim != 0:
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    
    styles_array = []
    
    for seed_idx, seed in enumerate(seeds): # 1 milyon 1-1000000
        if seed==seeds[-1]:
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        ws = G.mapping(z, label, truncation_psi=truncation_psi)
        torch.save(ws, './tensor.pt')
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
            ws = ws.to(torch.float32)
            #ws = ws.detach().cpu()      # ----------- modified ------------------
            #styles_array.append(ws)     # ----------- modified ------------------

        
            w_idx = 0
            for res in G.synthesis.block_resolutions:
                block = getattr(G.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        
        styles = torch.zeros(1,26,512, device=device) # ------------------- 26 to 18
        styles_idx = 0
        temp_shapes = []
        for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
            block = getattr(G.synthesis, f'b{res}')

            if res == 4:
              temp_shape = (block.conv1.affine.weight.shape[0], block.conv1.affine.weight.shape[0], block.torgb.affine.weight.shape[0])
              styles[0,:1,:] = block.conv1.affine(cur_ws[0,:1,:])
              styles[0,1:2,:] = block.torgb.affine(cur_ws[0,1:2,:])
              if seed_idx==(len(seeds)-1):
                block.conv1.affine = torch.nn.Identity()
                block.torgb.affine = torch.nn.Identity()
              styles_idx += 2
            else:
              temp_shape = (block.conv0.affine.weight.shape[0], block.conv1.affine.weight.shape[0], block.torgb.affine.weight.shape[0])
              styles[0,styles_idx:styles_idx+1,:temp_shape[0]] = block.conv0.affine(cur_ws[0,:1,:])
              styles[0,styles_idx+1:styles_idx+2,:temp_shape[1]] = block.conv1.affine(cur_ws[0,1:2,:])
              styles[0,styles_idx+2:styles_idx+3,:temp_shape[2]] = block.torgb.affine(cur_ws[0,2:3,:])
              if seed_idx==(len(seeds)-1):
                block.conv0.affine = torch.nn.Identity()
                block.conv1.affine = torch.nn.Identity()
                block.torgb.affine = torch.nn.Identity()
              styles_idx += 3
            temp_shapes.append(temp_shape)
            #print(temp_shape)


        styles = styles
        styles_array.append(styles)
        
    return torch.cat(styles_array)
    
def gen_im_2(styles_maskin, styles_maskout, mask):
    styles_idx = 0
    x2 = img2 = None
    x2_maskin = img2_maskin = None
    x2_maskout = img2_maskout = None
    

    #with torch.no_grad():
    for k , res in enumerate(G.synthesis.block_resolutions):
        block = getattr(G.synthesis, f'b{res}')
        
        if res == 4:
          x2_maskin, img2_maskin = block_forward(block, x2, img2_maskin, styles_maskin[:, styles_idx:styles_idx+2, :], temp_shapes[k], noise_mode=noise_mode)
          x2_maskout, img2_maskout = block_forward(block, x2, img2_maskout, styles_maskout[:, styles_idx:styles_idx+2, :], temp_shapes[k], noise_mode=noise_mode)
          #print(x2_maskin.shape[-2:])
          #print(mask.unsqueeze(0).shape)
          
          
          resizer_m = Resize(x2_maskin.shape[-2:])
          mask_resized = resizer_m(mask.unsqueeze(0))
          #mask_resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(x2_maskin.shape[-1],x2_maskin.shape[-1]),mode = 'nearest')[-2:]
          
          #print(mask_resized.shape)
          #print("1111: ", x2_maskout.shape, x2_maskin.shape)
          x2 = x2_maskout * (1-mask_resized) + x2_maskin * mask_resized
          #print(x2.shape)
          img2 = img2_maskout * (1-mask_resized) + img2_maskin * mask_resized
          styles_idx += 2
        else:
          x2_maskin, img2_maskin = block_forward(block, x2, img2_maskin, styles_maskin[:, styles_idx:styles_idx+3, :], temp_shapes[k], noise_mode=noise_mode)
          x2_maskout, img2_maskout = block_forward(block, x2, img2_maskout, styles_maskout[:, styles_idx:styles_idx+3, :], temp_shapes[k], noise_mode=noise_mode)
          #print(x2_maskin.shape)
          #print(x2_maskin.shape[-2:])
          #print(mask.shape)
          resizer_m = Resize(x2_maskin.shape[-2:])
          mask_resized = resizer_m(mask.unsqueeze(0))
          #mask_resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(x2_maskin.shape[-1],x2_maskin.shape[-1]),mode = 'nearest')[-2:]# resizer_m(mask)
          #print(mask_resized.shape)
          #print("1112: ", x2_maskout.shape, x2_maskin.shape)
          
          x2 = x2_maskout * (1-mask_resized) + x2_maskin * mask_resized
          #print(x2.shape)
          img2 = img2_maskout * (1-mask_resized) + img2_maskin * mask_resized
          styles_idx += 3

    #img2 =  F.interpolate(img2,size=256)
    resizer = Resize((dim,dim))
    img2  = resizer(img2)
    img2_maskin  = resizer(img2_maskin)
    img2_maskout  = resizer(img2_maskout)
    
    img2 = img2.clamp(-1,1)
    img2_maskin = img2_maskin.clamp(-1,1)
    img2_maskout = img2_maskout.clamp(-1,1)
       
    image1 = (img2[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255)
    #concat_created = Image.fromarray(image1.to(torch.uint8).cpu().numpy(), 'RGB')
    #concat_created.save("./masks/split_out/combined.jpg")
    
    image2 = (img2_maskin[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255)
    #concat_created = Image.fromarray(image2.to(torch.uint8).cpu().numpy(), 'RGB')
    #concat_created.save("./masks/split_out/mask_in.jpg")
    #image_array.append(concat_created)
    
    image3 = (img2_maskout[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255)
    #concat_created = Image.fromarray(image3.to(torch.uint8).cpu().numpy(), 'RGB')
    #concat_created.save("./masks/split_out/mask_out.jpg")
    return image1

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]
    










