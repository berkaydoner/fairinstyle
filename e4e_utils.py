from argparse import Namespace
import os
import sys
from PIL import Image
import torch
import torchvision.transforms as transforms
from encoder4editing.models.psp import pSp
import dnnlib
import legacy
import dlib
from encoder4editing.utils.alignment import align_face

EXPERIMENT_ARGS = {
                        "model_path": "/home/fair/fair/encoder4editing/pretrained_models/e4e_ffhq_encode.pt",
                        "transform": transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 
                    }

stylegan2_model_path ="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
noise_mode = 'const'
truncation_psi= 0.7
image_name = "a"
dim=1024
network_pkl = stylegan2_model_path

def load_net():
  
  
  if os.path.getsize(EXPERIMENT_ARGS['model_path']) < 1000000:
    raise ValueError("Pretrained model was unable to be downlaoded correctly!")
  
  model_path = EXPERIMENT_ARGS['model_path']
  ckpt = torch.load(model_path, map_location='cpu')
  opts = ckpt['opts']
  
  # update the training options
  opts['checkpoint_path'] = model_path
  if 'learn_in_w' not in opts:
      opts['learn_in_w'] = False
  if 'output_size' not in opts:
      opts['output_size'] = 1024
  
  
  opts = Namespace(**opts)
  net = pSp(opts)
  net.eval()
  net.cuda()
  print('Model successfully loaded!')
  return net


def get_encoding(inputs, net):
  _, result = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
  return result


def image_to_w(net, img_path):
    
    try:
      input_image = run_alignment(img_path)
    except:
      input_image = Image.open(img_path)
      input_image = input_image.convert("RGB")

    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)

    with torch.no_grad():
        w = get_encoding(transformed_image.unsqueeze(0), net)
    
    
    return w
    

def run_alignment(image_path):

  predictor = dlib.shape_predictor("./encoder4editing/pretrained_models/shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor) 
  
  return aligned_image    
    
def image_to_s(net, img_path):

     
    ws = image_to_w(net, img_path)
    

    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore



    # Labels.
    label = torch.zeros([1, G.c_dim], device=device).requires_grad_()


    # Generate images.
    for i in G.parameters():
      i.requires_grad = False


    ws = ws.to(device)#torch.tensor(ws, device=device)
    #ws = torch.load("example_celebs.pt")[5].unsqueeze(0).to(device)
    
    #ws = np.load("projected_w(1).npz")['w']
    #ws = torch.tensor(ws, device=device)

    block_ws = []
    with torch.autograd.profiler.record_function('split_ws'):
        ws = ws.to(torch.float32)

        w_idx = 0
        for res in G.synthesis.block_resolutions:
            block = getattr(G.synthesis, f'b{res}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv


    styles = torch.zeros(1,26,512, device=device)
    styles_idx = 0
    temp_shapes = []
    for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
        block = getattr(G.synthesis, f'b{res}')

        if res == 4:
          temp_shape = (block.conv1.affine.weight.shape[0], block.conv1.affine.weight.shape[0], block.torgb.affine.weight.shape[0])
          styles[0,:1,:] = block.conv1.affine(cur_ws[0,:1,:])
          styles[0,1:2,:] = block.torgb.affine(cur_ws[0,1:2,:])

          block.conv1.affine = torch.nn.Identity()
          block.torgb.affine = torch.nn.Identity()
          styles_idx += 2
        else:
          temp_shape = (block.conv0.affine.weight.shape[0], block.conv1.affine.weight.shape[0], block.torgb.affine.weight.shape[0])
          styles[0,styles_idx:styles_idx+1,:temp_shape[0]] = block.conv0.affine(cur_ws[0,:1,:])
          styles[0,styles_idx+1:styles_idx+2,:temp_shape[1]] = block.conv1.affine(cur_ws[0,1:2,:])
          styles[0,styles_idx+2:styles_idx+3,:temp_shape[2]] = block.torgb.affine(cur_ws[0,2:3,:])

          block.conv0.affine = torch.nn.Identity()
          block.conv1.affine = torch.nn.Identity()
          block.torgb.affine = torch.nn.Identity()
          styles_idx += 3
        temp_shapes.append(temp_shape)


    styles = styles
    return styles