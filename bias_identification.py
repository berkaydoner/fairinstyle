import torch
from generate_utils import generate_random, gen_im
from paint_debias import load_classifier
from PIL import Image, ImageDraw
import numpy as np
import torchvision.transforms as transforms



def get_positive_examples(classifier,num_examples):
  positive_style_vectors = []
  classifier.eval()
  generate_index = 0
  while(len(positive_style_vectors)<num_examples):
    with torch.no_grad():
      styles = generate_random(range(generate_index,generate_index+10))
      images = gen_im(styles)
      outputs = classifier(images)
      for i in range(len(outputs)):
        if outputs[i][1]<outputs[i][0]:
          positive_style_vectors.append(styles[i])
          if len(positive_style_vectors)==num_examples:
            break
      generate_index += 10
  return positive_style_vectors
  
def get_top_channels(positive_style_vectors):
  normalized_positives = torch.zeros((len(positive_style_vectors),26,512))
  means = torch.load("/home/fair/fair/data/population_mean.pt")
  stds = torch.load("/home/fair/fair/data/population_std.pt")
  for i in range(len(positive_style_vectors)):
    normalized_positives[i] = (positive_style_vectors[i] - means) / stds
  
  normalized_positives = normalized_positives.permute(1,2,0)
  
  diff_means = torch.mean(normalized_positives,dim=2)
  diff_stds  = torch.std(normalized_positives,dim=2)
  
  ratios = torch.abs(diff_means / diff_stds)
  ratios = torch.nan_to_num(ratios)
  
  top_channels = torch.topk(ratios.flatten(),20).indices
  return top_channels

def transform_img(img):
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                         std = [ 1., 1., 1. ]),
                                   ])
    
    img = invTrans(img)
    img = img.permute(1,2,0)*255
    
    return img


def main():
    #classifier = load_classifier("/home/fair/fair/paint_mobilenet-v2-age-celeba-best-1-debiased_real_balanced.pt")
    classifier = load_classifier("/home/fair/fair/models/mobilenet-v2-age-celeba-best-1.pt")
    num_samples = 500
    positive_samples = get_positive_examples(classifier,num_samples)
    #top_channels = get_top_channels(positive_samples)
    #top_channels = [torch.Tensor([x[0]*512+x[1]],dtype=torch.int8) for x in [(6,134),(9,186),(9,306),(11,232),(11,14),(11,286),(12,455),(3,120),(9,164),(8,281),(5,423),(3,508),(8,28),(2,105),(9,6),(8,38),(8,93),(6,273),(6,214),(8,63)]]
    top_channels = [torch.Tensor([x[0]*512+x[1]]) for x in [(9,435),(3,120),(9,381),(11,286),(9,6)]]
    top_channels = [channel.to(torch.int32) for channel in top_channels]
    print(top_channels)
    for channel in top_channels:
        print(channel.item()//512,channel.item()%512)

    white_pic = np.full((224,224,3), 255).astype('uint8') 
    image_grid = [[]]
    alphas = [10,15,20] #,15,20]
    
    # first row
    white_whole =  Image.fromarray((white_pic), 'RGB')    
    image_grid[0].append(white_whole)
    for alpha in alphas[::-1]:
        print('alpha ', alpha)
        alpha_text =  Image.fromarray((white_pic), 'RGB')
        ImageDraw.Draw(alpha_text).text((112, 200),"alpha : " + str(-1*alpha),(0, 0, 0))
        image_grid[0].append(alpha_text)
    
    alpha_text =  Image.fromarray((white_pic), 'RGB')
    ImageDraw.Draw(alpha_text).text((112, 200),"alpha : 0",(0, 0, 0))
    image_grid[0].append(alpha_text)
    
    for alpha in alphas:
        print('alpha ', alpha)
        alpha_text =  Image.fromarray((white_pic), 'RGB')
        ImageDraw.Draw(alpha_text).text((112, 200),"alpha : " + str(alpha),(0, 0, 0))
        image_grid[0].append(alpha_text)
        
    print('0: ', len(image_grid[0]))
    
    topk = 10
    num_ch = 1
    sample_indices = []
    seed_starts = {}
    seed_starts[(3,120)]=69
    seed_starts[(9,381)]=1
    seed_starts[(9,435)]=0
    seed_starts[(6,134)]=172
    seed_starts[(11,232)]=229
    seed_starts[(9,306)]=59
    seed_starts[(9,6)]=77
    seed_starts[(11,14)]=469
    seed_starts[(9,164)]=49
    seed_starts[(11,286)]= 27
    seed_starts[(8,281)]= 99
    

    for i,biases in enumerate(top_channels[:topk]): 
        print(biases)
        for sample_ind, sample in enumerate(positive_samples):
            if sample_ind in sample_indices:
                continue
            #print('heyyyy')
            t2 = transform_img(gen_im(sample[None,:,:])[0])
            original = Image.fromarray(t2.to(torch.uint8).cpu().numpy(), 'RGB')
            original.save("./biases/images/"+str(sample_ind)+".png")
                
            modified = sample.clone()
            modified[biases.item()//512,biases.item()%512] -= alphas[-1]
            modified = modified[None,:,:]
            output_after = torch.nn.functional.softmax(classifier(gen_im(modified)),dim=-1)[0,0]
            
            modified2 = sample.clone()
            modified2[biases.item()//512,biases.item()%512] += alphas[-1]
            modified2 = modified2[None,:,:]
            output_after2 = torch.nn.functional.softmax(classifier(gen_im(modified2)),dim=-1)[0,0]
            
            original_score = torch.nn.functional.softmax(classifier(gen_im(sample[None,:,:])),dim=-1)[0,0]
            
            threshold = 0.55
            if abs(output_after-original_score)>threshold or abs(output_after2-original_score)>threshold or sample_ind==seed_starts.get((biases.item()//512,biases.item()%512),-1):
                print("channel: ",biases.item()//512,biases.item()%512, sample_ind)
                if (biases.item()//512,biases.item()%512) in seed_starts:
                  print(seed_starts[(biases.item()//512,biases.item()%512)])
                  if sample_ind!=seed_starts[(biases.item()//512,biases.item()%512)]:
                    continue
                sample_indices.append(sample_ind)
                image_grid.append([])
                
                alphas_pic = {}
                for alpha in alphas:
                    alpha_neg = sample.clone()
                    alpha_neg[biases.item()//512,biases.item()%512] -= alpha
                    alpha_neg = alpha_neg[None,:,:]
                    output_neg = torch.nn.functional.softmax(classifier(gen_im(alpha_neg)),dim=-1)[0,0]
                    
                    tneg = transform_img(gen_im(alpha_neg)[0])
                    after_neg = Image.fromarray(tneg.to(torch.uint8).cpu().numpy(), 'RGB')
                    ImageDraw.Draw(after_neg).text((0, 0),"classifier score: " + str(round(output_neg.item(),2)),(255, 255, 255))
                    alphas_pic['-'+str(alpha)] = after_neg
                    
                    alpha_pos = sample.clone()
                    alpha_pos[biases.item()//512,biases.item()%512] += alpha
                    alpha_pos = alpha_pos[None,:,:]
                    output_pos = torch.nn.functional.softmax(classifier(gen_im(alpha_pos)),dim=-1)[0,0]
                    
                    tpos = transform_img(gen_im(alpha_pos)[0])
                    after_pos = Image.fromarray(tpos.to(torch.uint8).cpu().numpy(), 'RGB')
                    ImageDraw.Draw(after_pos).text((0, 0),"classifier score: " + str(round(output_pos.item(),2)),(255, 255, 255))
                    alphas_pic[str(alpha)] = after_pos
                    
                    
                channel =  Image.fromarray((white_pic), 'RGB')
                ImageDraw.Draw(channel).text((112, 112),str(biases.item()//512)+"_"+str(biases.item()%512),(0, 0, 0))
                
                #t1 = transform_img(gen_im(modified)[0])
                #after = Image.fromarray(t1.to(torch.uint8).cpu().numpy(), 'RGB')
                #draw = ImageDraw.Draw(after)
                #draw.text((0, 0),"classifier score: " + str(round(output_after.item(),2)),(255, 0, 0))
                #draw.text((224, 0),"alpha : -" + str(alpha),(255, 0, 0))
                
                t2 = transform_img(gen_im(sample[None,:,:])[0])
                original = Image.fromarray(t2.to(torch.uint8).cpu().numpy(), 'RGB')
                original.save("./biases/images/"+str(sample_ind)+".png")
                ImageDraw.Draw(original).text((0, 0),"classifier score: " + str(round(original_score.item(),2)),(255, 255, 255))
                #ImageDraw.Draw(before).text((224, 0),"alpha : " + str(0),(255, 0, 0))
                
                #t3 = transform_img(gen_im(modified2)[0])
                #after2 = Image.fromarray(t3.to(torch.uint8).cpu().numpy(), 'RGB')
                #ImageDraw.Draw(after2).text((0, 0),"classifier score: " + str(round(output_after2.item(),2)),(255, 0, 0))
                #ImageDraw.Draw(after2).text((224, 0),"alpha : " + str(alpha),(255, 0, 0))
                
                image_grid[num_ch].append(channel)
                
                for alpha in alphas[::-1]:
                    print('-'+str(alpha))
                    image_grid[num_ch].append(alphas_pic['-'+str(alpha)])

                image_grid[num_ch].append(original)
                
                for alpha in alphas:
                    print(str(alpha))
                    image_grid[num_ch].append(alphas_pic[str(alpha)])
                  
                print(num_ch,': ', len(image_grid[num_ch]))
                
                #image_grid[num_ch].append(after2)
                #bias = np.concatenate(image_array, axis = 1)
                #bias = Image.fromarray(bias,"RGB")
                #bias.save("./biases/"+str(biases.item()//512)+"_"+str(biases.item()%512)+".png")
                num_ch += 1
                break
        
    print('len ', len(image_grid))
    rows = []
    for i,r in enumerate(image_grid):
        #for g in r:
            #print('row ',i, ' ', g)
        res = np.concatenate(r,axis = 1) 
        rows.append(res)

    res = np.concatenate(rows, axis = 0)
    img_final = Image.fromarray(res, 'RGB')
    img_final.save("./biases/classifier_top"+str(topk)+"channels_"+str(num_samples)+"samples.png") 


if __name__ == "__main__":
  main()