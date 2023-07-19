import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
from models.stylegan2.model import Generator
import os
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import cv2
import torch
import clip
from PIL import Image
from models import irse, ir152, facenet
import torch.nn.functional as F
import numpy as np
import glob
import random
import torchvision
from utils.clip2protect_utils import *

#clip for nce loss
import criteria.clip_loss as clip_loss
import criteria.nce_loss as nce_loss
from torchvision import transforms



class FaceRecognitionModels:
    def __init__(self, device='cuda'):
        self.device = device
        self.fr_model_m = self.load_mobile_face_net()
        self.fr_model_facenet = self.load_facenet()
        self.fr_model_152 = self.load_ir152()
        self.fr_model_50 = self.load_irse50()

    def load_mobile_face_net(self):
        fr_model_m = irse.MobileFaceNet(512)
        fr_model_m.load_state_dict(torch.load('./models/mobile_face.pth'))
        fr_model_m.to(self.device)
        fr_model_m.eval()
        return fr_model_m

    def load_facenet(self):
        fr_model_facenet = facenet.InceptionResnetV1(num_classes=8631, device=self.device)
        fr_model_facenet.load_state_dict(torch.load('./models/facenet.pth'))
        fr_model_facenet.to(self.device)
        fr_model_facenet.eval()
        return fr_model_facenet

    def load_ir152(self):
        fr_model_152 = ir152.IR_152((112, 112))
        fr_model_152.load_state_dict(torch.load('./models/ir152.pth'))
        fr_model_152.to(self.device)
        fr_model_152.eval()
        return fr_model_152

    def load_irse50(self):
        fr_model_50 = irse.Backbone(50, 0.6, 'ir_se')
        fr_model_50.load_state_dict(torch.load('./models/irse50.pth'))
        fr_model_50.to(self.device)
        fr_model_50.eval()
        return fr_model_50
 
class Adversarial_Opt:
    def __init__(self,args):
        self.augment = transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5)
        self.num_aug = args.num_aug
        self.nce_loss = nce_loss.NCELoss('cuda', clip_model="ViT-B/32")
        self.source_text = args.source_text
        self.description = args.makeup_prompt
        self.face_models = FaceRecognitionModels()
        self.fr_model_m = self.face_models.load_mobile_face_net()
        self.fr_model_facenet = self.face_models.load_facenet()  
        self.fr_model_152 = self.face_models.load_ir152()
        self.fr_model_50 = self.face_models.load_irse50()
        
        self.steps = args.steps
        self.path = sorted(glob.glob(args.data_dir+'/*.jpg'))
        self.generators = sorted(glob.glob(args.checkpoint_dir+'/*.pt'))
        self.latents = torch.load(args.latent_path).unsqueeze(1)
        self.noi = torch.load(args.noise_path)
        self.target_choice = args.target_choice
        self.trans = trans()
        self.model = args.model
        self.impersonate = args.impersonate
        self.noise_optimize = args.noise_optimize
        self.margin = args.margin
        
        self.lat_hyp = args.lambda_lat 
        self.c_hyp = args.lambda_clip
        self.adv_hyp = args.lambda_adv
        
        self.protected_face_dir = args.protected_face_dir
        

    def get_target_embeddings(self):
        target, target_eval = get_target(self.target_choice,self.margin)
        with torch.no_grad():
            target_embbeding_m = self.fr_model_m((F.interpolate(target, size=(112,112), mode='bilinear')))
            target_embbeding_facenet = self.fr_model_facenet((F.interpolate(target, size=(160,160), mode='bilinear')))
            target_embbeding_152 = self.fr_model_152((F.interpolate(target, size=(112,112), mode='bilinear')))
            target_embbeding_50 = self.fr_model_50((F.interpolate(target, size=(112,112), mode='bilinear'))) 
        return target_embbeding_m, target_embbeding_facenet, target_embbeding_152, target_embbeding_50,target_eval
    

    def process_latent(self, latent, noi):
        latent = latent.cuda()
        latent_cl = latent.clone().detach()
        latent.requires_grad = True

        noisss = []
        noiss = noi
        for nois in noiss:
            if nois.shape[2] < 512:
                nois.requires_grad = True
            else:
                nois.requires_grad = False            
            noisss.append(nois)
        return latent, latent_cl, noisss
    

    def get_source_embedding(self, path):
        bb_src1 = alignment(Image.open(path))
        img_src1 = self.trans(Image.open(path)).unsqueeze(0)[:,:,round(bb_src1[1])-self.margin:round(bb_src1[3])+self.margin,round(bb_src1[0])-self.margin:round(bb_src1[2])+self.margin]
        norm_source_src1 = (F.interpolate((img_src1-0.5)*2, size=(112,112), mode='bilinear')).cuda()
        norm_source_facenet_src1 = (F.interpolate((img_src1-0.5)*2, size=(160,160), mode='bilinear')).cuda()

        with torch.no_grad():
            source_embbeding_m_ = self.fr_model_m(norm_source_src1)       
            source_embbeding_facenet_ = self.fr_model_facenet(norm_source_facenet_src1)
            source_embbeding_152_ = self.fr_model_152(norm_source_src1)
            source_embbeding_50_ = self.fr_model_50(norm_source_src1)
        return bb_src1,source_embbeding_m_.detach(), source_embbeding_facenet_.detach(), source_embbeding_152_.detach(), source_embbeding_50_.detach()
    
    
    def get_image_gen(self, latent, noisss, g_ema):
            
        with torch.no_grad():
            img_org, _ = g_ema([latent], input_is_latent=True, noise=noisss)

        img_org_ = img_org.detach().clone()
        img_org_ = ((img_org_+1)/2).clamp(0,1)
        img_org_ = img_org_.repeat(self.num_aug,1,1,1)

        return img_org_
    

    def get_adv_loss(self, img_gen, source_embbeding_m_, source_embbeding_facenet_, source_embbeding_152_, source_embbeding_50_,target_embbeding_m, target_embbeding_facenet, target_embbeding_152, target_embbeding_50):
        norm_source = (F.interpolate((img_gen-0.5)*2, size=(112,112), mode='bilinear'))
        norm_source_facenet = (F.interpolate((img_gen-0.5)*2, size=(160,160), mode='bilinear'))

        source_embbeding_m = self.fr_model_m(norm_source)       
        source_embbeding_facenet = self.fr_model_facenet(norm_source_facenet)
        source_embbeding_152 = self.fr_model_152(norm_source)
        source_embbeding_50 = self.fr_model_50(norm_source)

        adv_loss_m_sim = cal_adv_loss(source_embbeding_m, source_embbeding_m_)
        adv_loss_facenet_sim = cal_adv_loss(source_embbeding_facenet, source_embbeding_facenet_)
        adv_loss_152_sim = cal_adv_loss(source_embbeding_152, source_embbeding_152_)
        adv_loss_50_sim = cal_adv_loss(source_embbeding_50, source_embbeding_50_)
        
        adv_loss_m = cal_adv_loss(source_embbeding_m, target_embbeding_m.detach())
        adv_loss_facenet = cal_adv_loss(source_embbeding_facenet, target_embbeding_facenet.detach())
        adv_loss_152 = cal_adv_loss(source_embbeding_152, target_embbeding_152.detach())
        adv_loss_50 = cal_adv_loss(source_embbeding_50, target_embbeding_50.detach())
        
        return adv_loss_m_sim, adv_loss_facenet_sim, adv_loss_152_sim, adv_loss_50_sim,adv_loss_m,adv_loss_facenet,adv_loss_152,adv_loss_50
    
    
    def calculate_loss(self, model, l2_loss, c_loss, adv_loss_m_sim, adv_loss_facenet_sim, adv_loss_152_sim, adv_loss_50_sim, adv_loss_m, adv_loss_facenet, adv_loss_152, adv_loss_50):
        if model=='mobile_face':
            dis_loss =  adv_loss_facenet_sim+adv_loss_152_sim+adv_loss_50_sim
            sim_loss =  adv_loss_facenet+adv_loss_152+adv_loss_50

            #adv_loss = sim_loss-dis_loss
            adv_loss = sim_loss - dis_loss if not self.impersonate else sim_loss
            loss =  self.lat_hyp * l2_loss+self.adv_hyp*adv_loss+self.c_hyp*c_loss
            #print('adv_loss',adv_loss.item())

        elif model=='facenet':
            dis_loss =  adv_loss_m_sim+adv_loss_152_sim+adv_loss_50_sim
            sim_loss =  adv_loss_m+adv_loss_152+adv_loss_50
            adv_loss = sim_loss - dis_loss if not self.impersonate else sim_loss
            loss =  self.lat_hyp * l2_loss+self.adv_hyp*adv_loss+self.c_hyp*c_loss
            
        elif model=='irse50':
            dis_loss =  adv_loss_m_sim+adv_loss_facenet_sim+adv_loss_152_sim
            sim_loss =  adv_loss_m+adv_loss_152+adv_loss_facenet
            adv_loss = sim_loss - dis_loss if not self.impersonate else sim_loss
            loss =  self.lat_hyp * l2_loss+self.adv_hyp*adv_loss+self.c_hyp*c_loss
        else:
            dis_loss =  adv_loss_facenet_sim+adv_loss_50_sim+adv_loss_m_sim
            sim_loss =  adv_loss_facenet+adv_loss_50+adv_loss_m
            adv_loss = sim_loss - dis_loss if not self.impersonate else sim_loss
            loss =  self.lat_hyp * l2_loss+self.adv_hyp*adv_loss+self.c_hyp*c_loss

        return loss
    
    
    def run(self):
        
        target_embbeding_m, target_embbeding_facenet, target_embbeding_152, target_embbeding_50,target_eval = self.get_target_embeddings()
        
        for ff, (latent, path) in enumerate(zip(self.latents, self.path)):
            
            with torch.no_grad():
                g_ema = torch.load(self.generators[ff]).eval() #loading fine-tuned generator
   
            _,latent_cl, noisss = self.process_latent(latent, self.noi[ff]) #processing latent and noise
            img_org_ = self.get_image_gen(latent, noisss,g_ema) #augmenting image
            
            
            optimizer = torch.optim.Adam([latent] + (noisss if self.noise_optimize else []), lr=0.01)
            
            bb_src1,source_embbeding_m_, source_embbeding_facenet_, source_embbeding_152_, source_embbeding_50_ = self.get_source_embedding(path)

            for i in range(self.steps):
                
                optimizer.zero_grad()
                
                img_gen_, _ = g_ema([latent], input_is_latent=True, noise=noisss)
                img_gen_ = ((img_gen_+1)/2).clamp(0,1)

                img_gen_aug = torch.cat([self.augment(img_gen_) for i in range(self.num_aug)], dim=0)                     
                c_loss = self.nce_loss(img_org_, self.source_text,img_gen_aug, self.description).sum()

                l2_loss = ((latent_cl - latent) ** 2).sum()
 
                #cropping
                img_gen = img_gen_[:,:,round(bb_src1[1])-self.margin:round(bb_src1[3])+self.margin,round(bb_src1[0])-self.margin:round(bb_src1[2])+self.margin]

                adv_loss_m_sim, adv_loss_facenet_sim, adv_loss_152_sim, adv_loss_50_sim,adv_loss_m,adv_loss_facenet,adv_loss_152,adv_loss_50 = self.get_adv_loss(img_gen, source_embbeding_m_, source_embbeding_facenet_, source_embbeding_152_, source_embbeding_50_,target_embbeding_m, target_embbeding_facenet, target_embbeding_152, target_embbeding_50)
                                            
                
                loss = self.calculate_loss(self.model, l2_loss, c_loss, adv_loss_m_sim, adv_loss_facenet_sim, adv_loss_152_sim, adv_loss_50_sim, adv_loss_m, adv_loss_facenet, adv_loss_152, adv_loss_50)               
                loss.backward()
                
                latent.grad[0][0:8] = torch.zeros(8,512) 
                #latent.grad[0][14:18] = torch.zeros(4,512)

                optimizer.step()

                if (i+1) % self.steps == 0:
                    
                     with torch.no_grad():
                        success_count = quan(black_box(img_gen.detach(),target_eval,self.model),self.model)

                     torchvision.utils.save_image(img_gen_, f"{self.protected_face_dir}/{str(ff)+'_'+str(i).zfill(5)}.jpg", normalize=True, range=(0, 1))

        print(f"Total successes: {success_count[0]} out of {len(self.path)}")
