
import cv2
import torch
from torchvision import transforms
from models import irse, ir152, facenet
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

def preprocess(im, mean, std, device):
    if len(im.size()) == 3:
        im = im.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    elif len(im.size()) == 4:
        im = im.transpose(1, 3).transpose(2, 3)

    mean = torch.tensor(mean).to(device)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std).to(device)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    im = (im - mean) / std
    return im

def read_img(data_dir, mean, std, device):
    img = cv2.imread(data_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    img = torch.from_numpy(img).to(torch.float32).to(device)
    img = preprocess(img, mean, std, device)
    return img

from facenet_pytorch import MTCNN
mtcnn = MTCNN(image_size=1024, margin=0, post_process=False, select_largest=False, device='cuda')
def alignment(image):
    boxes, probs = mtcnn.detect(image)
    return boxes[0]

def trans():
    return transforms.Compose([transforms.ToTensor()])


class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.model.eval();
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda").view(1,3,1,1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda").view(1,3,1,1)

    def forward(self, image, text):
        #image = image.add(1).div(2)
        image = image.sub(self.mean).div(self.std)
        image = self.face_pool(image)
        similarity = 1 - self.model(image, text)[0]/ 100
        return similarity

def cos_simi(emb_1, emb_2):
    return torch.mean(torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1))

def cal_adv_loss(source_resize, target_resize):
    cos_loss = (1-cos_simi(source_resize, target_resize))
    return cos_loss

def black_box(img_gen,target_eval,model):
    
    if model == 'mobile_face':


        fr_model_m = irse.MobileFaceNet(512)
        fr_model_m.load_state_dict(torch.load('./models/mobile_face.pth'))
        fr_model_m.cuda()
        fr_model_m.eval()
        with torch.no_grad():
            s_e3 = fr_model_m((F.interpolate((img_gen-0.5)*2, size=(112,112), mode='bilinear')))
            te3 = fr_model_m((F.interpolate(target_eval, size=(112,112), mode='bilinear')))
        cos_simi_m = torch.cosine_similarity(s_e3, te3)
        
        return [cos_simi_m.item()]
    
    elif model == 'ir152':
        
        fr_model_152 = ir152.IR_152((112, 112))
        fr_model_152.load_state_dict(torch.load('./models/ir152.pth'))
        fr_model_152.cuda()
        fr_model_152.eval()
        with torch.no_grad():
            s_e3 = fr_model_152((F.interpolate((img_gen-0.5)*2, size=(112,112), mode='bilinear')))
            te3 = fr_model_152((F.interpolate(target_eval, size=(112,112), mode='bilinear')))
        cos_simi_152 = torch.cosine_similarity(s_e3, te3)
        
        return [cos_simi_152.item()]
    
    elif model == 'facenet':

        fr_model_facenet = facenet.InceptionResnetV1(num_classes=8631, device='cuda')
        fr_model_facenet.load_state_dict(torch.load('./models/facenet.pth'))
        fr_model_facenet.cuda()
        fr_model_facenet.eval()
        with torch.no_grad():
            s_e3 = fr_model_facenet((F.interpolate((img_gen-0.5)*2, size=(160,160), mode='bilinear')))
            te3 = fr_model_facenet((F.interpolate(target_eval, size=(160,160), mode='bilinear')))
        cos_simi_facenet = torch.cosine_similarity(s_e3, te3)
        
        return [cos_simi_facenet.item()]
    
    else:

        fr_model_50 = irse.Backbone(50, 0.6, 'ir_se')
        fr_model_50.load_state_dict(torch.load('./models/irse50.pth'))
        fr_model_50.cuda()
        fr_model_50.eval()
        with torch.no_grad():
            s_e3 = fr_model_50((F.interpolate((img_gen-0.5)*2, size=(112,112), mode='bilinear')))
            te3 = fr_model_50((F.interpolate(target_eval, size=(112,112), mode='bilinear')))
        cos_simi_50 = torch.cosine_similarity(s_e3, te3)
        
        return [cos_simi_50.item()]

arr0 = 0
def quan(arr,model):
    global arr0,arr1,arr2
    if model == 'mobile_face':
        if arr[0] > 0.301611:
            arr0 += 1
        return [arr0]
    elif model == 'facenet':
        if arr[0] > 0.409131:
            arr0 += 1
        return [arr0]
    elif model == 'irse50':
        if arr[0] > 0.241045: 
            arr0 += 1
        return [arr0]
    else: 
        if arr[0] > 0.166788:
            arr0 += 1
        return [arr0]

# utils.py
def get_target(target_choice,margin):
    device='cuda'
    if target_choice == '1':
        target = read_img('005869.jpg', 0.5, 0.5, device)[:,:,168-margin:912+margin,205-margin:765+margin]
        target_eval = read_img('008793.jpg', 0.5, 0.5, device)[:,:,145-margin:920+margin,202-margin:775+margin]
    elif target_choice == '2':
        target = read_img('085807.jpg', 0.5, 0.5, device)[:,:,187-margin:891+margin,244-margin:764+margin]
        target_eval = read_img('047073.jpg', 0.5, 0.5, device)[:,:,234-margin:905+margin,266-margin:791+margin]
    elif target_choice == '3':
        target = read_img('116481.jpg', 0.5, 0.5, device)[:,:,214-margin:955+margin,188-margin:773+margin]
        target_eval = read_img('055622.jpg', 0.5, 0.5, device)[:,:,185-margin:931+margin,198-margin:780+margin]
    elif target_choice == '4':
        target = read_img('169284.jpg', 0.5, 0.5, device)[:,:,173-margin:925+margin,233-margin:792+margin]
        target_eval = read_img('166607.jpg', 0.5, 0.5, device)[:,:,172-margin:917+margin,219-margin:779+margin]
    else:
        raise ValueError("Invalid target choice. Choose between '1', '2', '3', or '4'.")
    return target, target_eval

    
