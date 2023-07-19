import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
import math
import torchvision
import lpips
from models.stylegan2.model import Generator

class PivotTuning:
    def __init__(self, args):
        self.num_steps = args.num_steps
        self.lr = args.gt_lr
        self.G_ = Generator(1024, 512, 8)
        self.G_.load_state_dict(torch.load('pretrained_models/stylegan2-ffhq-config-f.pt')["g_ema"], strict=False)
        self.G_ = self.G_.cuda()
        self.l2_criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.G_.parameters(), lr=self.lr)
        self.percept = lpips.LPIPS(net="vgg").cuda()

    def make_noises(self):
        noises_single = self.G_.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(1, 1, 1, 1).normal_())
        for noise in noises:
            noise.requires_grad = False
        return noises

    def train(self, w, real_image):
        noises = self.make_noises()
        for i in range(self.num_steps):
            img_gen, _ = self.G_([w], input_is_latent=True, noise=noises)
            img_gen = torch.nn.functional.interpolate(img_gen, size=256)
            real_image = torch.nn.functional.interpolate(real_image, size=256)
            lossl2 = self.l2_criterion(img_gen, real_image)
            p_loss = self.percept(img_gen, real_image).sum()
            loss = p_loss + 0.1 * lossl2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self.G_, noises


class GeneratorTuning:
    def __init__(self, args):
        self.latents = torch.load(args.latent_path).unsqueeze(1)
        self.g_ema = Generator(1024, 512, 8)
        self.g_ema.load_state_dict(torch.load('pretrained_models/stylegan2-ffhq-config-f.pt')["g_ema"], strict=False)
        self.g_ema = self.g_ema.eval().cuda()
        self.dataloader = args.dataloader
        self.checkpoint_dir = args.checkpoint_dir
        self.inverted_images = args.inverted_image_path
        self.pt = PivotTuning(args)
        self.noise_path = args.noise_path
        self.noise_save = []

    def run(self):
        for ff, (fname, image) in enumerate(tqdm(self.dataloader)):
            image_name = fname[0]
            image = image.cuda()
            w = self.latents[ff].cuda()
            G_, noises = self.pt.train(w, image)
            G_.eval()
            torch.save(G_, f'{self.checkpoint_dir}/{image_name}.pt')
            img_gen, _ = G_([w], input_is_latent=True, noise=noises)
            noise_ = []
            for noise in noises:
                noise_.append(noise.cpu())
            self.noise_save.append(noises)
            torchvision.utils.save_image(img_gen, f"{self.inverted_images}/{image_name}.jpg", normalize=True, range=(-1, 1))
        torch.save(self.noise_save, self.noise_path)

