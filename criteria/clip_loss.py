
import torch
import clip


class CLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)
        self.mse_loss = torch.nn.MSELoss()
        
    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity

    def norm_loss(self, image_pre, image_post):
        norm_pre = self.model.encode_image(self.avg_pool(self.upsample(image_pre))).norm(dim=-1)
        norm_post = self.model.encode_image(self.avg_pool(self.upsample(image_post))).norm(dim=-1)

        return self.mse_loss(norm_pre, norm_post)

class CLIPText(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPText, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=1024 // 32) #i made changes here

    def forward(self, text):
        # image = self.avg_pool(self.upsample(image))
        # image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        return text_features
