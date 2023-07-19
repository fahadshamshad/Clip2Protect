import clip
import torch
import torchvision.transforms as transforms

from criteria.text_templates import imagenet_templates
from criteria.infonce import InfoNCE


class NCELoss(torch.nn.Module):
    def __init__(self, device, clip_model='ViT-B/32'):
        super(NCELoss, self).__init__()

        self.device = 'cuda'
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor


        self.InfoNCE = InfoNCE()

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)
    
    def distance_with_templates(self, img: torch.Tensor, class_str: str, templates=imagenet_templates) -> torch.Tensor:

        text_features  = self.get_text_features(class_str, templates)
        image_features = self.get_image_features(img)

        similarity = image_features @ text_features.T

        return 1. - similarity
    
    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:
        template_text = self.compose_text_with_templates(class_str, templates)
        tokens = clip.tokenize(template_text).to(self.device)
        #print(tokens.shape)

        text_features = self.encode_text(tokens).detach()
        #print(text_features.shape)

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]

    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def infonce_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        
        source_text = self.get_text_features(source_class)

        target_text = self.get_text_features(target_class).mean(axis=0, keepdim=True)

        src_img    = self.get_image_features(src_img)
        target_img = self.get_image_features(target_img)
        target_text = target_text.repeat(target_img.shape[0],1)
        query = (target_img - src_img)
        query /= query.clone().norm(dim=-1, keepdim=True)

        pos1_direction = self.compute_text_direction(source_class, target_class)
        pos1_direction = pos1_direction.repeat(target_img.shape[0],1)

        pos2_direction = (target_text - src_img)
        pos2_direction /= pos2_direction.clone().norm(dim=-1, keepdim=True)
       
        neg_direction = (source_text - src_img[:1])
        neg_direction /= neg_direction.clone().norm(dim=-1, keepdim=True)
        # target img - source img -> query
        # target text - source text-> pos1
        # target text - source img -> pos2
        # souce text - source img -> neg
        return self.InfoNCE(query.repeat(2,1), torch.cat([pos1_direction,pos2_direction], 0), neg_direction)


    def forward(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str):
        clip_loss = self.infonce_loss(src_img, source_class, target_img, target_class)

        return clip_loss
