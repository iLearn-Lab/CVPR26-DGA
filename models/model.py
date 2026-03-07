import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from .clip_text import CLIP_Text
from .vit import ViT


class CLIP_attn(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = ViT(clip_model.visual)

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features

    def encode_image(self, image):
        return self.image_encoder(image)

