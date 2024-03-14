from util import get_engine
import torch.nn as nn

class SupConCLIP(nn.Module):
    """encoder w/ or w/o projection head"""
    def __init__(self, name='vitb32_openclip_laion400m'):
        super(SupConCLIP, self).__init__()
        model, preprocess, tokenizer = get_engine(model_cfg=name)
        self.encoder = model.encode_image
        self.visual = model.visual
        self.transformer = model.transformer
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        # add a mlp head
        """
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )
        """

    def forward(self, x):
        feat = self.encoder(x)
        # feat = self.head(feat) # add a mlp head
        feat = feat / feat.norm(dim=-1, keepdim=True) # Normalization
        return feat


class SupCECLIP(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='vitb32_openclip_laion400m', num_classes=200):
        super(SupCECLIP, self).__init__()
        model, preprocess, tokenizer = get_engine(model_cfg=name)
        self.encoder = model.encode_image
        self.fc = nn.Linear(512, num_classes)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def forward(self, x):

        return self.fc(self.encoder(x))