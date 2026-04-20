import torch
import torch.nn.functional as F
import clip

from configs.base_config import CLIP_MODEL_NAME, DEVICE


CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPFeatureExtractor:
    def __init__(self, model_name=CLIP_MODEL_NAME, device=DEVICE):
        self.device = device
        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    def preprocess(self, images):
        # Project the training/test tensor range from [-1, 1] back to CLIP's
        # expected [0, 1] image space before applying CLIP normalization.
        images = ((images.to(self.device, dtype=torch.float32) + 1.0) / 2.0).clamp(0.0, 1.0)
        images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
        return (images - self.mean) / self.std

    def encode(self, images):
        features = self.model.encode_image(self.preprocess(images)).float()
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def __call__(self, images):
        return self.encode(images)
