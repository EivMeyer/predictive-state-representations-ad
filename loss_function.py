# loss_function.py

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms

def gaussian(window_size, sigma):
    x = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Ensure the images are in the format [B, C, H, W]
    img1 = img1.permute(0, 3, 1, 2)
    img2 = img2.permute(0, 3, 1, 2)

    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    
    # Adjust window size if it's larger than the input dimensions
    window_size = min(window_size, height, width)
    
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        self.channel = 3  # Assume 3 channels for RGB images
        self.window = None

    def forward(self, img1, img2):
        # Convert to float if necessary
        img1 = img1.float() if img1.dtype != torch.float32 else img1
        img2 = img2.float() if img2.dtype != torch.float32 else img2

        # Normalize to [0, 1] range if coming from uint8
        if img1.max() > 1:
            img1 = img1 / 255.0
        if img2.max() > 1:
            img2 = img2 / 255.0
            
        (_, height, width, channel) = img1.size()

        if self.window is None:
            self.window_size = min(self.window_size, height, width)
            self.window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)

        return ssim(img1, img2, window=self.window, window_size=self.window_size, size_average=self.size_average)

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg.children())[:16]).to(device)
        for param in self.layers.parameters():
            param.requires_grad = False
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = device

    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        x = self.transform(x.permute(0, 3, 1, 2))
        y = self.transform(y.permute(0, 3, 1, 2))
        x_features = self.layers(x)
        y_features = self.layers(y)
        return nn.functional.mse_loss(x_features, y_features)

class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.2, beta=0.3, gamma=0.5, delta=0.1, device="cpu"):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta  # Weight for diversity loss
        self.mse_loss = nn.MSELoss() if self.alpha > 0.0 else None
        self.ssim_module = SSIM() if self.beta > 0.0 else None
        self.perceptual_loss = PerceptualLoss(device) if self.gamma > 0.0 else None

    def diversity_loss(self, pred):
        # Reshape predictions to (batch_size, -1)
        pred_flat = pred.view(pred.size(0), -1)
        
        # Calculate pairwise distances
        dist_matrix = torch.cdist(pred_flat, pred_flat)
        
        # Calculate mean distance (excluding self-distances)
        mean_dist = dist_matrix.sum() / (pred.size(0) * (pred.size(0) - 1))
        
        # Return negative of mean distance (we want to maximize diversity)
        return -mean_dist

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target) if self.alpha > 0.0 else 0.0
        ssim_loss = 1 - self.ssim_module(pred, target) if self.beta > 0.0 else 0.0
        perceptual = self.perceptual_loss(pred, target) if self.gamma > 0.0 else 0.0
        diversity = self.diversity_loss(pred) if self.delta > 0.0 else 0.0
        
        total_loss = (self.alpha * mse + 
                      self.beta * ssim_loss + 
                      self.gamma * perceptual + 
                      self.delta * diversity)
        
        return total_loss, {
            'mse': mse.item() if isinstance(mse, torch.Tensor) else mse,
            'ssim': ssim_loss.item() if isinstance(ssim_loss, torch.Tensor) else ssim_loss,
            'perceptual': perceptual.item() if isinstance(perceptual, torch.Tensor) else perceptual,
            'diversity': diversity.item() if isinstance(diversity, torch.Tensor) else diversity
        }