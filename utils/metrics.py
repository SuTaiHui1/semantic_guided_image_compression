import os

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_fid.inception import InceptionV3
from scipy.linalg import sqrtm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net="alex").to(DEVICE).eval()
for param in lpips_model.parameters():
    param.requires_grad_(False)


def evaluate_metrics(original_imgs, recon_imgs, img_names, save_recon=False, recon_path=None):
    if len(original_imgs) != len(recon_imgs):
        raise ValueError("original_imgs and recon_imgs must have the same length")

    if save_recon:
        if not recon_path:
            raise ValueError("recon_path is required when save_recon=True")
        os.makedirs(recon_path, exist_ok=True)

    psnr_list = []
    ssim_list = []
    lpips_list = []

    for i in tqdm(range(len(original_imgs)), desc="计算指标中"):
        orig_tensor = original_imgs[i].detach().cpu().clamp(0, 1)
        recon_tensor = recon_imgs[i].detach().cpu().clamp(0, 1)

        orig = (orig_tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        recon = (recon_tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

        psnr_list.append(compute_psnr(orig, recon, data_range=255))
        ssim_list.append(compute_ssim(orig, recon, data_range=255, channel_axis=2))

        with torch.no_grad():
            lpips_val = lpips_model(
                orig_tensor.unsqueeze(0).to(DEVICE) * 2.0 - 1.0,
                recon_tensor.unsqueeze(0).to(DEVICE) * 2.0 - 1.0,
            ).item()
        lpips_list.append(lpips_val)

        if save_recon:
            save_img = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(recon_path, img_names[i]), save_img)

    return {
        "avg_psnr": float(np.mean(psnr_list)),
        "avg_ssim": float(np.mean(ssim_list)),
        "avg_lpips": float(np.mean(lpips_list)),
    }


def calculate_fid(real_dir, fake_dir, batch_size=16, dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(DEVICE).eval()

    def get_features(path):
        files = [
            os.path.join(path, filename)
            for filename in sorted(os.listdir(path))
            if filename.lower().endswith(("png", "jpg", "jpeg"))
        ]
        if not files:
            raise FileNotFoundError(f"no images found in {path}")

        features = []
        with torch.no_grad():
            for start in tqdm(range(0, len(files), batch_size), desc="提取FID特征"):
                batch_files = files[start:start + batch_size]
                batch = []
                for file_path in batch_files:
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError(f"failed to read image: {file_path}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    batch.append(img)

                pred = model(torch.stack(batch).to(DEVICE))[0]
                if pred.shape[-2:] != (1, 1):
                    pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
                features.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())

        features = np.concatenate(features, axis=0)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    mu1, sigma1 = get_features(real_dir)
    mu2, sigma2 = get_features(fake_dir)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)
