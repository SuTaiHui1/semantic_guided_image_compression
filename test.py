import glob
import logging
import os
import warnings

import torch
from tqdm import tqdm

from configs.base_config import (
    DEVICE,
    KODAK_TEST_PATH,
    LOG_PATH,
    MODEL_ARCH_VERSION,
    MODEL_SAVE_PATH,
    RECON_IMAGE_PATH,
    SAVE_RECON_IMAGES,
)
from data.dataloader import build_dataloader
from experiments import (
    DEFAULT_BASELINE_EXPERIMENT,
    DEFAULT_ENHANCED_EXPERIMENT,
    get_experiment,
)
from models.enhanced_vit import EnhancedViTCompressor
from utils.clip_utils import CLIPFeatureExtractor
from utils.metrics import calculate_fid, evaluate_metrics


warnings.filterwarnings("ignore", category=UserWarning)


def init_logger():
    os.makedirs(LOG_PATH, exist_ok=True)
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(LOG_PATH, "test.log"), mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


logger = init_logger()


def checkpoint_path(checkpoint_stem):
    return os.path.join(MODEL_SAVE_PATH, f"{checkpoint_stem}_model_best.pth")


RECON_DIR_NAMES = {
    "baseline": "baseline_recon",
    "full_optimized": "enhanced_recon",
    "clip": "semantic_probe_clip",
    "zeros": "semantic_probe_zeros",
    "random": "semantic_probe_random",
    "dataset_shuffle": "semantic_probe_shuffle",
}


def recon_output_path(experiment_name, semantic_override=None):
    if semantic_override is None:
        suffix = RECON_DIR_NAMES.get(experiment_name, experiment_name)
    else:
        suffix = RECON_DIR_NAMES.get(semantic_override, f"{experiment_name}_{semantic_override}")
    return os.path.join(RECON_IMAGE_PATH, suffix)


def semantic_features_for_mode(clip_features, mode):
    if clip_features is None:
        return None
    if mode == "clip":
        return clip_features
    if mode == "zeros":
        return torch.zeros_like(clip_features)
    if mode == "random":
        random_features = torch.randn_like(clip_features)
        return torch.nn.functional.normalize(random_features, dim=-1)
    raise ValueError(f"unsupported semantic mode: {mode}")


def clear_recon_dir(recon_path):
    if not os.path.isdir(recon_path):
        return
    for file_name in os.listdir(recon_path):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(recon_path, file_name))


def validate_image_paths(original_path, recon_path):
    original_imgs = sorted(
        glob.glob(os.path.join(original_path, "*.png")) + glob.glob(os.path.join(original_path, "*.jpg"))
    )
    recon_imgs = sorted(
        glob.glob(os.path.join(recon_path, "*.png")) + glob.glob(os.path.join(recon_path, "*.jpg"))
    )

    if not original_imgs:
        raise FileNotFoundError(f"original test image path is empty: {original_path}")
    if not recon_imgs:
        raise FileNotFoundError(f"reconstructed image path is empty: {recon_path}")
    if len(original_imgs) != len(recon_imgs):
        logger.warning(f"image count mismatch: original={len(original_imgs)}, recon={len(recon_imgs)}")

    logger.info(f"validated image paths: original={len(original_imgs)} | recon={len(recon_imgs)}")
    return original_imgs, recon_imgs


def build_model(experiment):
    return EnhancedViTCompressor(
        use_semantic=experiment["use_semantic_modules"],
        semantic_input_mode=experiment["semantic_input_mode"],
    )


def load_model(model, experiment_name):
    experiment = get_experiment(experiment_name)
    model_path = checkpoint_path(experiment["checkpoint_stem"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model checkpoint does not exist: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if checkpoint.get("arch_version") != MODEL_ARCH_VERSION:
        raise RuntimeError(
            f"checkpoint architecture mismatch: found {checkpoint.get('arch_version')}, "
            f"expected {MODEL_ARCH_VERSION}. Please retrain with the current code."
        )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE)
    logger.info(f"loaded model weights: {model_path}")
    return model


def test_model(experiment_name, semantic_override=None):
    experiment = get_experiment(experiment_name)
    test_loader = build_dataloader(split="test")
    logger.info(f"test set loaded: {len(test_loader.dataset)} images")
    recon_path = recon_output_path(experiment_name, semantic_override)

    model = build_model(experiment)
    model = load_model(model, experiment_name)
    model.eval()

    clip_extractor = (
        CLIPFeatureExtractor()
        if experiment["use_semantic_modules"] or experiment["use_clip_semantic_loss"]
        else None
    )

    original_imgs = []
    recon_imgs = []
    img_names = []

    if SAVE_RECON_IMAGES:
        os.makedirs(recon_path, exist_ok=True)
        clear_recon_dir(recon_path)

    cached_batches = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="testing")
        for img, names in pbar:
            img = img.to(DEVICE, dtype=torch.float32)
            clip_features = clip_extractor(img) if clip_extractor is not None else None
            mode = semantic_override if semantic_override is not None else experiment["semantic_input_mode"]

            if mode == "dataset_shuffle":
                cached_batches.append((img, names, clip_features))
                continue

            model_clip_features = semantic_features_for_mode(clip_features, mode)
            recon_img, _ = model(img, model_clip_features)

            img = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
            recon_img = ((recon_img + 1.0) / 2.0).clamp(0.0, 1.0)

            for idx, img_name in enumerate(names):
                original_imgs.append(img[idx].cpu())
                recon_imgs.append(recon_img[idx].cpu())
                img_names.append(img_name)

    if semantic_override == "dataset_shuffle":
        feature_bank = torch.cat([batch_clip.cpu() for _, _, batch_clip in cached_batches], dim=0)
        feature_bank = feature_bank.roll(shifts=1, dims=0)
        offset = 0
        for img, names, _ in cached_batches:
            batch_size = img.shape[0]
            model_clip_features = feature_bank[offset : offset + batch_size].to(DEVICE)
            offset += batch_size
            with torch.no_grad():
                recon_img, _ = model(img, model_clip_features)
            img = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
            recon_img = ((recon_img + 1.0) / 2.0).clamp(0.0, 1.0)
            for idx, img_name in enumerate(names):
                original_imgs.append(img[idx].cpu())
                recon_imgs.append(recon_img[idx].cpu())
                img_names.append(img_name)

    basic_metrics = evaluate_metrics(
        original_imgs,
        recon_imgs,
        img_names,
        save_recon=SAVE_RECON_IMAGES,
        recon_path=recon_path,
    )

    fid_val = 0.0
    if SAVE_RECON_IMAGES and os.path.exists(recon_path):
        try:
            validate_image_paths(KODAK_TEST_PATH, recon_path)
            fid_val = calculate_fid(KODAK_TEST_PATH, recon_path)
            logger.info(f"FID computed: {fid_val:.2f}")
        except Exception as exc:
            logger.error(f"FID computation failed: {exc}")
            logger.error(f"paths: original={KODAK_TEST_PATH} | recon={recon_path}")
    else:
        logger.warning("reconstructed images not saved, skip FID")

    result_name = experiment["display_name"] if semantic_override is None else f"{experiment['display_name']} [{semantic_override}]"
    logger.info(f"\n========== {result_name} results ==========")
    logger.info(f"PSNR: {basic_metrics['avg_psnr']:.2f} dB")
    logger.info(f"SSIM: {basic_metrics['avg_ssim']:.4f}")
    logger.info(f"LPIPS: {basic_metrics['avg_lpips']:.4f}")
    logger.info(f"FID: {fid_val:.2f}")

    return {
        "experiment": experiment_name,
        "display_name": experiment["display_name"],
        "semantic_override": semantic_override,
        "psnr": basic_metrics["avg_psnr"],
        "ssim": basic_metrics["avg_ssim"],
        "lpips": basic_metrics["avg_lpips"],
        "fid": fid_val,
    }


if __name__ == "__main__":
    try:
        if SAVE_RECON_IMAGES:
            os.makedirs(RECON_IMAGE_PATH, exist_ok=True)

        logger.info("========== test default baseline ==========")
        test_model(DEFAULT_BASELINE_EXPERIMENT)

        logger.info("========== test default enhanced ==========")
        test_model(DEFAULT_ENHANCED_EXPERIMENT)

        logger.info("========== all tests completed ==========")
    except Exception as exc:
        logger.error(f"testing failed: {exc}", exc_info=True)
        raise
