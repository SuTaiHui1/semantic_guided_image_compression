import logging
import os

import lpips
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from configs.base_config import (
    ALEX_LPIPS_GAP_MARGIN,
    ALEX_LPIPS_GAP_WEIGHT,
    ALEX_LPIPS_LOSS_WEIGHT,
    AUX_LEARNING_RATE,
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    LOG_PATH,
    MODEL_ARCH_VERSION,
    MODEL_SAVE_PATH,
    RATE_LOSS_WEIGHT,
    SEMANTIC_CONTRASTIVE_TEMPERATURE,
    SEMANTIC_CONTRASTIVE_WEIGHT,
    SEMANTIC_LOSS_WEIGHT,
    SEMANTIC_MISMATCH_MARGIN,
    SEMANTIC_MISMATCH_WEIGHT,
    SEMANTIC_LR_MULTIPLIER,
    WEIGHT_DECAY,
)
from data.dataloader import build_dataloader
from experiments import (
    DEFAULT_BASELINE_EXPERIMENT,
    DEFAULT_ENHANCED_EXPERIMENT,
    get_experiment,
)
from models.enhanced_vit import EnhancedViTCompressor
from utils.clip_utils import CLIPFeatureExtractor


TRAIN_ENHANCED_ONLY = False
GRAD_CLIP_NORM = 1.0


def setup_logger():
    os.makedirs(LOG_PATH, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(LOG_PATH, "train.log"), mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def checkpoint_path(checkpoint_stem, best=True):
    suffix = "best" if best else "final"
    return os.path.join(MODEL_SAVE_PATH, f"{checkpoint_stem}_model_{suffix}.pth")


def semantic_features_for_mode(clip_features, mode):
    if clip_features is None:
        return None
    if mode == "clip":
        return clip_features
    if mode == "zeros":
        return torch.zeros_like(clip_features)
    if mode == "shuffle":
        return clip_features.roll(shifts=1, dims=0)
    if mode == "random":
        random_features = torch.randn_like(clip_features)
        return F.normalize(random_features, dim=-1)
    raise ValueError(f"unsupported semantic_input_mode: {mode}")


def build_model(experiment):
    return EnhancedViTCompressor(
        use_semantic=experiment["use_semantic_modules"],
        semantic_input_mode=experiment["semantic_input_mode"],
    ).to(DEVICE)


def load_pretrained_weights(model, experiment_name, logger):
    experiment = get_experiment(experiment_name)

    if experiment["resume_from_own_checkpoint"]:
        own_path = checkpoint_path(experiment["checkpoint_stem"], best=True)
        if os.path.exists(own_path):
            checkpoint = torch.load(own_path, map_location=DEVICE)
            if checkpoint.get("arch_version") == MODEL_ARCH_VERSION:
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                logger.info(f"resumed {experiment_name} from existing checkpoint: {own_path}")
                return model

    init_from_experiment = experiment.get("init_from_experiment")
    if init_from_experiment is not None:
        init_experiment = get_experiment(init_from_experiment)
        init_path = checkpoint_path(init_experiment["checkpoint_stem"], best=True)
        if os.path.exists(init_path):
            checkpoint = torch.load(init_path, map_location=DEVICE)
            if checkpoint.get("arch_version") == MODEL_ARCH_VERSION:
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                logger.info(f"initialized {experiment_name} from {init_from_experiment}: {init_path}")
                return model

    if not experiment["warm_start_from_baseline"]:
        return model

    baseline_path = checkpoint_path("baseline", best=True)
    if not os.path.exists(baseline_path):
        logger.warning(f"baseline weights not found, {experiment_name} will train from scratch: {baseline_path}")
        return model

    checkpoint = torch.load(baseline_path, map_location=DEVICE)
    if checkpoint.get("arch_version") != MODEL_ARCH_VERSION:
        logger.warning("baseline checkpoint architecture does not match current code, skip warm start")
        return model

    if experiment["use_semantic_modules"]:
        model_dict = model.state_dict()
        baseline_dict = {
            key: value
            for key, value in checkpoint["model_state_dict"].items()
            if key in model_dict and "semantic_adapter" not in key and "semantic_refiner" not in key
        }
        model_dict.update(baseline_dict)
        model.load_state_dict(model_dict)
        logger.info(f"loaded {len(baseline_dict)}/{len(model_dict)} parameters from baseline checkpoint")
        return model

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    logger.info(f"loaded baseline weights into {experiment_name} from {baseline_path}")
    return model


def build_optimizer(model, experiment):
    if not experiment["use_semantic_modules"]:
        return optim.AdamW(
            model.main_parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999),
        )

    named_main_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and not name.endswith(".quantiles")
    ]
    semantic_params = [
        param
        for name, param in named_main_params
        if name.startswith("semantic_adapter") or name.startswith("semantic_refiner")
    ]
    codec_params = [
        param
        for name, param in named_main_params
        if not (name.startswith("semantic_adapter") or name.startswith("semantic_refiner"))
    ]

    return optim.AdamW(
        [
            {"params": codec_params, "lr": LEARNING_RATE},
            {"params": semantic_params, "lr": LEARNING_RATE * SEMANTIC_LR_MULTIPLIER},
        ],
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )


def load_baseline_teacher(experiment, logger):
    if not experiment["use_baseline_teacher"]:
        return None

    baseline_path = checkpoint_path("baseline", best=True)
    if not os.path.exists(baseline_path):
        logger.warning(f"baseline teacher not found: {baseline_path}")
        return None

    checkpoint = torch.load(baseline_path, map_location=DEVICE)
    if checkpoint.get("arch_version") != MODEL_ARCH_VERSION:
        logger.warning("baseline teacher architecture does not match current code, skip teacher guidance")
        return None

    teacher = EnhancedViTCompressor(use_semantic=False).to(DEVICE)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    logger.info(f"loaded baseline teacher from {baseline_path}")
    return teacher


def build_alex_lpips_model(experiment):
    if not experiment["use_alex_lpips_objectives"]:
        return None

    model = lpips.LPIPS(net="alex").to(DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def resolve_experiment_name(use_semantic=True, variant_name=None):
    if variant_name is not None:
        return variant_name
    return DEFAULT_ENHANCED_EXPERIMENT if use_semantic else DEFAULT_BASELINE_EXPERIMENT


def train_model(use_semantic=True, variant_name=None):
    experiment_name = resolve_experiment_name(use_semantic=use_semantic, variant_name=variant_name)
    experiment = get_experiment(experiment_name)
    logger.info(f"========== start training {experiment_name} ==========")

    model = build_model(experiment)
    model = load_pretrained_weights(model, experiment_name, logger)
    clip_extractor = (
        CLIPFeatureExtractor()
        if experiment["use_semantic_modules"] or experiment["use_clip_semantic_loss"]
        else None
    )
    baseline_teacher = load_baseline_teacher(experiment, logger)
    alex_lpips_model = build_alex_lpips_model(experiment)

    train_loader = build_dataloader(split="train", batch_size=BATCH_SIZE)
    val_loader = build_dataloader(split="val", batch_size=BATCH_SIZE)
    logger.info(f"train batches={len(train_loader)} | val batches={len(val_loader)}")

    optimizer = build_optimizer(model, experiment)
    aux_optimizer = optim.Adam(model.aux_parameters(), lr=AUX_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_recon = 0.0
        epoch_train_perceptual = 0.0
        epoch_train_mse = 0.0
        epoch_train_bpp = 0.0
        epoch_train_semantic = 0.0
        epoch_train_contrastive = 0.0
        epoch_train_mismatch = 0.0
        epoch_train_alex_lpips = 0.0
        epoch_train_alex_gap = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for img in pbar:
            img = img.to(DEVICE, dtype=torch.float32)
            clip_features = clip_extractor(img) if clip_extractor is not None else None
            model_clip_features = semantic_features_for_mode(clip_features, experiment["semantic_input_mode"])

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                recon_img, loss_dict = model(img, model_clip_features)
                total_loss = loss_dict["recon_loss"] * 100.0 + loss_dict["bpp"] * RATE_LOSS_WEIGHT
                semantic_loss = torch.zeros((), device=DEVICE)
                contrastive_loss = torch.zeros((), device=DEVICE)
                mismatch_loss = torch.zeros((), device=DEVICE)
                alex_lpips_loss = torch.zeros((), device=DEVICE)
                alex_gap_loss = torch.zeros((), device=DEVICE)
                recon_clip_features = None

                if experiment["use_clip_semantic_loss"]:
                    recon_clip_features = clip_extractor.encode(recon_img)
                    semantic_loss = (1.0 - F.cosine_similarity(recon_clip_features, clip_features.detach(), dim=-1)).mean()
                    total_loss = total_loss + semantic_loss * SEMANTIC_LOSS_WEIGHT

                if experiment["use_clip_contrastive_loss"]:
                    if recon_clip_features is None:
                        recon_clip_features = clip_extractor.encode(recon_img)
                    logits = recon_clip_features @ clip_features.detach().transpose(0, 1)
                    logits = logits / SEMANTIC_CONTRASTIVE_TEMPERATURE
                    targets = torch.arange(logits.shape[0], device=DEVICE)
                    contrastive_loss = 0.5 * (
                        F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets)
                    )
                    total_loss = total_loss + contrastive_loss * SEMANTIC_CONTRASTIVE_WEIGHT

                if experiment["use_semantic_mismatch_loss"] and experiment["use_semantic_modules"] and clip_features.shape[0] > 1:
                    shuffled_clip_features = semantic_features_for_mode(clip_features, "shuffle")
                    mismatch_recon_img, _ = model(img, shuffled_clip_features)
                    mismatch_clip_features = clip_extractor.encode(mismatch_recon_img)
                    if recon_clip_features is None:
                        recon_clip_features = clip_extractor.encode(recon_img)
                    matched_similarity = F.cosine_similarity(
                        recon_clip_features,
                        clip_features.detach(),
                        dim=-1,
                    )
                    mismatched_similarity = F.cosine_similarity(
                        mismatch_clip_features,
                        clip_features.detach(),
                        dim=-1,
                    )
                    mismatch_loss = F.relu(
                        mismatched_similarity - matched_similarity + SEMANTIC_MISMATCH_MARGIN
                    ).mean()
                    total_loss = total_loss + mismatch_loss * SEMANTIC_MISMATCH_WEIGHT

                if experiment["use_alex_lpips_objectives"]:
                    original_01 = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
                    recon_01 = ((recon_img + 1.0) / 2.0).clamp(0.0, 1.0)
                    alex_lpips_loss = alex_lpips_model(recon_01 * 2.0 - 1.0, original_01 * 2.0 - 1.0).mean()
                    total_loss = total_loss + alex_lpips_loss * ALEX_LPIPS_LOSS_WEIGHT

                    if baseline_teacher is not None:
                        with torch.no_grad():
                            teacher_recon, _ = baseline_teacher(img, None)
                            teacher_recon_01 = ((teacher_recon + 1.0) / 2.0).clamp(0.0, 1.0)
                            teacher_alex_lpips = alex_lpips_model(
                                teacher_recon_01 * 2.0 - 1.0,
                                original_01 * 2.0 - 1.0,
                            ).mean()
                        alex_gap_loss = F.relu(
                            alex_lpips_loss - teacher_alex_lpips.detach() + ALEX_LPIPS_GAP_MARGIN
                        )
                        total_loss = total_loss + alex_gap_loss * ALEX_LPIPS_GAP_WEIGHT

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            aux_optimizer.zero_grad(set_to_none=True)
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_train_recon += loss_dict["recon_loss"].item()
            epoch_train_perceptual += loss_dict["perceptual_loss"].item()
            epoch_train_mse += loss_dict["mse_loss"].item()
            epoch_train_bpp += loss_dict["bpp"].item()
            epoch_train_semantic += semantic_loss.item()
            epoch_train_contrastive += contrastive_loss.item()
            epoch_train_mismatch += mismatch_loss.item()
            epoch_train_alex_lpips += alex_lpips_loss.item()
            epoch_train_alex_gap += alex_gap_loss.item()

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                recon=f"{loss_dict['recon_loss'].item():.4f}",
                bpp=f"{loss_dict['bpp'].item():.4f}",
                semantic=f"{semantic_loss.item():.4f}",
                contrastive=f"{contrastive_loss.item():.4f}",
                mismatch=f"{mismatch_loss.item():.4f}",
                alex_lpips=f"{alex_lpips_loss.item():.4f}",
                alex_gap=f"{alex_gap_loss.item():.4f}",
            )

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_semantic = 0.0
        epoch_val_contrastive = 0.0
        epoch_val_mismatch = 0.0
        epoch_val_alex_lpips = 0.0
        epoch_val_alex_gap = 0.0
        with torch.no_grad():
            for img in val_loader:
                img = img.to(DEVICE, dtype=torch.float32)
                clip_features = clip_extractor(img) if clip_extractor is not None else None
                model_clip_features = semantic_features_for_mode(clip_features, experiment["semantic_input_mode"])
                recon_img, loss_dict = model(img, model_clip_features)
                total_loss = loss_dict["recon_loss"] * 100.0 + loss_dict["bpp"] * RATE_LOSS_WEIGHT
                recon_clip_features = None

                if experiment["use_clip_semantic_loss"]:
                    recon_clip_features = clip_extractor.encode(recon_img)
                    semantic_loss = (1.0 - F.cosine_similarity(recon_clip_features, clip_features, dim=-1)).mean()
                    total_loss = total_loss + semantic_loss * SEMANTIC_LOSS_WEIGHT
                    epoch_val_semantic += semantic_loss.item()

                if experiment["use_clip_contrastive_loss"]:
                    if recon_clip_features is None:
                        recon_clip_features = clip_extractor.encode(recon_img)
                    logits = recon_clip_features @ clip_features.transpose(0, 1)
                    logits = logits / SEMANTIC_CONTRASTIVE_TEMPERATURE
                    targets = torch.arange(logits.shape[0], device=DEVICE)
                    contrastive_loss = 0.5 * (
                        F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets)
                    )
                    total_loss = total_loss + contrastive_loss * SEMANTIC_CONTRASTIVE_WEIGHT
                    epoch_val_contrastive += contrastive_loss.item()

                if experiment["use_semantic_mismatch_loss"] and experiment["use_semantic_modules"] and clip_features.shape[0] > 1:
                    shuffled_clip_features = semantic_features_for_mode(clip_features, "shuffle")
                    mismatch_recon_img, _ = model(img, shuffled_clip_features)
                    mismatch_clip_features = clip_extractor.encode(mismatch_recon_img)
                    if recon_clip_features is None:
                        recon_clip_features = clip_extractor.encode(recon_img)
                    matched_similarity = F.cosine_similarity(recon_clip_features, clip_features, dim=-1)
                    mismatched_similarity = F.cosine_similarity(mismatch_clip_features, clip_features, dim=-1)
                    mismatch_loss = F.relu(
                        mismatched_similarity - matched_similarity + SEMANTIC_MISMATCH_MARGIN
                    ).mean()
                    total_loss = total_loss + mismatch_loss * SEMANTIC_MISMATCH_WEIGHT
                    epoch_val_mismatch += mismatch_loss.item()

                if experiment["use_alex_lpips_objectives"]:
                    original_01 = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
                    recon_01 = ((recon_img + 1.0) / 2.0).clamp(0.0, 1.0)
                    alex_lpips_loss = alex_lpips_model(recon_01 * 2.0 - 1.0, original_01 * 2.0 - 1.0).mean()
                    total_loss = total_loss + alex_lpips_loss * ALEX_LPIPS_LOSS_WEIGHT
                    epoch_val_alex_lpips += alex_lpips_loss.item()

                    if baseline_teacher is not None:
                        teacher_recon, _ = baseline_teacher(img, None)
                        teacher_recon_01 = ((teacher_recon + 1.0) / 2.0).clamp(0.0, 1.0)
                        teacher_alex_lpips = alex_lpips_model(
                            teacher_recon_01 * 2.0 - 1.0,
                            original_01 * 2.0 - 1.0,
                        ).mean()
                        alex_gap_loss = F.relu(
                            alex_lpips_loss - teacher_alex_lpips + ALEX_LPIPS_GAP_MARGIN
                        )
                        total_loss = total_loss + alex_gap_loss * ALEX_LPIPS_GAP_WEIGHT
                        epoch_val_alex_gap += alex_gap_loss.item()

                epoch_val_loss += total_loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_train_recon = epoch_train_recon / len(train_loader)
        avg_train_perceptual = epoch_train_perceptual / len(train_loader)
        avg_train_mse = epoch_train_mse / len(train_loader)
        avg_train_bpp = epoch_train_bpp / len(train_loader)
        avg_train_semantic = epoch_train_semantic / len(train_loader)
        avg_train_contrastive = epoch_train_contrastive / len(train_loader)
        avg_train_mismatch = epoch_train_mismatch / len(train_loader)
        avg_train_alex_lpips = epoch_train_alex_lpips / len(train_loader)
        avg_train_alex_gap = epoch_train_alex_gap / len(train_loader)
        avg_val_semantic = epoch_val_semantic / len(val_loader) if experiment["use_clip_semantic_loss"] else 0.0
        avg_val_contrastive = epoch_val_contrastive / len(val_loader) if experiment["use_clip_contrastive_loss"] else 0.0
        avg_val_mismatch = epoch_val_mismatch / len(val_loader) if experiment["use_semantic_mismatch_loss"] else 0.0
        avg_val_alex_lpips = epoch_val_alex_lpips / len(val_loader) if experiment["use_alex_lpips_objectives"] else 0.0
        avg_val_alex_gap = epoch_val_alex_gap / len(val_loader) if experiment["use_alex_lpips_objectives"] else 0.0

        scheduler.step(avg_val_loss)

        logger.info(f"Epoch {epoch} training summary")
        logger.info(f"  experiment={experiment_name}")
        logger.info(f"  train_total={avg_train_loss:.4f} | val_total={avg_val_loss:.4f}")
        logger.info(
            f"  recon={avg_train_recon:.4f} "
            f"(perceptual={avg_train_perceptual:.4f}, mse={avg_train_mse:.4f})"
        )
        logger.info(f"  bpp={avg_train_bpp:.4f}")
        if experiment["use_clip_semantic_loss"]:
            logger.info(f"  semantic_loss={avg_train_semantic:.4f}")
            logger.info(f"  val_semantic_loss={avg_val_semantic:.4f}")
        if experiment["use_clip_contrastive_loss"]:
            logger.info(f"  contrastive_loss={avg_train_contrastive:.4f}")
            logger.info(f"  val_contrastive_loss={avg_val_contrastive:.4f}")
        if experiment["use_semantic_mismatch_loss"]:
            logger.info(f"  mismatch_loss={avg_train_mismatch:.4f}")
            logger.info(f"  val_mismatch_loss={avg_val_mismatch:.4f}")
        if experiment["use_alex_lpips_objectives"]:
            logger.info(f"  alex_lpips_loss={avg_train_alex_lpips:.4f}")
            logger.info(f"  val_alex_lpips_loss={avg_val_alex_lpips:.4f}")
            logger.info(f"  alex_lpips_gap_loss={avg_train_alex_gap:.4f}")
            logger.info(f"  val_alex_lpips_gap_loss={avg_val_alex_gap:.4f}")
        logger.info(f"  lr={optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = checkpoint_path(experiment["checkpoint_stem"], best=True)
            torch.save(
                {
                    "epoch": epoch,
                    "arch_version": MODEL_ARCH_VERSION,
                    "experiment": experiment_name,
                    "use_semantic": experiment["use_semantic_modules"],
                    "semantic_input_mode": experiment["semantic_input_mode"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "aux_optimizer_state_dict": aux_optimizer.state_dict(),
                    "best_loss": best_val_loss,
                },
                save_path,
            )
            logger.info(f"saved best {experiment_name} checkpoint to {save_path} with val_total={best_val_loss:.4f}")

    final_save_path = checkpoint_path(experiment["checkpoint_stem"], best=False)
    torch.save(
        {
            "epoch": EPOCHS,
            "arch_version": MODEL_ARCH_VERSION,
            "experiment": experiment_name,
            "use_semantic": experiment["use_semantic_modules"],
            "semantic_input_mode": experiment["semantic_input_mode"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "aux_optimizer_state_dict": aux_optimizer.state_dict(),
            "final_loss": avg_val_loss,
        },
        final_save_path,
    )
    logger.info(f"finished training {experiment_name}, final checkpoint saved to {final_save_path}")


if __name__ == "__main__":
    if TRAIN_ENHANCED_ONLY:
        train_model(variant_name=DEFAULT_ENHANCED_EXPERIMENT)
    else:
        train_model(variant_name=DEFAULT_BASELINE_EXPERIMENT)
        train_model(variant_name=DEFAULT_ENHANCED_EXPERIMENT)
