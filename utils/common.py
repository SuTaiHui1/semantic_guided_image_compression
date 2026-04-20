import torch
import os
import logging
from configs.base_config import LOG_PATH, MODEL_SAVE_PATH, DEVICE

# 初始化日志
def init_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOG_PATH, "train.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# 保存模型
def save_model(model, epoch, is_baseline=False):
    model_name = "baseline_vit.pth" if is_baseline else "enhanced_vit.pth"
    save_path = os.path.join(MODEL_SAVE_PATH, model_name)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }, save_path)
    logging.info(f"模型已保存至：{save_path}")

# 加载模型
def load_model(model, is_baseline=False):
    model_name = "baseline_vit.pth" if is_baseline else "enhanced_vit.pth"
    load_path = os.path.join(MODEL_SAVE_PATH, model_name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"模型文件不存在：{load_path}")
    
    checkpoint = torch.load(load_path, map_location=DEVICE)
    # 兼容base_vit前缀的权重加载
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    logging.info(f"模型已从：{load_path} 加载")
    return model