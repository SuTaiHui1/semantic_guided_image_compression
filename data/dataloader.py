import os
import random

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from configs.base_config import (
    BATCH_SIZE,
    DEVICE,
    DIV2K_TRAIN_PATH,
    IMAGE_SIZE,
    KODAK_TEST_PATH,
    RANDOM_SEED,
    TEST_BATCH_SIZE,
    VAL_SPLIT,
)


def _list_images(root):
    files = []
    for file_name in sorted(os.listdir(root)):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, file_name)
            if os.path.exists(img_path):
                files.append((img_path, file_name))
    if not files:
        raise FileNotFoundError(f"no image files found under {root}")
    return files


def _split_train_val(img_info):
    rng = random.Random(RANDOM_SEED)
    shuffled = list(img_info)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * VAL_SPLIT))
    val_info = sorted(shuffled[:val_count], key=lambda item: item[1])
    train_info = sorted(shuffled[val_count:], key=lambda item: item[1])
    return train_info, val_info


class ImageDataset(Dataset):
    def __init__(self, img_info, transform=None, return_name=False):
        self.img_info = img_info
        self.transform = transform
        self.return_name = return_name

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        img_path, img_name = self.img_info[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"failed to read image: {img_path}") from exc

        if self.transform is not None:
            img = self.transform(img)

        if self.return_name:
            return img, img_name
        return img


def build_dataloader(split="train", batch_size=None):
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if batch_size is None:
        batch_size = BATCH_SIZE if split != "test" else TEST_BATCH_SIZE

    if split == "test":
        dataset = ImageDataset(_list_images(KODAK_TEST_PATH), transform=test_transform, return_name=True)
        shuffle = False
        drop_last = False
    else:
        train_info, val_info = _split_train_val(_list_images(DIV2K_TRAIN_PATH))
        dataset = ImageDataset(
            train_info if split == "train" else val_info,
            transform=train_transform if split == "train" else eval_transform,
            return_name=False,
        )
        shuffle = split == "train"
        drop_last = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 if os.name == "nt" else 4,
        pin_memory=DEVICE.type == "cuda",
        drop_last=drop_last,
    )


if __name__ == "__main__":
    for split_name in ("train", "val", "test"):
        loader = build_dataloader(split=split_name)
        batch = next(iter(loader))
        if split_name == "test":
            img, name = batch
            print(split_name, img.shape, name[0])
        else:
            print(split_name, batch.shape)
