"""
AIGCIQA-20K official split dataloader.

The dataset provides fixed train/val/test image folders. Validation and test
annotations only include image names and prompts, so MOS labels are completed
from info_merge.csv by image name.
"""

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .cv_utils import get_image_transforms, worker_init_fn

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


DATASET_ROOT = "/home/dataset/AIGCIQA-20k"


class AIGCIQA20KDataset(Dataset):
    """Lazy image-loading dataset for AIGCIQA-20K."""

    def __init__(self, image_paths, labels, prompts, transforms, text_encoder_path, image_names):
        self.image_paths = image_paths
        self.label = labels
        self.text_prompt = prompts
        self.transforms = transforms
        self.image_names = image_names

        if text_encoder_path == "clip_convnext":
            if not OPEN_CLIP_AVAILABLE:
                raise ImportError("open_clip不可用，无法创建CLIP-ConvNeXt tokenizer")
            self.tokenizer = open_clip.get_tokenizer('convnext_large_d_320')
            self.use_clip_tokenizer = True
            self.max_length = 77
        else:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers不可用，无法创建文本tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            self.use_clip_tokenizer = False
            self.max_length = 512

        if len(image_paths) != len(labels) or len(image_paths) != len(prompts):
            raise ValueError(
                f"AIGCIQA20K数据维度不匹配: 图像={len(image_paths)}, "
                f"标签={len(labels)}, 提示词={len(prompts)}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        text_prompt = self.text_prompt[idx]
        if not text_prompt or pd.isna(text_prompt):
            text_prompt = "empty prompt"

        if self.use_clip_tokenizer:
            encoded_tokens = self.tokenizer([text_prompt])
            if encoded_tokens.dim() == 2:
                encoded_tokens = encoded_tokens.squeeze(0)
            attention_mask = torch.ones_like(encoded_tokens)
            prompt = encoded_tokens
        else:
            encoded = self.tokenizer.encode_plus(
                text_prompt,
                add_special_tokens=True,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
            prompt = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)

        return {
            'image': self.transforms(image),
            'MOS_score': self.label[idx],
            'prompt': prompt,
            'attention_mask': attention_mask,
            'image_name': self.image_names[idx]
        }


def _text_encoder_path(args):
    if hasattr(args, 'text_encoder') and args.text_encoder == 'clip_convnext':
        print("🔤 使用CLIP-ConvNeXt文本编码器")
        return "clip_convnext"
    print("🔤 使用DeBERTa文本编码器")
    return "deberta-v3-base"


def _read_official_split(split):
    if split not in {'train', 'val', 'test'}:
        raise ValueError(f"AIGCIQA20K split必须是train/val/test，当前: {split}")

    split_path = os.path.join(DATASET_ROOT, f"info_{split}.xlsx")
    merge_path = os.path.join(DATASET_ROOT, "info_merge.csv")
    image_base_path = os.path.join(DATASET_ROOT, split)

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"AIGCIQA20K官方划分文件不存在: {split_path}")
    if not os.path.exists(merge_path):
        raise FileNotFoundError(f"AIGCIQA20K完整标注文件不存在: {merge_path}")
    if not os.path.isdir(image_base_path):
        raise FileNotFoundError(f"AIGCIQA20K图像目录不存在: {image_base_path}")

    split_df = pd.read_excel(split_path)
    merge_df = pd.read_csv(merge_path)
    required_split_cols = {'name', 'prompt'}
    required_merge_cols = {'name', 'mos'}

    missing_split = sorted(required_split_cols - set(split_df.columns))
    missing_merge = sorted(required_merge_cols - set(merge_df.columns))
    if missing_split:
        raise ValueError(f"{split_path}缺少必需列: {missing_split}")
    if missing_merge:
        raise ValueError(f"{merge_path}缺少必需列: {missing_merge}")

    if 'mos' not in split_df.columns:
        split_df = split_df.merge(merge_df[['name', 'mos']], on='name', how='left')

    missing_mos = split_df['mos'].isna().sum()
    if missing_mos:
        raise ValueError(f"AIGCIQA20K {split} split有{missing_mos}条样本无法从info_merge.csv匹配MOS")

    image_paths = []
    labels = []
    prompts = []
    image_names = []

    for row in split_df.itertuples(index=False):
        image_name = str(getattr(row, 'name'))
        image_path = os.path.join(image_base_path, image_name)
        if not os.path.exists(image_path):
            print(f"⚠️  图像文件不存在: {image_path}")
            continue
        image_paths.append(image_path)
        labels.append(float(getattr(row, 'mos')))
        prompts.append(str(getattr(row, 'prompt')))
        image_names.append(image_name)

    print(f"📁 AIGCIQA20K {split}: 标注{len(split_df)}条，匹配到{len(image_paths)}张图像")
    if labels:
        print(f"   MOS范围: [{min(labels):.3f}, {max(labels):.3f}], mean={sum(labels) / len(labels):.3f}")

    return image_paths, labels, prompts, image_names


def _create_lazy_dataloader(image_paths, labels, prompts, transforms, text_encoder_path,
                            batch_size, shuffle, drop_last, image_names):
    dataset = AIGCIQA20KDataset(
        image_paths=image_paths,
        labels=labels,
        prompts=prompts,
        transforms=transforms,
        text_encoder_path=text_encoder_path,
        image_names=image_names
    )
    num_workers = min(4, os.cpu_count() or 0)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )


def get_AIGCIQA20K_dataloaders(args):
    """Get dataloaders for the official AIGCIQA-20K train/val/test split."""
    print("🔄 AIGCIQA20K 官方划分: train/val/test")
    train_paths, train_labels, train_prompts, train_image_names = _read_official_split('train')
    val_paths, val_labels, val_prompts, val_image_names = _read_official_split('val')
    test_paths, test_labels, test_prompts, test_image_names = _read_official_split('test')

    train_transforms, test_transforms = get_image_transforms(args)
    text_encoder_path = _text_encoder_path(args)

    dataloaders = {
        'train': _create_lazy_dataloader(
            image_paths=train_paths,
            labels=train_labels,
            prompts=train_prompts,
            transforms=train_transforms,
            text_encoder_path=text_encoder_path,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=True,
            image_names=train_image_names
        ),
        'val': _create_lazy_dataloader(
            image_paths=val_paths,
            labels=val_labels,
            prompts=val_prompts,
            transforms=test_transforms,
            text_encoder_path=text_encoder_path,
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=False,
            image_names=val_image_names
        ),
        'test': _create_lazy_dataloader(
            image_paths=test_paths,
            labels=test_labels,
            prompts=test_prompts,
            transforms=test_transforms,
            text_encoder_path=text_encoder_path,
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=False,
            image_names=test_image_names
        ),
        'dataset': 'AIGCIQA20K',
        'split_protocol': 'official-train-val-test'
    }

    print("✅ AIGCIQA20K官方划分数据加载器创建完成")
    print(f"   train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")
    return dataloaders
