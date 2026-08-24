"""
I2IQA数据集交叉验证数据加载器
支持质量(q)、真实性(a)、一致性(c)三类评分。
"""

import os
from .cv_utils import (
    load_cv_data_from_csv,
    get_image_transforms,
    create_cv_dataloader,
    get_cv_folds_csv_paths
)


def get_I2IQA_CV_dataloaders(args, fold=1, score_type='q'):
    """
    获取I2IQA数据集的交叉验证数据加载器。

    Args:
        args: 配置参数
        fold: 交叉验证fold编号 (1-5)
        score_type: 评分类型 ('q'=quality, 'a'=authenticity, 'c'=correspondence)

    Returns:
        dict: 包含train和val数据加载器的字典
    """
    if score_type not in ['q', 'a', 'c']:
        raise ValueError(f"score_type必须为'q'、'a'或'c'，当前: {score_type}")

    image_base_path = "/home/dataset/I2IQA/Generated_image/All"

    if hasattr(args, 'text_encoder') and args.text_encoder == 'clip_convnext':
        text_encoder_path = "clip_convnext"
        print("🔤 使用CLIP-ConvNeXt文本编码器")
    else:
        text_encoder_path = "deberta-v3-base"
        print("🔤 使用DeBERTa文本编码器")

    score_column_map = {
        'q': 'MOS_q',
        'a': 'MOS_a',
        'c': 'MOS_c'
    }
    score_col = score_column_map[score_type]
    train_csv_path, val_csv_path = get_cv_folds_csv_paths(fold, 'I2IQA')

    print(f"🔄 I2IQA 交叉验证 - 第 {fold} 折 | 分数类型: {score_type} ({score_col})")
    print(f"   训练数据: {train_csv_path}")
    print(f"   验证数据: {val_csv_path}")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"训练数据文件不存在: {train_csv_path}")
    if not os.path.exists(val_csv_path):
        raise FileNotFoundError(f"验证数据文件不存在: {val_csv_path}")

    required_cols = ['image_name', 'prompt', score_col]
    train_images, train_labels, train_prompts, train_image_names = load_cv_data_from_csv(
        csv_path=train_csv_path,
        image_base_path=image_base_path,
        image_col='image_name',
        prompt_col='prompt',
        score_col=score_col,
        required_cols=required_cols
    )
    val_images, val_labels, val_prompts, val_image_names = load_cv_data_from_csv(
        csv_path=val_csv_path,
        image_base_path=image_base_path,
        image_col='image_name',
        prompt_col='prompt',
        score_col=score_col,
        required_cols=required_cols
    )

    train_transforms, test_transforms = get_image_transforms(args)
    dataloaders = {}

    print("📦 创建训练数据加载器...")
    dataloaders['train'] = create_cv_dataloader(
        images=train_images,
        labels=train_labels,
        prompts=train_prompts,
        transforms=train_transforms,
        text_encoder_path=text_encoder_path,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        image_names=train_image_names
    )

    print("📦 创建验证数据加载器...")
    dataloaders['val'] = create_cv_dataloader(
        images=val_images,
        labels=val_labels,
        prompts=val_prompts,
        transforms=test_transforms,
        text_encoder_path=text_encoder_path,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        image_names=val_image_names
    )

    dataset_name_map = {
        'q': 'I2IQAq',
        'a': 'I2IQAa',
        'c': 'I2IQAc'
    }
    dataloaders['fold'] = fold
    dataloaders['dataset'] = dataset_name_map[score_type]
    dataloaders['score_type'] = score_type

    print(f"✅ I2IQA 第 {fold} 折数据加载器创建完成 (score: {score_col})")
    print(f"   训练集: {len(train_images)} 张图像")
    print(f"   验证集: {len(val_images)} 张图像")

    return dataloaders


def get_I2IQAq_CV_dataloaders(args, fold=1):
    return get_I2IQA_CV_dataloaders(args, fold=fold, score_type='q')


def get_I2IQAa_CV_dataloaders(args, fold=1):
    return get_I2IQA_CV_dataloaders(args, fold=fold, score_type='a')


def get_I2IQAc_CV_dataloaders(args, fold=1):
    return get_I2IQA_CV_dataloaders(args, fold=fold, score_type='c')
