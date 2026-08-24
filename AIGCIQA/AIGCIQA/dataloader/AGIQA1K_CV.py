"""
AGIQA-1K数据集交叉验证数据加载器
支持使用预先划分的10折交叉验证数据
"""

import os
from .cv_utils import (
    load_cv_data_from_csv,
    get_image_transforms,
    create_cv_dataloader,
    get_cv_folds_csv_paths
)


def get_AGIQA1K_CV_dataloaders(args, fold=1):
    """
    获取AGIQA-1K数据集的交叉验证数据加载器
    
    Args:
        args: 配置参数
        fold: 交叉验证的fold编号 (1-5)
        
    Returns:
        dict: 包含train和val数据加载器的字典
    """
    # 路径配置
    image_base_path = "/home/dataset/AGIQA-1K/file"
    
    # 根据args.text_encoder选择text_encoder_path
    if hasattr(args, 'text_encoder') and args.text_encoder == 'clip_convnext':
        text_encoder_path = "clip_convnext"
        print("🔤 使用CLIP-ConvNeXt文本编码器")
    else:
        text_encoder_path = "deberta-v3-base"
        print("🔤 使用DeBERTa文本编码器")
    
    train_csv_path, val_csv_path = get_cv_folds_csv_paths(fold, 'AGIQA1K')
    
    print(f"🔄 AGIQA-1K 交叉验证 - 第 {fold} 折")
    print(f"   训练数据: {train_csv_path}")
    print(f"   验证数据: {val_csv_path}")
    
    # 检查文件是否存在
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"训练数据文件不存在: {train_csv_path}")
    if not os.path.exists(val_csv_path):
        raise FileNotFoundError(f"验证数据文件不存在: {val_csv_path}")
    
    # 加载训练数据
    train_images, train_labels, train_prompts, train_image_names = load_cv_data_from_csv(
        csv_path=train_csv_path,
        image_base_path=image_base_path,
        image_col='image_name',
        prompt_col='prompt',
        score_col='quality_score',
        required_cols=['image_name', 'prompt', 'quality_score']
    )
    
    # 加载验证数据
    val_images, val_labels, val_prompts, val_image_names = load_cv_data_from_csv(
        csv_path=val_csv_path,
        image_base_path=image_base_path,
        image_col='image_name',
        prompt_col='prompt',
        score_col='quality_score',
        required_cols=['image_name', 'prompt', 'quality_score']
    )
    
    # 获取图像变换
    train_transforms, test_transforms = get_image_transforms(args)
    
    # 创建数据加载器
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
    
    # 添加额外信息
    dataloaders['fold'] = fold
    dataloaders['dataset'] = 'AGIQA1K'
    
    print(f"✅ AGIQA-1K 第 {fold} 折数据加载器创建完成")
    print(f"   训练集: {len(train_images)} 张图像")
    print(f"   验证集: {len(val_images)} 张图像")
    
    return dataloaders


def get_all_AGIQA1K_CV_folds(args):
    """
    获取AGIQA-1K所有5折的数据加载器
    
    Args:
        args: 配置参数
        
    Returns:
        dict: 所有fold的数据加载器字典 {fold: dataloaders}
    """
    all_folds = {}
    
    print("🔄 加载AGIQA-1K所有5折交叉验证数据...")
    
    for fold in range(1, 6):
        try:
            all_folds[fold] = get_AGIQA1K_CV_dataloaders(args, fold)
            print(f"✅ 第 {fold} 折加载成功")
        except Exception as e:
            print(f"❌ 第 {fold} 折加载失败: {e}")
            
    print(f"📊 总共成功加载 {len(all_folds)} 折数据")
    
    return all_folds 