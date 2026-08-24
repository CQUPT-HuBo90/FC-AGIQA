"""
AIGCIQA2023数据集交叉验证数据加载器
支持使用预先划分的10折交叉验证数据
支持不同的评分类型: quality (q), authenticity (a), correspondence (c)
"""

import os
import pandas as pd
from .cv_utils import (
    load_cv_data_from_csv,
    get_image_transforms,
    create_cv_dataloader,
    get_cv_folds_csv_paths
)


def load_aigciqa2023_cv_data_from_csv(csv_path, image_base_path, score_type='q', normalize_to_five=False):
    """
    从CSV文件加载AIGCIQA2023交叉验证数据，支持不同评分类型
    
    Args:
        csv_path: CSV文件路径
        image_base_path: 图像文件基础路径  
        score_type: 评分类型 ('q'=quality, 'a'=authenticity, 'c'=correspondence)
        normalize_to_five: 是否将评分归一化到[0,5]范围
        
    Returns:
        tuple: (image_list, label_list, text_prompt_list, image_names_list, mapping_info)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    # 读取CSV数据
    data = pd.read_csv(csv_path)
    print(f"📁 从 {csv_path} 加载AIGCIQA2023数据: {len(data)} 条记录 (评分类型: {score_type})")

    # 根据评分类型选择对应的列
    score_column_map = {
        'q': 'Quality' if 'Quality' in data.columns else 'mosz1',
        'a': 'Authenticity' if 'Authenticity' in data.columns else 'mosz2',
        'c': 'Correspondence' if 'Correspondence' in data.columns else 'mosz3'
    }

    if score_type not in score_column_map:
        raise ValueError(f"无效的评分类型: {score_type}, 必须是 'q', 'a' 或 'c'")

    score_col = score_column_map[score_type]

    # 检查必需的列
    required_cols = ['image_name', score_col, 'prompt']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少必需的列: {missing_cols}")

    # 提取数据
    image_names = data['image_name'].tolist()
    text_prompts = data['prompt'].tolist()
    scores = data[score_col].tolist()
    model_names = data['model'].tolist() if 'model' in data.columns else None
    
    # 处理评分标准化
    original_min = min(scores)
    original_max = max(scores)
    new_cv_score_cols = {'Quality', 'Authenticity', 'Correspondence'}

    if score_col in new_cv_score_cols:
        target_min = 0.0
        target_max = 5.0
        mapped_scores = [float(score) * 5.0 for score in scores]
        print(f"🎯 检测到新版cv_folds AIGCIQA2023评分列: {score_col}")
        print(f"   原始评分范围: [{original_min:.3f}, {original_max:.3f}] -> 映射到0-5分制")
    else:
        # AIGCIQA2023数据集检测：原生0-5范围，不需要映射
        # 判断方式：检查数据分布是否符合0-5范围的特征
        is_aigciqa2023_like = (original_min >= 0.0 and original_max <= 5.0 and (original_max - original_min) > 1.0)

        if is_aigciqa2023_like:
            # AIGCIQA2023数据集：直接使用原始分数，不进行映射
            print(f"🎯 检测到AIGCIQA2023数据集(原生0-5范围)，直接使用原始评分")
            print(f"   原始评分范围: [{original_min:.3f}, {original_max:.3f}]")
            print(f"   ✅ 跳过评分映射，保持原始0-5范围")

            # 直接使用原始分数
            mapped_scores = scores.copy()

            # 设置1:1映射信息
            target_min = original_min
            target_max = original_max
        else:
            # 其他数据集：按原有逻辑进行映射
            if normalize_to_five:
                target_min = 0.0
                target_max = 5.0
                print(f"📊 其他数据集评分从范围[{original_min:.3f}, {original_max:.3f}]映射到[0, 5]")
            else:
                target_min = 0.0
                target_max = 100.0
                print(f"📊 其他数据集评分从范围[{original_min:.3f}, {original_max:.3f}]映射到[0, 100]")

            # 线性映射评分
            mapped_scores = []
            for score in scores:
                mapped_score = (score - original_min) / (original_max - original_min) * (target_max - target_min) + target_min
                mapped_score = max(target_min, min(target_max, mapped_score))
                mapped_scores.append(mapped_score)
    
    # 加载图像
    from PIL import Image
    image_list = []
    valid_indices = []
    
    for i, image_name in enumerate(image_names):
        if model_names is None:
            image_path = os.path.join(image_base_path, str(image_name))
        else:
            model_name = model_names[i]
            image_path = os.path.join(image_base_path, str(model_name), str(image_name))

        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                image_list.append(image)
                valid_indices.append(i)
            except Exception as e:
                print(f"⚠️  加载图像失败 {image_path}: {e}")
        else:
            print(f"⚠️  图像文件不存在: {image_path}")
    
    # 过滤有效数据
    valid_image_names = [image_names[i] for i in valid_indices]
    valid_text_prompts = [text_prompts[i] for i in valid_indices]
    valid_scores = [mapped_scores[i] for i in valid_indices]
    
    print(f"✅ 成功加载 {len(image_list)} 张图像 (跳过 {len(image_names) - len(image_list)} 张)")
    
    # 映射信息
    mapping_info = {
        'original_min': original_min,
        'original_max': original_max,
        'target_min': target_min,
        'target_max': target_max,
        'score_type': score_type
    }
    
    return image_list, valid_scores, valid_text_prompts, valid_image_names, mapping_info


def get_AIGCIQA2023_CV_dataloaders(args, fold=1, score_type='q'):
    """
    获取AIGCIQA2023数据集的交叉验证数据加载器
    
    Args:
        args: 配置参数
        fold: 交叉验证的fold编号 (1-5)
        score_type: 评分类型 ('q'=quality, 'a'=authenticity, 'c'=correspondence)
        
    Returns:
        dict: 包含train和val数据加载器的字典
    """
    # 验证评分类型
    if score_type not in ['q', 'a', 'c']:
        raise ValueError(f"无效的评分类型: {score_type}, 必须是 'q', 'a' 或 'c'")
    
    # 路径配置 (新版cv_folds的image_name可能已包含模型子目录)
    image_base_path = "/home/dataset/AIGCIQA2023/Image"
    
    # 根据args.text_encoder选择text_encoder_path
    if hasattr(args, 'text_encoder') and args.text_encoder == 'clip_convnext':
        text_encoder_path = "clip_convnext"
        print("🔤 使用CLIP-ConvNeXt文本编码器")
    else:
        text_encoder_path = "deberta-v3-base"
        print("🔤 使用DeBERTa文本编码器")
    
    train_csv_path, val_csv_path = get_cv_folds_csv_paths(fold, 'AIGCIQA2023')

    # 检查是否使用5分制
    normalize_to_five = hasattr(args, 'normalize_scores_to_five') and args.normalize_scores_to_five
    
    print(f"🔄 AIGCIQA2023 交叉验证 - 第 {fold} 折 (评分类型: {score_type})")
    print(f"   训练数据: {train_csv_path}")
    print(f"   验证数据: {val_csv_path}")
    print(f"   评分标准化: {'5分制' if normalize_to_five else '100分制'}")
    
    # 检查文件是否存在
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"训练数据文件不存在: {train_csv_path}")
    if not os.path.exists(val_csv_path):
        raise FileNotFoundError(f"验证数据文件不存在: {val_csv_path}")
    
    # 加载训练数据
    train_images, train_labels, train_prompts, train_image_names, train_mapping_info = load_aigciqa2023_cv_data_from_csv(
        csv_path=train_csv_path,
        image_base_path=image_base_path,
        score_type=score_type,
        normalize_to_five=normalize_to_five
    )
    
    # 加载验证数据
    val_images, val_labels, val_prompts, val_image_names, val_mapping_info = load_aigciqa2023_cv_data_from_csv(
        csv_path=val_csv_path,
        image_base_path=image_base_path,
        score_type=score_type,
        normalize_to_five=normalize_to_five
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
    dataloaders['dataset'] = 'AIGCIQA2023'
    dataloaders['score_type'] = score_type
    dataloaders['mapping_info'] = train_mapping_info  # 训练和验证的映射信息应该相同
    dataloaders['normalize_to_five'] = normalize_to_five
    
    print(f"✅ AIGCIQA2023 第 {fold} 折数据加载器创建完成 (评分类型: {score_type})")
    print(f"   训练集: {len(train_images)} 张图像")
    print(f"   验证集: {len(val_images)} 张图像")
    
    return dataloaders


def get_all_AIGCIQA2023_CV_folds(args, score_type='q'):
    """
    获取AIGCIQA2023所有5折的数据加载器
    
    Args:
        args: 配置参数
        score_type: 评分类型 ('q'=quality, 'a'=authenticity, 'c'=correspondence)
        
    Returns:
        dict: 所有fold的数据加载器字典 {fold: dataloaders}
    """
    all_folds = {}
    
    print(f"🔄 加载AIGCIQA2023所有5折交叉验证数据 (评分类型: {score_type})...")

    for fold in range(1, 6):
        try:
            all_folds[fold] = get_AIGCIQA2023_CV_dataloaders(args, fold, score_type)
            print(f"✅ 第 {fold} 折加载成功")
        except Exception as e:
            print(f"❌ 第 {fold} 折加载失败: {e}")
            
    print(f"📊 总共成功加载 {len(all_folds)} 折数据")
    
    return all_folds 