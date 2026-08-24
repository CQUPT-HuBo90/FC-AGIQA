import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_info',
                        type=str,
                        help='info that will be displayed when logging',
                        default='AGIQA3K')

    parser.add_argument('--lr',
                        type=float,
                        help='learning rate',
                        default=1e-4)

    parser.add_argument('--contrastive_weight',
                        type=float,
                        help='weight for contrastive loss',
                        default=0.5)

    parser.add_argument('--text_lr_factor',
                        type=float,
                        help='learning rate factor for text encoder',
                        default=0.1)

    parser.add_argument('--image_lr_factor',
                        type=float,
                        help='learning rate factor for image encoder',
                        default=0.5)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='L2 weight decay',
                        default=1e-5)

    parser.add_argument('--seed',
                        type=int,
                        help='manual seed',
                        default=224)

    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        default='0')  # 改为默认GPU 0，避免与实际使用冲突

    parser.add_argument('--train_batch_size',
                        type=int,
                        help='batch size for training phase',
                        default=16)

    parser.add_argument('--test_batch_size',
                        type=int,
                        help='batch size for test phase',
                        default=32)

    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of training epochs',
                        default=200)

    parser.add_argument('--save_best_start_epoch',
                        type=int,
                        help='从第N个epoch开始保存最优模型(人类计数，从1开始)',
                        default=1)

    parser.add_argument('--patience',
                        type=int,
                        help='patience for early stopping',
                        default=200)

    parser.add_argument('--scheduler',
                        type=str,
                        help='learning rate scheduler',
                        choices=['cosine', 'warmup_cosine', 'plateau'],
                        default='cosine')

    parser.add_argument('--quality_preset',
                        type=str,
                        choices=['none', 'iqa_stable'],
                        default='none',
                        help='quality assessment training preset')

    parser.add_argument('--min_lr',
                        type=float,
                        help='minimum learning rate for cosine annealing scheduler',
                        default=1e-6)

    parser.add_argument('--t_max',
                        type=int,
                        help='T_max for cosine annealing scheduler (default: num_epochs)',
                        default=None)

    parser.add_argument('--warmup_epochs',
                        type=int,
                        help='warmup epochs for warmup_cosine scheduler',
                        default=5)

    parser.add_argument('--verbose',
                        action='store_true',
                        help='print verbose debug information',
                        default=False)

    parser.add_argument('--backbone',
                        type=str,
                        help='which backbone model to use',
                        default='swin',
                        choices=['swin', 'clipconvnext'])

    parser.add_argument('--text_encoder',
                        type=str,
                        help='which text encoder to use',
                        default='deberta',
                        choices=['bert', 'roberta', 'deberta', 'clip_convnext'])

    parser.add_argument('--text_encoder_path',
                        type=str,
                        help='path to pretrained text encoder',
                        default='./deberta-v3-base')

    parser.add_argument('--benchmark',
                        type=str,
                        help='which dataset to use',
                        default='AGIQA3K')

    # 跨数据集实验参数（本模型 train-on-source / test-on-target）
    parser.add_argument('--cross_dataset', action='store_true',
                       help='启用跨数据集实验模式')
    parser.add_argument('--train_benchmark', type=str, default='',
                       choices=['', 'AGIQA1K', 'AGIQA3K', 'AIGCIQA2023q', 'I2IQAq'],
                       help='跨数据集实验的训练源数据集')
    parser.add_argument('--test_benchmark', type=str, default='',
                       choices=['', 'AGIQA1K', 'AGIQA3K', 'AIGCIQA2023q', 'I2IQAq'],
                       help='跨数据集实验的测试目标数据集')
    parser.add_argument('--cross_results_csv', type=str, default='cross_dataset_results.csv',
                       help='跨数据集实验结果CSV文件路径')

    parser.add_argument('--train_aug_policy',
                        type=str,
                        choices=['legacy', 'quality_stable', 'correspondence_stable'],
                        default='legacy',
                        help='clipconvnext training augmentation policy')

    parser.add_argument('--use_frequency_features',
                        action='store_true',
                        help='是否使用频域特征提取',
                        default=True)

    parser.add_argument('--freq_feature_dim',
                        type=int,
                        help='频域特征输出维度',
                        default=768)

    parser.add_argument('--num_heads',
                        type=int,
                        help='多头注意力机制的头数',
                        default=4)

    # 增强功能配置
    parser.add_argument("--enhancement_stage", type=int, default=1, choices=[1, 2, 3, 4],
                       help="增强功能实施阶段 (1: 频域特征, 2: 多模态融合, 3: 损失函数, 4: 训练策略)")

    parser.add_argument("--use_enhanced_fusion", action="store_true", default=False,
                       help="使用高级多模态融合")
    parser.add_argument("--fusion_type", type=str, default="standard", 
                       choices=["standard", "transformer", "dynamic", "gated"],
                       help="多模态融合类型")
    
    # 频域增强融合参数
    parser.add_argument("--use_freq_enhanced_fusion", action="store_true", default=False,
                       help="使用两阶段频域增强融合")
    parser.add_argument("--freq_enhancement_weight", type=float, default=0.1,
                       help="频域增强权重初始值")
    parser.add_argument("--contrastive_temperature", type=float, default=0.07,
                       help="对比学习温度参数")
    # 全局InfoNCE对比（批内负样本）
    parser.add_argument('--use_global_contrastive', action='store_true', default=False,
                       help='使用全局InfoNCE对比损失（批内负样本），增强文本-图像对齐')
    parser.add_argument('--global_contrastive_weight', type=float, default=0.2,
                       help='全局InfoNCE对比损失权重')
    parser.add_argument("--use_innovative_loss", action="store_true", default=False,
                       help="使用创新损失函数")
    parser.add_argument("--use_enhanced_training", action="store_true", default=False,
                       help="使用高级训练策略")

    parser.add_argument('--loss_smooth_l1_weight', type=float, default=0.55,
                       help='SmoothL1 loss weight in quality regression loss')
    parser.add_argument('--loss_pearson_weight', type=float, default=0.20,
                       help='Pearson correlation loss weight in quality regression loss')
    parser.add_argument('--loss_rank_weight', type=float, default=0.25,
                       help='Pairwise ranking loss weight in quality regression loss')
    parser.add_argument('--rank_min_diff', type=float, default=0.03,
                       help='minimum MOS difference used by pairwise ranking loss')
    parser.add_argument('--rank_temperature', type=float, default=0.5,
                       help='temperature used by pairwise ranking loss')
    parser.add_argument('--rank_loss_type', type=str, default='hard',
                       choices=['hard', 'soft_weighted'],
                       help='pairwise rank loss variant')
    parser.add_argument('--label_norm', type=str, default='none',
                       choices=['none', 'train_zscore'],
                       help='fold-level label normalization used for training and correlation metrics')
    parser.add_argument('--freeze_text_encoder', action='store_true', default=False,
                       help='freeze text encoder parameters during training')
    parser.add_argument('--log_training_diagnostics', action='store_true', default=False,
                       help='print label/loss/rank diagnostics for the first training batches')

    # 第四阶段：高级训练策略
    parser.add_argument('--use_advanced_training', action='store_true', help='启用高级训练策略')
    parser.add_argument('--use_ema', action='store_true', help='启用指数移动平均')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA衰减率')
    parser.add_argument('--use_mixup', action='store_true', help='启用Mixup数据增强')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup参数alpha')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--use_progressive_learning', action='store_true', help='启用渐进式学习')
    parser.add_argument('--progressive_warmup_epochs', type=int, default=5, help='渐进式学习预热epoch数')
    parser.add_argument('--use_adaptive_loss_weighting', action='store_true', help='启用自适应损失权重')
    parser.add_argument('--use_cosine_warmup', action='store_true', help='启用带预热的余弦退火调度器')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='预热步数')
    parser.add_argument('--use_balanced_sampling', action='store_true', help='启用数据平衡采样')
    parser.add_argument('--num_quality_bins', type=int, default=5, help='质量分组数量')
    
    # AIGCIQA2023数据集分数范围参数
    parser.add_argument('--normalize_scores_to_five', action='store_true', 
                       help='将AIGCIQA2023数据集分数缩放到[0,5]范围进行训练')
    parser.add_argument('--restore_original_scores_for_metrics', action='store_true',
                       help='在计算评价指标时将预测分数还原到原始范围')

    # 交叉验证相关参数
    parser.add_argument('--use_cv', action='store_true',
                       help='启用交叉验证模式，使用预先划分的5折交叉验证数据')
    parser.add_argument('--fold', type=int, default=1, choices=range(1, 6),
                       help='指定使用的交叉验证fold编号 (1-5)')
    parser.add_argument('--cv_score_type', type=str, default='q',
                       choices=['q', 'a', 'c'],
                       help='AIGCIQA2023数据集的评分类型: q=quality, a=authenticity, c=correspondence')

    # ========== 消融实验控制参数 ==========
    # 控制变量法设计: full为基准，每次只移除一个模块
    parser.add_argument('--ablation_mode', type=str, default='full',
                       choices=['full', 'baseline', 'freq_encoder_only', 'freq_clip_fusion',
                                'no_freq_encoder', 'no_freq_clip_fusion', 'no_mfb'],
                       help='消融实验模式: '
                            'full=完整模型(CLIP+频域+MFB), '
                            'baseline=纯CLIP+简单融合, '
                            'freq_encoder_only=CLIP+频域编码器(无融合), '
                            'freq_clip_fusion=CLIP+频域融合(无MFB), '
                            'no_freq_encoder=移除频域编码器, '
                            'no_freq_clip_fusion=移除Freq-CLIP融合, '
                            'no_mfb=移除MFB双线性池化')

    parser.add_argument('--disable_freq_clip_fusion', action='store_true', default=False,
                       help='[消融] 禁用频域-CLIP自适应融合，直接使用CLIP特征')

    parser.add_argument('--disable_mfb_fusion', action='store_true', default=False,
                       help='[消融] 禁用MFB双线性池化，使用简单注意力融合')

    # 频域编码器内部消融（可选）
    parser.add_argument('--disable_y_channel_enhance', action='store_true', default=False,
                       help='[消融] 禁用Y通道高频增强')

    parser.add_argument('--disable_freq_stats', action='store_true', default=False,
                       help='[消融] 禁用频域统计特征')

    # 消融实验输出控制
    parser.add_argument('--no_save_model', action='store_true', default=False,
                       help='不保存模型权重（消融实验用）')
    parser.add_argument('--results_csv', type=str, default='ablation_results.csv',
                       help='消融实验结果CSV文件路径')
    parser.add_argument('--run_id', type=str, default='',
                       help='本次运行的标识，用于在共享结果CSV中区分不同实验/进程')

    return parser
