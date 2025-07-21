
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTForImageClassification, ViTConfig
import timm
from typing import Optional, Union, Dict, Any
import copy
import warnings


class AlphaVisionTransformer(nn.Module):
    """
    基于原始CLIP代码风格的Alpha通道Vision Transformer
    采用双卷积层 + 直接相加融合的设计
    """
    def __init__(
        self, 
        input_resolution: int = 224, 
        patch_size: int = 16, 
        width: int = 768, 
        layers: int = 12, 
        heads: int = 12, 
        output_dim: int = 768,
        num_classes: int = 1000,
        dropout: float = 0.0,
        lora_adapt: bool = False, 
        rank: int = 16
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # RGB通道的patch embedding (标准3通道)
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=width, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=False
        )
        
        # Alpha通道的patch embedding (1通道)
        self.conv1_alpha = nn.Conv2d(
            in_channels=1, 
            out_channels=width, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)

        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=width * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=False,
                norm_first=True
            ),
            num_layers=layers
        )

        self.ln_post = nn.LayerNorm(width)
        
        # 输出投影层
        if output_dim != width:
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        else:
            self.proj = None
            
        # 分类头 (如果需要)
        if num_classes > 0:
            self.head = nn.Linear(output_dim if output_dim != width else width, num_classes)
        else:
            self.head = None

        # 初始化Alpha通道权重为零 (关键设计)
        self._init_alpha_weights()

    def _init_alpha_weights(self):
        """初始化Alpha通道权重为零，确保未训练时不影响RGB性能"""
        with torch.no_grad():
            nn.init.zeros_(self.conv1_alpha.weight)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor = None, return_features: bool = False):
        """
        前向传播
        
        Args:
            x: [B, 3, H, W] RGB图像
            alpha: [B, 1, H, W] Alpha通道 (必须提供)
            return_features: 是否返回特征而非分类结果
            
        Returns:
            分类logits 或 特征向量
        """
        assert alpha is not None, "Alpha channel is required!"
        
        # Patch embedding: RGB + Alpha特征直接相加
        x = self.conv1(x)  # [B, width, grid, grid]
        x = x + self.conv1_alpha(alpha)  # 关键：直接相加融合
        
        # 转换为序列格式
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid^2]
        x = x.permute(0, 2, 1)  # [B, grid^2, width]
        
        # 添加CLS token
        cls_tokens = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tokens, x], dim=1)  # [B, grid^2+1, width]
        
        # 添加位置编码
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # Transformer编码 (注意：PyTorch Transformer期望 [seq_len, batch, embed_dim])
        x = x.permute(1, 0, 2)  # [grid^2+1, B, width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, grid^2+1, width]

        # 后处理
        x = self.ln_post(x[:, 0, :])  # 使用CLS token

        # 输出投影
        if self.proj is not None:
            x = x @ self.proj
            
        # 返回特征或分类结果
        if return_features or self.head is None:
            return x
        else:
            return self.head(x)


def load_pretrained_alpha_vit(
    pretrained_model_name: str = "google/vit-base-patch16-224",
    num_classes: Optional[int] = None,
    alpha_vision_ckpt_pth: str = "None",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    use_timm: bool = False,
    lora_adapt: bool = False,
    rank: int = 16
):
    """
    加载预训练ViT并扩展为Alpha通道支持版本
    
    Args:
        pretrained_model_name: 预训练模型名称
        num_classes: 分类类别数
        alpha_vision_ckpt_pth: Alpha视觉编码器检查点路径
        device: 设备
        use_timm: 是否使用timm库
        lora_adapt: 是否使用LoRA适配
        rank: LoRA rank
        
    Returns:
        (model, preprocess_fn): Alpha ViT模型和预处理函数
    """
    
    if use_timm:
        return _load_timm_alpha_vit(
            pretrained_model_name, num_classes, alpha_vision_ckpt_pth, 
            device, lora_adapt, rank
        )
    else:
        return _load_transformers_alpha_vit(
            pretrained_model_name, num_classes, alpha_vision_ckpt_pth, 
            device, lora_adapt, rank
        )


def _load_transformers_alpha_vit(
    model_name: str, num_classes: Optional[int], alpha_ckpt_pth: str,
    device: Union[str, torch.device], lora_adapt: bool, rank: int
):
    """使用HuggingFace transformers加载预训练模型"""
    
    # 加载预训练ViT
    if num_classes and num_classes > 0:
        pretrained_model = ViTForImageClassification.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        config = pretrained_model.config
        vit_model = pretrained_model.vit
    else:
        pretrained_model = ViTModel.from_pretrained(model_name)
        config = pretrained_model.config
        vit_model = pretrained_model
    
    # 创建Alpha ViT模型
    alpha_model = AlphaVisionTransformer(
        input_resolution=config.image_size,
        patch_size=config.patch_size,
        width=config.hidden_size,
        layers=config.num_hidden_layers,
        heads=config.num_attention_heads,
        output_dim=config.hidden_size,
        num_classes=num_classes or 0,
        dropout=config.hidden_dropout_prob,
        lora_adapt=lora_adapt,
        rank=rank
    )
    
    # 转换并加载权重
    state_dict = _convert_transformers_weights(vit_model.state_dict(), config)
    
    # 添加Alpha权重 (零初始化)
    if 'conv1_alpha.weight' not in state_dict:
        rgb_weight = state_dict['conv1.weight']
        alpha_weight = torch.zeros_like(rgb_weight)[:, 0:1, :, :]  # 只保留1个通道
        state_dict['conv1_alpha.weight'] = alpha_weight
    
    # 加载权重
    alpha_model.load_state_dict(state_dict, strict=False)
    
    # 如果有独立的Alpha检查点，加载它
    if alpha_ckpt_pth != "None":
        alpha_checkpoint = torch.load(alpha_ckpt_pth, map_location="cpu")
        alpha_model.load_state_dict(alpha_checkpoint, strict=False)
        alpha_model.eval()  # 合并LoRA参数 (如果存在)
    
    alpha_model = alpha_model.to(device)
    
    # 创建预处理函数
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return alpha_model, preprocess


def _load_timm_alpha_vit(
    model_name: str, num_classes: Optional[int], alpha_ckpt_pth: str,
    device: Union[str, torch.device], lora_adapt: bool, rank: int
):
    """使用timm加载预训练模型"""
    try:
        import timm
    except ImportError:
        raise ImportError("Please install timm: pip install timm")
    
    # 加载预训练模型
    pretrained_model = timm.create_model(model_name, pretrained=True, num_classes=0)
    
    # 获取配置信息
    img_size = pretrained_model.default_cfg.get('input_size', [3, 224, 224])[-1]
    patch_size = getattr(pretrained_model.patch_embed, 'patch_size', [16, 16])[0]
    embed_dim = pretrained_model.embed_dim
    num_layers = len(pretrained_model.blocks)
    num_heads = pretrained_model.blocks[0].attn.num_heads
    
    # 创建Alpha ViT模型
    alpha_model = AlphaVisionTransformer(
        input_resolution=img_size,
        patch_size=patch_size,
        width=embed_dim,
        layers=num_layers,
        heads=num_heads,
        output_dim=embed_dim,
        num_classes=num_classes or 0,
        lora_adapt=lora_adapt,
        rank=rank
    )
    
    # 转换权重
    state_dict = _convert_timm_weights(pretrained_model.state_dict())
    
    # 添加Alpha权重
    if 'conv1_alpha.weight' not in state_dict:
        rgb_weight = state_dict['conv1.weight']
        alpha_weight = torch.zeros_like(rgb_weight)[:, 0:1, :, :]
        state_dict['conv1_alpha.weight'] = alpha_weight
    
    alpha_model.load_state_dict(state_dict, strict=False)
    
    # 加载Alpha检查点
    if alpha_ckpt_pth != "None":
        alpha_checkpoint = torch.load(alpha_ckpt_pth, map_location="cpu")
        alpha_model.load_state_dict(alpha_checkpoint, strict=False)
        alpha_model.eval()
    
    alpha_model = alpha_model.to(device)
    
    # 预处理函数
    preprocess = timm.data.create_transform(**pretrained_model.default_cfg)
    
    return alpha_model, preprocess


def _convert_transformers_weights(state_dict: Dict[str, torch.Tensor], config) -> Dict[str, torch.Tensor]:
    """转换HuggingFace权重格式"""
    new_state_dict = {}
    
    # 映射权重名称
    weight_mapping = {
        'embeddings.patch_embeddings.projection.weight': 'conv1.weight',
        'embeddings.cls_token': 'class_embedding',
        'embeddings.position_embeddings': 'positional_embedding',
        'layernorm.weight': 'ln_post.weight',
        'layernorm.bias': 'ln_post.bias',
    }
    
    for old_name, new_name in weight_mapping.items():
        if old_name in state_dict:
            new_state_dict[new_name] = state_dict[old_name]
    
    # 转换Transformer层
    for i in range(config.num_hidden_layers):
        layer_mapping = {
            f'encoder.layer.{i}.layernorm_before.weight': f'transformer.layers.{i}.norm1.weight',
            f'encoder.layer.{i}.layernorm_before.bias': f'transformer.layers.{i}.norm1.bias',
            f'encoder.layer.{i}.attention.attention.query.weight': f'transformer.layers.{i}.self_attn.in_proj_weight',
            f'encoder.layer.{i}.attention.attention.query.bias': f'transformer.layers.{i}.self_attn.in_proj_bias',
            # ... 其他层的映射
        }
        
        for old_name, new_name in layer_mapping.items():
            if old_name in state_dict:
                new_state_dict[new_name] = state_dict[old_name]
    
    return new_state_dict


def _convert_timm_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """转换timm权重格式"""
    new_state_dict = {}
    
    weight_mapping = {
        'patch_embed.proj.weight': 'conv1.weight',
        'cls_token': 'class_embedding',
        'pos_embed': 'positional_embedding',
        'norm.weight': 'ln_post.weight',
        'norm.bias': 'ln_post.bias',
    }
    
    for old_name, new_name in weight_mapping.items():
        if old_name in state_dict:
            tensor = state_dict[old_name]
            if old_name == 'pos_embed':
                # 去掉batch维度
                tensor = tensor.squeeze(0)
            elif old_name == 'cls_token':
                tensor = tensor.squeeze(0).squeeze(0)
            new_state_dict[new_name] = tensor
    
    return new_state_dict


def available_models():
    """返回可用的预训练模型列表"""
    return {
        "transformers": [
            "google/vit-base-patch16-224",
            "google/vit-large-patch16-224",
            "google/vit-base-patch32-224",
            "facebook/deit-base-patch16-224",
        ],
        "timm": [
            "vit_base_patch16_224",
            "vit_large_patch16_224",
            "vit_small_patch16_224",
            "deit_base_patch16_224",
        ]
    }


# 使用示例
if __name__ == "__main__":
    print("=== Alpha ViT模型测试 ===")
    
    # 方式1: 从头创建
    print("\n1. 从头创建Alpha ViT:")
    model = AlphaVisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        num_classes=1000
    )
    
    # 测试输入
    rgb = torch.randn(2, 3, 224, 224)
    alpha = torch.randn(2, 1, 224, 224)
    
    with torch.no_grad():
        output = model(rgb, alpha)
        print(f"输出形状: {output.shape}")
    
    # 方式2: 基于预训练模型
    print("\n2. 基于预训练模型:")
    try:
        pretrained_model, preprocess = load_pretrained_alpha_vit(
            pretrained_model_name="google/vit-base-patch16-224",
            num_classes=1000,
            use_timm=False
        )
        
        with torch.no_grad():
            output = pretrained_model(rgb, alpha)
            print(f"预训练模型输出形状: {output.shape}")
            
    except Exception as e:
        print(f"预训练模型加载失败: {e}")
        print("请确保安装了transformers: pip install transformers")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    alpha_params = sum(p.numel() for p in model.parameters() if 'alpha' in str(p))
    print(f"\n总参数量: {total_params:,}")
    print(f"Alpha相关参数: {model.conv1_alpha.weight.numel():,}")
    
    # 显示可用模型
    print(f"\n可用的预训练模型:")
    models = available_models()
    print("HuggingFace transformers:", models["transformers"])
    print("timm:", models["timm"])