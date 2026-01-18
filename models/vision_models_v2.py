from typing import Sequence
import timm_3d
import torch
from torch import nn
import sys
sys.path.append("/raid/data/uniferum/uniferum-main/models/")

from nnssl.architectures.get_network_by_name import get_network_by_name
from dataclasses import asdict
from nnssl.architectures.get_network_from_plan import get_network_from_plans 
import json
from nnssl.adaptation_planning.adaptation_plan import DYN_ARCHITECTURE_PRESETS, AdaptationPlan

# Import MONAI for SwinUNETR
try:
    from monai.networks.nets import SwinUNETR
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not available. SwinUNETRViT will not work.")



# Helper to adapt first-conv-like weights when only input channel dimension differs
# Adapt the NNSSL conv
def adapt_input_conv_weights(src_tensor, tgt_shape):
    # src_tensor: tensor from checkpoint, tgt_shape: desired shape tuple
    src = src_tensor
    if src.ndim < 2:
        return src
    # If spatial/kernel dims match and out_channels match but in_channels differ, tile/copy
    if src.shape[0] == tgt_shape[0] and src.shape[2:] == tgt_shape[2:]:
        src_in = src.shape[1]
        tgt_in = tgt_shape[1]
        if src_in == tgt_in:
            return src
        # common case: src_in == 1 and tgt_in == 3 -> repeat across channel dim
        if src_in == 1:
            return src.repeat(1, tgt_in, *([1] * (src.ndim - 2))).contiguous()
        # if tgt_in is multiple of src_in, tile it
        if tgt_in % src_in == 0:
            reps = tgt_in // src_in
            return src.repeat(1, reps, *([1] * (src.ndim - 2))).contiguous()[:,:tgt_in,...]
        # fallback: average src channels then repeat
        avg = src.mean(dim=1, keepdim=True)
        return avg.repeat(1, tgt_in, *([1] * (src.ndim - 2))).contiguous()
    else:
        # shapes incompatible for simple adaptation, return None to indicate skipping
        return None
        
        
class EfficientNetViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = timm_3d.create_model(**config["model_args"]) 
        self.feature_dim = config["feature_dim"]  # checkout efficientnet specs
        self.llm_emb_dim = config["llm_emb_dim"]
        self.seqlen = config["seqlen"]
        self.emb_start = nn.Parameter(torch.randn(1, 1, self.llm_emb_dim))
        self.emb_end = nn.Parameter(torch.randn(1, 1, self.llm_emb_dim))
        self.lm_adaptor = nn.Sequential(
            #nn.TransformerEncoderLayer(
            #    d_model=self.feature_dim, nhead=config["nhead"], batch_first=True
            #), 
            nn.Linear(self.feature_dim, self.llm_emb_dim),
        )
        #self.lm_reduce = nn.Linear(512, self.seqlen - 2) ###1352 512

    def forward(self, imgs):
        """
        take imgs and output sequence of embeddings for llama
        imgs: (B, in_channels, H, W, D), (H, W, D) = image_sizes
        out: (B, seqlen, lm_dim)
        """
        batchsize = imgs.size(0)
        out = imgs
        out = self.model(out)
        out = out.view(batchsize, out.size(1), -1).swapaxes(1, 2)  # B, 8x8x8 , 1280
        out = self.lm_adaptor(out)  # B, 512, 768
        
        #out = self.lm_reduce(out.swapaxes(1, 2)).swapaxes(1, 2) # B, 256, 768
        
        emb_s = self.emb_start.expand(batchsize, -1, -1)
        emb_e = self.emb_end.expand(batchsize, -1, -1)
        out = torch.cat([emb_s, out, emb_e], 1)
        return out

class ResEncL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = config["feature_dim"]  # 320
        self.llm_emb_dim = config["llm_emb_dim"]  ##768
        self.net_plan = config['net_plan']
        self.pretrain_checkpoint= config['pretrain_checkpoint']
        self.emb_start = nn.Parameter(torch.randn(1, 1, self.llm_emb_dim))
        self.emb_end = nn.Parameter(torch.randn(1, 1, self.llm_emb_dim))
        self.lm_adaptor = nn.Sequential(
            #nn.TransformerEncoderLayer(
            #    d_model=self.feature_dim, nhead=config["nhead"], batch_first=True
            #), 
            nn.Linear(self.feature_dim, self.llm_emb_dim),
        )
        ### load NNSSL pretrain weight
        with open(self.net_plan, "r") as f:
            adapt_plan_dict = json.load(f)
        adapt_plan = AdaptationPlan.from_dict(adapt_plan_dict)
        arch_name = adapt_plan.architecture_plans.arch_class_name
        net3_rgb = get_network_by_name(
                adapt_plan,
                arch_name,
                3,
                1, ### now we handle pd, pd fs, t2 fs as 3 channels 
                deep_supervision=False
            )
        
        
        if self.pretrain_checkpoint is not None:    
            # 3) load checkpoint
            ckpt = torch.load(self.pretrain_checkpoint, map_location="cpu")
            state = ckpt.get("state_dict", ckpt.get("network_weights", ckpt))
            
            adapted = {}
            for k, tgt in net3_rgb.state_dict().items():
                if k in state and state[k].shape == tgt.shape:
                    adapted[k] = state[k]  # exact match
                elif k in state:
                    # try to adapt only if shapes differ in the input-channel dim (common for conv weights)
                    src = state[k]
                    new = adapt_input_conv_weights(src, tgt.shape)
                    if new is not None and new.shape == tgt.shape:
                        adapted[k] = new
                        print(f'Adapted {k}: {src.shape} -> {new.shape}')
                    else:
                        # skip loading this key so model's init remains for that layer
                        print(f"Skipping {k}: checkpoint {src.shape} does not fit target {tgt.shape}")
                else:
                    # key not in checkpoint -> leave uninitialized (model default)
                    pass
        
            missing, unexpected = net3_rgb.load_state_dict(adapted, strict=False)
        ### only use net3 encoder
        self.model = net3_rgb.encoder

    def forward(self, imgs):
        """
        take imgs and output sequence of embeddings for llama
        imgs: (B, in_channels, H, W, D), (H, W, D) = image_sizes
        out: (B, seqlen, lm_dim)
        """
        batchsize = imgs.size(0)
        out = imgs
        out = self.model(out)[-1]
        out = out.view(batchsize, out.size(1), -1).swapaxes(1, 2)  # B, 5x5x5 , 320
        out = self.lm_adaptor(out)  # B, 125, 768
        
        #out = self.lm_reduce(out.swapaxes(1, 2)).swapaxes(1, 2) # B, 256, 768
        
        emb_s = self.emb_start.expand(batchsize, -1, -1)
        emb_e = self.emb_end.expand(batchsize, -1, -1)
        out = torch.cat([emb_s, out, emb_e], 1)
        return out


class SwinUNETRViT(nn.Module):
    """
    Vision encoder using SwinUNETR from MONAI.
    Uses only the encoder part (swinViT) for feature extraction.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is required for SwinUNETRViT but not available")
        
        # Extract SwinUNETR configuration
        img_size = config.get("img_size", (160, 160, 160))
        in_channels = config.get("in_channels", 1)
        feature_size = config.get("feature_size", 48)
        use_checkpoint = config.get("use_checkpoint", True)
        
        # Create SwinUNETR model  
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=1,  # dummy, we only use encoder
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )
        
        # Extract only the encoder part (use only one reference to avoid shared tensors)
        self.model = self.swin_unetr.swinViT
        # Remove the full SwinUNETR reference to avoid shared tensor issues during saving
        del self.swin_unetr
        
        # Configuration for adapter layers
        self.feature_dim = config["feature_dim"]  # 768 (from SwinUNETR last layer)
        self.llm_emb_dim = config["llm_emb_dim"]  # target embedding dim (768)
        self.seqlen = config["seqlen"]  # sequence length (5x5x5 = 125 + 2 = 127)
        
        # Start and end embeddings
        self.emb_start = nn.Parameter(torch.randn(1, 1, self.llm_emb_dim))
        self.emb_end = nn.Parameter(torch.randn(1, 1, self.llm_emb_dim))
        
        # Adapter to convert SwinUNETR features to LLM embedding dimension
        self.lm_adaptor = nn.Sequential(
            nn.Linear(self.feature_dim, self.llm_emb_dim),
        )

    def forward(self, imgs):
        """
        Take imgs and output sequence of embeddings for LLM
        imgs: (B, in_channels, H, W, D), (H, W, D) = image_sizes
        out: (B, seqlen, lm_dim)
        
        Expected input: (B, 1, 160, 160, 160)
        Expected output: (B, 127, 768) where 127 = 125 spatial + 2 special tokens
        """
        batchsize = imgs.size(0)
        
        # Forward through SwinUNETR encoder
        # The swinViT encoder returns a list of features, we need the last one
        out_list = self.model(imgs)
        out = out_list[-1]  # Select last element with shape (B, 768, 5, 5, 5)
        
        # Reshape from (B, 768, 5, 5, 5) to (B, 125, 768)
        out = out.view(batchsize, out.size(1), -1).permute(0, 2, 1)  # B, 125, 768
        
        # Apply adapter to ensure correct embedding dimension
        out = self.lm_adaptor(out)  # B, 125, llm_emb_dim
        
        # Add start and end embeddings
        emb_s = self.emb_start.expand(batchsize, -1, -1)
        emb_e = self.emb_end.expand(batchsize, -1, -1)
        out = torch.cat([emb_s, out, emb_e], 1)  # B, 127, llm_emb_dim
        
        return out


class SwinUNETRTinyViT(nn.Module):
    """
    Memory-efficient SwinUNETR encoder with reduced parameters
    Significantly smaller than full SwinUNETR while maintaining transformer architecture
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is required for SwinUNETRTinyViT. Please install monai.")
        
        # Get configuration parameters
        img_size = config.get("img_size", [96, 96, 96])  # Smaller default
        in_channels = config.get("in_channels", 1)
        feature_size = config.get("feature_size", 24)  # Much smaller
        use_checkpoint = config.get("use_checkpoint", True)
        
        # Create SwinUNETR model with reduced parameters
        self.full_model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=1,  # Dummy output channels
            feature_size=feature_size,  # Reduced feature size
            use_checkpoint=use_checkpoint,
            depths=[2, 2, 2, 2],  # Reduced depth (default is [2, 2, 6, 2])
            num_heads=[3, 6, 12, 24],  # Reduced heads
        )
        
        # Extract only the encoder part
        self.model = self.full_model.swinViT
        
        # Configuration for output processing
        self.feature_dim = config["feature_dim"]  # Should be 384 for feature_size=24
        self.llm_emb_dim = config["llm_emb_dim"]
        self.seqlen = config["seqlen"]  # Adjust based on output size
        
        # Start and end embeddings
        self.emb_start = nn.Parameter(torch.randn(1, 1, self.llm_emb_dim))
        self.emb_end = nn.Parameter(torch.randn(1, 1, self.llm_emb_dim))
        
        # Adapter to convert SwinUNETR features to LLM embedding dimension
        self.lm_adaptor = nn.Sequential(
            nn.Linear(self.feature_dim, self.llm_emb_dim),
        )

    def forward(self, imgs):
        """
        Take imgs and output sequence of embeddings for LLM
        Expected output size will be smaller due to reduced img_size
        """
        batchsize = imgs.size(0)
        
        # Forward through SwinUNETR encoder
        out_list = self.model(imgs)
        out = out_list[-1]  # Select last element
        
        # Reshape spatial dimensions to sequence
        out = out.view(batchsize, out.size(1), -1).permute(0, 2, 1)
        
        # Apply adapter
        out = self.lm_adaptor(out)
        
        # Add start and end embeddings
        emb_s = self.emb_start.expand(batchsize, -1, -1)
        emb_e = self.emb_end.expand(batchsize, -1, -1)
        out = torch.cat([emb_s, out, emb_e], 1)
        
        return out