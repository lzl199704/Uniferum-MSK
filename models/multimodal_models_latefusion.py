from typing import List, Optional, Tuple, Union

import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, BertConfig, BertModel

from models import vision_models_v2
from models.loss_fcts import get_loss_function
import time 

def setup_llm(llm_config, lora_config=None):
    llm_module = getattr(transformers, llm_config["model_class"])
    model = llm_module.from_pretrained(
        llm_config["model_id"],
    )
    
    if llm_config.get("bert_encoder_layers", None):
        n_layers = llm_config["bert_encoder_layers"]
        model.encoder.layer = model.encoder.layer[:n_layers]

    if lora_config is not None:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)
    return model


def setup_vision(vision_config):
    model_module = getattr(vision_models_v2, vision_config["model_class"])
    model = model_module(vision_config)
    return model


def setup_late_fusion_vision(vision_config):
    """Setup dual vision encoders for late fusion"""
    model_module = getattr(vision_models_v2, vision_config["model_class"])
    
    # Create separate configs for PD and PDFS
    pd_config = vision_config.copy()
    pdfs_config = vision_config.copy()
    
    # Ensure single channel input for each encoder
    if "model_args" in pd_config:
        pd_config["model_args"] = pd_config["model_args"].copy()
        pd_config["model_args"]["in_chans"] = 1
        pdfs_config["model_args"] = pdfs_config["model_args"].copy()
        pdfs_config["model_args"]["in_chans"] = 1
    
    pd_model = model_module(pd_config)
    pdfs_model = model_module(pdfs_config)
    
    return pd_model, pdfs_model


class ConcatLayer(nn.Module):
    def forward(self, llm_embeds, vis_embeds, attention_mask, position_ids):
        mm_embeds = torch.cat([vis_embeds, llm_embeds], dim=1)
        batchsize, vseqlen = vis_embeds.size(0), vis_embeds.size(1)
        if attention_mask is not None:
            # check if to set attention_mask as ones if given None
            padding_ = torch.ones(
                (batchsize, vseqlen),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((padding_, attention_mask), dim=1)

        if position_ids is not None:
            vis_pos_ids = torch.arange(
                0, vseqlen, dtype=position_ids.dtype, device=position_ids.device
            )
            vis_pos_ids = vis_pos_ids.unsqueeze(0).expand(batchsize, -1)
            position_ids = torch.cat((vis_pos_ids, position_ids), dim=1)
        return mm_embeds, attention_mask, position_ids


def _pos_id_from_embeds(embeds):
    pos_ids = torch.arange(0, embeds.size(1), device=embeds.device, dtype=torch.long)
    return pos_ids.unsqueeze(0).expand(embeds.size(0), -1)


class LatentMemLayer(nn.Module):
    def __init__(self, config):
        super(LatentMemLayer, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config["d_model"], nhead=config["nhead"], batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config["num_layers"]
        )
        max_pos_ids = config.get("max_pos_ids", 576)
        self.lm_pos_embeds = nn.Embedding(max_pos_ids, config["d_model"])
        self.vis_pos_embeds = nn.Embedding(max_pos_ids, config["d_model"])
        self.vision_as_mem = config["vision_as_mem"]

    def forward(self, lm_embeds, vis_embeds, attention_mask, position_ids):
        lm_padding_mask = attention_mask == 0
        
        vis_pos_ids = _pos_id_from_embeds(vis_embeds)
        lm_embeds = lm_embeds + self.lm_pos_embeds(_pos_id_from_embeds(lm_embeds))
        vis_embeds = vis_embeds + self.vis_pos_embeds(vis_pos_ids)
        
        #print('vis_embeds shape', vis_embeds.shape, 'lm_embeds shape', lm_embeds.shape, 'mm embeds shape', mm_embeds)
        if self.vision_as_mem:
            mm_embeds = self.decoder(
                lm_embeds, vis_embeds, tgt_key_padding_mask=lm_padding_mask
            )
            #print('vis_embeds shape', vis_embeds.shape, 'lm_embeds shape', lm_embeds.shape, 'mm embeds shape', mm_embeds.shape)
            return mm_embeds, attention_mask, position_ids
        # language_as_mem
        # add attention as memory_mask
        mm_embeds = self.decoder(
            vis_embeds, lm_embeds, memory_key_padding_mask=lm_padding_mask
        )
        attention_mask = None
        return mm_embeds, attention_mask, vis_pos_ids


class LateFusionLayer(nn.Module):
    def __init__(self, config):
        super(LateFusionLayer, self).__init__()
        self.fusion_type = config.get("fusion_type", "attention")
        self.feature_dim = config.get("feature_dim", 768)
        
        if self.fusion_type == "attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.feature_dim, 
                num_heads=config.get("num_heads", 8), 
                batch_first=True
            )
            self.self_attention = nn.MultiheadAttention(
                embed_dim=self.feature_dim, 
                num_heads=config.get("num_heads", 8), 
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(self.feature_dim)
            self.norm2 = nn.LayerNorm(self.feature_dim)
            
        elif self.fusion_type == "concat":
            self.fusion_proj = nn.Linear(self.feature_dim * 2, self.feature_dim)
            
        elif self.fusion_type == "weighted_sum":
            self.pd_weight = nn.Parameter(torch.tensor(0.5))
            self.pdfs_weight = nn.Parameter(torch.tensor(0.5))
            
        elif self.fusion_type == "gated_fusion":
            self.gate_proj = nn.Sequential(
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.ReLU(),
                nn.Linear(self.feature_dim, 2),
                nn.Softmax(dim=-1)
            )

    def forward(self, pd_embeds, pdfs_embeds):
        """
        Args:
            pd_embeds: [batch, seq_len, feature_dim] 
            pdfs_embeds: [batch, seq_len, feature_dim]
        Returns:
            fused_embeds: [batch, seq_len, feature_dim]
        """
        if self.fusion_type == "attention":
            # Cross-attention: PD attends to PDFS
            pd_attended, _ = self.cross_attention(pd_embeds, pdfs_embeds, pdfs_embeds)
            pd_attended = self.norm1(pd_attended + pd_embeds)
            
            # Self-attention on the attended features
            fused_embeds, _ = self.self_attention(pd_attended, pd_attended, pd_attended)
            fused_embeds = self.norm2(fused_embeds + pd_attended)
            
        elif self.fusion_type == "concat":
            # Concatenate and project
            combined = torch.cat([pd_embeds, pdfs_embeds], dim=-1)
            fused_embeds = self.fusion_proj(combined)
            
        elif self.fusion_type == "weighted_sum":
            # Learned weighted sum
            weights = torch.softmax(torch.stack([self.pd_weight, self.pdfs_weight]), dim=0)
            fused_embeds = weights[0] * pd_embeds + weights[1] * pdfs_embeds
            
        elif self.fusion_type == "gated_fusion":
            # Gated fusion with learned gates
            combined = torch.cat([pd_embeds, pdfs_embeds], dim=-1)
            gates = self.gate_proj(combined)  # [batch, seq_len, 2]
            fused_embeds = gates[..., 0:1] * pd_embeds + gates[..., 1:2] * pdfs_embeds
            
        else:  # default: simple addition
            fused_embeds = pd_embeds + pdfs_embeds
            
        return fused_embeds


def setup_mixer(mixer_config):
    if mixer_config is None or mixer_config["name"] == "concatenate":
        return ConcatLayer()
    if mixer_config["name"] == "latent_mem":
        return LatentMemLayer(mixer_config)
    if mixer_config["name"] == "late_fusion":
        return LateFusionLayer(mixer_config)


class UniferumLateFusion(BertModel):
    config_class = BertConfig

    def __init__(self, config):
        # this is annoying, fixme
        llm_config = AutoConfig.from_pretrained(config["llm_args"]["model_id"])
        super().__init__(llm_config)
        self.config = llm_config
        self.setup_config = config
        self.model = setup_llm(config["llm_args"], lora_config=config.get("lora_args"))
        
        # Setup dual vision encoders for late fusion
        self.vision_model_pd, self.vision_model_pdfs = setup_late_fusion_vision(config["vision_args"])
        
        # Setup fusion mechanism
        self.fusion_layer = LateFusionLayer(config.get("fusion_args", {"fusion_type": "attention"}))
        self.mixer = setup_mixer(config.get("mixer_args"))
        
        self.vocab_size = llm_config.vocab_size
        self.lm_head_cls = nn.Linear(llm_config.hidden_size, 1)
        self.lm_head_seg = nn.Linear(llm_config.hidden_size, 64)
        self.loss_fct_cls = get_loss_function(
            config.get("loss_fct_cls", "BCEWithLogitsLoss"), reduction="none"
        )
        self.loss_fct_seg = get_loss_function(
            config.get("loss_fct_seg", "SigmoidFocalLoss"), reduction="none"
        )
        self.post_init()

    def get_multimodal_embeds(self, input_ids, position_ids, attention_mask, images):
        """Process dual images with late fusion"""
        if isinstance(images, (list, tuple)) and len(images) == 2:
            pd_images, pdfs_images = images
            
            # Encode each sequence separately
            pd_embeds = self.vision_model_pd(pd_images)  # [batch, seq_len, feature_dim]
            pdfs_embeds = self.vision_model_pdfs(pdfs_images)  # [batch, seq_len, feature_dim]
            
            # Late fusion
            vis_embeds = self.fusion_layer(pd_embeds, pdfs_embeds)
            
        else:
            # Fallback for single image (use PD encoder)
            vis_embeds = self.vision_model_pd(images)
        
        lm_embeds = self.model.embeddings(input_ids)  # [batch, seqlen1, fea_lm]
        
        v_batch = vis_embeds.size(0)
        lm_batch = lm_embeds.size(0)
        if v_batch == 1 and lm_batch > 1:
            vis_embeds = vis_embeds.expand(lm_batch, -1, -1, -1)

        # need to handle attention
        mm_embeds, attention_mask, position_ids = self.mixer(
            lm_embeds, vis_embeds, attention_mask, position_ids
        )

        # TODO improve position ids
        position_ids = _pos_id_from_embeds(mm_embeds)
        return None, position_ids, attention_mask, mm_embeds

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        seg_label=None,
        task_mask=None,
        **kwargs,
    ) -> Union[Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            input_ids,
            position_ids,
            attention_mask,
            inputs_embeds,
        ) = self.get_multimodal_embeds(
            input_ids,
            position_ids,
            attention_mask,
            images,
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = transformer_outputs.last_hidden_state
        logits_cls = self.lm_head_cls(last_hidden_state[:, 0])
        logits_seg = self.lm_head_seg(last_hidden_state[:, 1:65]) ##75  ###126 for EfficientNet 
        
        if labels is not None:
            cls_losses = self.loss_fct_cls(logits_cls, labels)
            seg_losses = self.loss_fct_seg(logits_seg, seg_label)
            losses = task_mask * cls_losses.mean(dim=1) + 10.0 * (
                1 - task_mask
            ) * seg_losses.mean(dim=2).mean(dim=1)
            loss = losses.mean()
           
        else:
            return logits_cls

        if not return_dict:  ## not tested
            output = (logits_cls,) + transformer_outputs
            return ((loss,) + output) if loss is not None else output
        return {"loss": loss, "logits": logits_cls, "logits_seg": logits_seg}