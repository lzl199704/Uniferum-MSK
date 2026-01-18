import os

import pandas as pd
from transformers import AutoTokenizer, Trainer, TrainingArguments

from bin.utils import (
    ConfigFileArgs,
    create_transform,
    load_checkpoint_from_config_if_available,
    load_yaml,
    save_yaml,
)
from data_utils.vqa_dataset_sportsmed_knee_latefusion import VQABinaryDataCollator, VQAMaskDatasetLateFusion
from models.multimodal_models_latefusion import UniferumLateFusion
import torch

def create_datasets(
    input_file: str,
    tokenizer,
    train_transform=None,
    predict_transform=None,
    train_transform_imgonly=None,
    normalization_strategy="joint",
):
    df_data = pd.read_parquet(input_file)
    df_train = df_data[df_data.split == "train"]
    df_val = df_data[df_data.split == "val"]

    # Create datasets with dual image inputs (PD and PDFS) using late fusion
    train_dataset = VQAMaskDatasetLateFusion(
        df_train.question.to_list(),
        df_train.img_file_path_pd.to_list(),  # PD sequence paths
        df_train.img_file_path_pdfs.to_list(),  # PDFS sequence paths
        None,  # seg_files
        tokenizer,
        df_train.label.to_list(),
        transform=train_transform,
        transform_imgonly=train_transform_imgonly,
        normalization_strategy=normalization_strategy,
    )

    eval_dataset = VQAMaskDatasetLateFusion(
        df_val.question.to_list(),
        df_val.img_file_path_pd.to_list(),  # PD sequence paths
        df_val.img_file_path_pdfs.to_list(),  # PDFS sequence paths
        None,  # seg_files
        tokenizer,
        df_val.label.to_list(),
        transform=predict_transform,
        normalization_strategy=normalization_strategy,
    )

    collector = VQABinaryDataCollator(train_dataset.tokenizer, True)
    return train_dataset, eval_dataset, collector


def create_and_prepare_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config["llm_args"]["model_id"])
    
    # Ensure vision model is configured for single-channel input (late fusion)
    if "vision_args" in config and "model_args" in config["vision_args"]:
        config["vision_args"]["model_args"]["in_chans"] = 1
        print(f"Updated vision model to use in_chans=1 for late fusion architecture")
    
    # Add default fusion configuration if not specified
    if "fusion_args" not in config:
        config["fusion_args"] = {
            "fusion_type": "attention",  # Options: "attention", "concat", "weighted_sum", "gated_fusion"
            "feature_dim": config["vision_args"].get("llm_emb_dim", 768),
            "num_heads": 8
        }
        print(f"Added default fusion configuration: {config['fusion_args']}")
    
    model = UniferumLateFusion(config)
    model = load_checkpoint_from_config_if_available(model, config)

    # Freeze LLM by default unless explicitly set to False
    freeze_llm = config.get("freeze_llm", True)
    freeze_llm = freeze_llm if isinstance(freeze_llm, bool) else True
    if "lora_args" not in config and freeze_llm:
        print("freezing llm")
        for param in model.model.parameters():
            param.requires_grad = False

    # Ensure task heads are trainable
    for param in model.lm_head_cls.parameters():
        param.requires_grad = True
    for param in model.lm_head_seg.parameters():
        param.requires_grad = True
    for param in model.model.pooler.parameters():
        param.requires_grad = True
    
    # Ensure vision encoders and fusion layer are trainable
    for param in model.vision_model_pd.parameters():
        param.requires_grad = True
    for param in model.vision_model_pdfs.parameters():
        param.requires_grad = True
    for param in model.fusion_layer.parameters():
        param.requires_grad = True
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    return model, tokenizer


def main():
    config_file = ConfigFileArgs().parse_args().config_file
    config = load_yaml(config_file)
    if "logging_dir" not in config["hf_TrainingArguments"]:
        out_dir = config["hf_TrainingArguments"].get("output_dir", "./")
        config["hf_TrainingArguments"]["logging_dir"] = os.path.join(out_dir, "logs")

    file_path = os.path.join(
        config["hf_TrainingArguments"]["output_dir"], "train_config.yaml"
    )
    save_yaml(config, file_path)

    # Get normalization strategy from config
    normalization_strategy = config.get("normalization_strategy", "joint")
    print(f"Using normalization strategy: {normalization_strategy}")

    train_transform = None
    if config.get("train_transform") is not None:
        train_transform = create_transform(config["train_transform"])
    train_transform_imgonly = None
    if config.get("train_transform_imgonly") is not None:
        train_transform_imgonly = create_transform(config["train_transform_imgonly"])
    predict_transform = None
    if config.get("predict_transform") is not None:
        predict_transform = create_transform(config["predict_transform"])

    model, tokenizer = create_and_prepare_model(config)
    
    # Handle state_dict contiguity issue
    orig_sd = model.state_dict
    def contiguous_state_dict(*args, **kwargs):
        sd = orig_sd(*args, **kwargs)
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and not v.is_contiguous():
                sd[k] = v.contiguous()
        return sd
    model.state_dict = contiguous_state_dict
 
    train_dataset, eval_dataset, collector = create_datasets(
        config["input_file"],
        tokenizer,
        train_transform=train_transform,
        predict_transform=predict_transform,
        train_transform_imgonly=train_transform_imgonly,
        normalization_strategy=normalization_strategy,
    )

    # Set training arguments
    training_args = TrainingArguments(**config["hf_TrainingArguments"])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collector,
    )

    print("Starting training with late fusion architecture...")
    trainer.train() #resume_from_checkpoint=True


if __name__ == "__main__":
    main()
