from dataclasses import dataclass
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
from skimage.measure import block_reduce
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import time

def patchify_3d(volume, patch_size):
    D, H, W = volume.shape
    pD, pH, pW = patch_size

    assert (
        D % pD == 0 and H % pH == 0 and W % pW == 0
    ), "Volume must be divisible by patch size"

    nD, nH, nW = D // pD, H // pH, W // pW

    # Reshape and reorder axes to gather patches
    patches = volume.reshape(nD, pD, nH, pH, nW, pW)
    patches = patches.transpose(0, 2, 4, 1, 3, 5)  # → (nD, nH, nW, pD, pH, pW)
    patches = patches.reshape(-1, pD * pH * pW)  # → (N, patch_volume)

    return patches


def unpatchify_3d(patches, volume_shape, patch_size):
    D, H, W = volume_shape
    pD, pH, pW = patch_size

    assert (
        D % pD == 0 and H % pH == 0 and W % pW == 0
    ), "Volume must be divisible by patch size"

    nD, nH, nW = D // pD, H // pH, W // pW
    N = nD * nH * nW
    assert patches.shape == (
        N,
        pD * pH * pW,
    ), "Patches shape doesn't match expected size"

    # Reshape to 6D
    patches = patches.reshape(nD, nH, nW, pD, pH, pW)
    patches = patches.transpose(0, 3, 1, 4, 2, 5)  # → (nD, pD, nH, pH, nW, pW)
    volume = patches.reshape(D, H, W)

    return volume


@dataclass
class VQABinaryDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer
    with_label: bool

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        result = VQAMaskDatasetLateFusion._get_inputs_from_batch(
            batch=batch,
            tokenizer=self.tokenizer,
            with_label=self.with_label,
        )
        return result


class VQAMaskDatasetLateFusion(Dataset):
    def __init__(
        self,
        text_list: List[str],
        img_files: List[str],
        img_files_pdfs: List[str],  # PDFS sequence file paths
        seg_files: List[str],
        tokenizer: PreTrainedTokenizer,
        labels: Optional[List[int]] = None,
        transform=None,
        transform_imgonly=None,
        normalization_strategy: str = "joint",  # "joint", "independent", "percentile", "z_score_joint"
    ):

        if img_files is not None:
            assert len(text_list) == len(img_files)
            self.img_files = img_files
        if img_files_pdfs is not None:
            assert len(text_list) == len(img_files_pdfs)
            self.img_files_pdfs = img_files_pdfs
        if seg_files is not None:
            assert len(text_list) == len(seg_files)
            self.seg_files = seg_files
        if labels is not None:
            assert len(text_list) == len(labels)
        self.labels = labels
        self.text_list = text_list

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.transform = transform
        self.transform_imgonly = transform_imgonly
        self.mask_pool_size = (8, 8, 8)
        self.patch_size = (4, 4, 4)
        self.normalization_strategy = normalization_strategy

    def __len__(self):
        return len(self.text_list)

    def read_img(self, img_file):
        if img_file.endswith((".nii", ".nii.gz")):
            img = nib.load(img_file).get_fdata(dtype=np.float32)
            return img
        elif img_file.endswith(".npz"):
            img = np.load(img_file)['volume'].astype(np.float32)
            return img
        else:
            raise ValueError(
                f"Unsupported file type: {img_file}. Only .nii, .nii.gz, and .bin are supported."
            )

    def normalize_dual_images(self, img_pd, img_pdfs, strategy="joint"):
        """
        Improved normalization strategies that preserve relationships between sequences
        """
        if strategy == "joint":
            # Joint min-max normalization - preserves intensity relationships
            combined = np.stack([img_pd, img_pdfs], axis=0)
            global_min, global_max = combined.min(), combined.max()
            if global_max > global_min:
                img_pd = (img_pd - global_min) / (global_max - global_min)
                img_pdfs = (img_pdfs - global_min) / (global_max - global_min)
            else:
                img_pd = img_pd * 0.0
                img_pdfs = img_pdfs * 0.0
                
        elif strategy == "percentile":
            # Percentile-based joint normalization (more robust to outliers)
            combined = np.stack([img_pd, img_pdfs], axis=0)
            p1, p99 = np.percentile(combined, [1, 99])
            if p99 > p1:
                img_pd = np.clip((img_pd - p1) / (p99 - p1), 0, 1)
                img_pdfs = np.clip((img_pdfs - p1) / (p99 - p1), 0, 1)
            else:
                img_pd = np.clip(img_pd, 0, 1)
                img_pdfs = np.clip(img_pdfs, 0, 1)
                
        elif strategy == "z_score_joint":
            # Joint z-score normalization
            combined = np.stack([img_pd, img_pdfs], axis=0)
            mean, std = combined.mean(), combined.std()
            if std > 0:
                img_pd = (img_pd - mean) / std
                img_pdfs = (img_pdfs - mean) / std
                
                # Rescale to [0,1] range
                combined_norm = np.stack([img_pd, img_pdfs], axis=0)
                min_val, max_val = combined_norm.min(), combined_norm.max()
                if max_val > min_val:
                    img_pd = (img_pd - min_val) / (max_val - min_val)
                    img_pdfs = (img_pdfs - min_val) / (max_val - min_val)
                    
        else:  # independent (original method)
            pd_min, pd_max = img_pd.min(), img_pd.max()
            pdfs_min, pdfs_max = img_pdfs.min(), img_pdfs.max()
            
            if pd_max > pd_min:
                img_pd = (img_pd - pd_min) / (pd_max - pd_min)
            else:
                img_pd = img_pd * 0.0
                
            if pdfs_max > pdfs_min:
                img_pdfs = (img_pdfs - pdfs_min) / (pdfs_max - pdfs_min)
            else:
                img_pdfs = img_pdfs * 0.0
        
        return img_pd, img_pdfs

    def get_dual_img(self, img_file_pd: str, img_file_pdfs: str, transform=None):
        """Load and process both PD and PDFS sequences with improved normalization"""
        # Load PD sequence
        img_pd = self.read_img(img_file_pd)
        # Load PDFS sequence  
        img_pdfs = self.read_img(img_file_pdfs)
        
        # Apply improved normalization
        img_pd, img_pdfs = self.normalize_dual_images(
            img_pd, img_pdfs, self.normalization_strategy
        )
        
        # Convert to appropriate data type
        img_pd = np.float16(img_pd)
        img_pdfs = np.float16(img_pdfs)
        
        if transform is not None:
            img_pd = np.expand_dims(img_pd, 0)
            img_pdfs = np.expand_dims(img_pdfs, 0)
            img_pd = transform(img_pd)
            img_pdfs = transform(img_pdfs)
            return [img_pd, img_pdfs]  # Return as list for late fusion
        
        img_pd = torch.tensor(img_pd).unsqueeze(0)
        img_pdfs = torch.tensor(img_pdfs).unsqueeze(0)
        return [img_pd, img_pdfs]  # Return as list for late fusion

    def __getitem__(self, idx: int):
        text = self.text_list[idx]

        # TODO: simplify

        if self.img_files is not None :
            # Use dual-channel input for PD and PDFS sequences
            img = self.get_dual_img(
                self.img_files[idx], 
                self.img_files_pdfs[idx], 
                transform=self.transform
            )
            if self.transform_imgonly is not None:
                img = [self.transform_imgonly(img[0]), self.transform_imgonly(img[1])]
        if self.labels is not None:
            # Use the first image (PD) to determine dimensions for segmentation labels
            if isinstance(img, list):
                _, H, W, D = img[0].shape
            else:
                _, H, W, D = img.shape
            m1 = np.prod(self.mask_pool_size)
            m2 = np.prod(self.patch_size)
            label_size = H * D * W // m1 // m2
            patchify_label = np.zeros((label_size , m2), dtype="float16")
            task_mask = 1
        content = {"text": text, "image": img}
        if self.labels is not None:
            content["label"] = self.labels[idx]
            content["seg_label"] = patchify_label
            content["task_mask"] = task_mask
        return content

    def collate_fn(self, batch: List[Dict]):
        return self._get_inputs_from_batch(
            batch=batch,
            tokenizer=self.tokenizer,
            with_label=(self.labels is not None),
            load_image=(self.img_files is not None),
        )

    @staticmethod
    def _get_inputs_from_batch(
        batch: List[Dict[str, torch.Tensor]],
        tokenizer: PreTrainedTokenizer,
        with_label: bool = True,
        load_image: bool = True,
    ):
        sentences = [item["text"] for item in batch]

        inputs = tokenizer(sentences, padding=True, return_tensors="pt")

        if load_image:
            images = [item["image"] for item in batch]
            # Handle dual-channel images - keep as separate lists for late fusion
            if isinstance(images[0], list) and len(images[0]) == 2:
                # Separate PD and PDFS images
                pd_images = [img[0] for img in images]
                pdfs_images = [img[1] for img in images]
                inputs["images"] = [torch.stack(pd_images), torch.stack(pdfs_images)]
            else:
                inputs["images"] = torch.stack(images)

        if with_label:
            labels = np.array([item["label"] for item in batch], dtype=np.float16)
            inputs["labels"] = torch.from_numpy(labels[:, None])

            seg_labels = np.stack([item["seg_label"] for item in batch]).astype(
                np.float16
            )
            inputs["seg_label"] = torch.from_numpy(seg_labels)

            task_masks = np.array(
                [item["task_mask"] for item in batch], dtype=np.float16
            )
            inputs["task_mask"] = torch.from_numpy(task_masks)
        return inputs
