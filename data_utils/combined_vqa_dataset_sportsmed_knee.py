from dataclasses import dataclass
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
from skimage.measure import block_reduce
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from data_utils.anatomical_map import anatomical_map
from img_utils.zst_utils import load_zst16
from img_utils.img_process import process_image
import time

def patchify_3d(volume, patch_size): ###[256,256,32] -> [32,32,4] -> [8,8,1]
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
        #start = time.time()
        result = VQAMaskDataset._get_inputs_from_batch(
            batch=batch,
            tokenizer=self.tokenizer,
            with_label=self.with_label,
        )
        #end = time.time()
        #print(f"Data collator time: {end - start:.4f}s")
        return result


class VQAMaskDataset(Dataset):
    def __init__(
        self,
        text_list: List[str],
        img_files: List[str],
        seg_files: List[str],
        tokenizer: PreTrainedTokenizer,
        labels: Optional[List[int]] = None,
        transform=None,
        transform_imgonly=None,
    ):

        if img_files is not None:
            assert len(text_list) == len(img_files)
            self.img_files = img_files
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
        self.tsseg_map = anatomical_map
        self.mask_pool_size = (8, 8, 8)
        self.patch_size = (4, 4, 4)

    def __len__(self):
        return len(self.text_list)

    def read_img(self, img_file):
        if img_file.endswith((".nii", ".nii.gz")):
            img = nib.load(img_file).get_fdata(dtype=np.float32)
        elif img_file.endswith(".bin"):
            img = load_zst16(img_file).astype("float32")
        elif img_file.endswith(".npz"):
            img = np.load(img_file)['volume'].astype(np.float32)
            return img
        else:
            raise ValueError(
                f"Unsupported file type: {img_file}. Only .nii, .nii.gz, and .bin are supported."
            )
        img = img / 2000.0 + 0.5
        return img

    def get_img(self, img_file: str,seg_file: str, transform: None):
        img = self.read_img(img_file)
        
        
        img = process_image(img,None, False)
        
        img = np.float16(img) / 255.0 
        
        if transform is not None:
            img = np.expand_dims(img, 0)
            return transform(img)
        img = torch.tensor(img).unsqueeze(0)
        return img

    def get_img_mask(self, img_file: str, seg_file: str, transform: None):
        img = self.read_img(img_file)
        if seg_file.endswith((".nii", ".nii.gz")):
            seg = nib.load(seg_file).get_fdata(dtype=np.float32).astype("int")
        elif seg_file.endswith(".npz"):
            seg = np.load(seg_file)['volume'].astype("uint8")
        
        img = (img - img.min()) / (img.max()-img.min())
        
        img = np.float16(img) 
        
        ### seg_indices should be []  "femur": 1,     "patella": 2,   "fibula": 3,    "tibia": 4, 
        masks = []
        seg_indices = [1,2,3,4]
        for id11 in seg_indices:
            mask = np.isin(seg, [id11]).astype("float16")
            masks.append(mask)
        #mask = np.isin(seg, seg_indices).astype("float16")

        if transform is not None:
            img_mask = np.stack([img, masks[0],masks[1],masks[2],masks[3]], 0)
            img_mask = transform(img_mask)
            return img_mask[[0]], [img_mask[[1]], img_mask[[2]], img_mask[[3]], img_mask[[4]]]

        img = torch.tensor(img).unsqueeze(0)  # [1, 256, 256, 128]
        masks = [torch.tensor(m).unsqueeze(0) for m in masks]  # [1, 256, 256, 128]
        return img, masks

    def get_seg_info(self, img_file: str, text: str):
        #seg_tag = text.split(":")[1]
        seg_tag = text.lower()
        
        # TODO: simplify
        if seg_tag == "Nodule":
            text = "Segment pulmonary nodules in the image."
            seg_indices = [1]
        else:
            seg_indices = self.tsseg_map[seg_tag]
            #text = f"Segment {seg_tag} in the image."
        return seg_indices

    def __getitem__(self, idx: int):
        text = self.text_list[idx]

        # TODO: simplify

        if self.img_files is not None:
            img_file = self.img_files[idx]
            seg_file = self.seg_files[idx]


            ### we will get three segmentation at same time
            #seg_indices = self.get_seg_info(img_file, org1) seg)indices is fixed in knee
            img, masks = self.get_img_mask(
                img_file,
                seg_file,
                transform=self.transform,
            )
            if self.transform_imgonly is not None:
                img = self.transform_imgonly(img)

        if self.labels is not None:
            #print('seg shape',mask[0].numpy().shape)
            #print('img shape', img.shape)
            patchify_label_ = []
            for m in masks:
                mask_label = block_reduce(
                    m[0].numpy(), block_size=self.mask_pool_size, func=np.max
                )
                patchify_label = patchify_3d(
                    mask_label, self.patch_size
                )  # (nDxnHxnW, pDxpHxpW)   
                patchify_label_.append(patchify_label)
            patchify_label = np.stack(patchify_label_, axis=0) ##stack along anatomy dimension to get [4, 64, 64]
            #print('patchify_label size', patchify_label.shape)
           
        
        content = {"text": text, "image": img}
        if self.labels is not None:
            content["label"] = self.labels[idx]
            content["seg_label"] = patchify_label
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
            inputs["images"] = torch.stack(images)

        if with_label:
            labels = np.array([item["label"] for item in batch], dtype=np.float16)
            inputs["labels"] = torch.from_numpy(labels[:, None])

            seg_labels = np.stack([item["seg_label"] for item in batch]).astype(
                np.float16
            )
            inputs["seg_label"] = torch.from_numpy(seg_labels)

            
        return inputs



