# Code ported from https://github.com/openai/CLIP

import hashlib
import os
import urllib
import warnings
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, InterpolationMode, RandomCrop, RandomRotation
from tqdm import tqdm


__all__ = ["available_models", "load", "tokenize"]
# _tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}


class NormalizeByImage(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        for t in tensor:
            t.sub_(t.mean()).div_(t.std() + 1e-7)
        return tensor


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def _convert_to_rgb(image):
    return image.convert('RGB')

def _transform(n_px_tr: int, n_px_val: int, is_train: bool, normalize:str = "dataset", preprocess:str = "downsize"):
    if normalize == "img":
        normalize = NormalizeByImage()
    elif normalize == "dataset":
        normalize = Normalize((47.1314, 40.8138, 53.7692, 46.2656, 28.7243), (47.1314, 40.8138, 53.7692, 46.2656, 28.7243))  # normalize for CellPainting
    if normalize == "None":
        normalize = None

    if is_train:
        if preprocess == "crop":
            #resize = RandomResizedCrop(n_px_tr, scale=(0.25,0.3), ratio=(0.95, 1.05), interpolation=InterpolationMode.BICUBIC)
            resize =  RandomCrop(n_px_tr)
        elif preprocess == "downsize":
            resize = RandomResizedCrop(n_px_tr, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC)
        elif preprocess == "rotate":
            resize = Compose([
                              RandomRotation((0, 360)),
                              CenterCrop(n_px_tr)
                            ])

    else:
        if preprocess == "crop" or "rotate":
            resize = Compose([
                              #RandomResizedCrop(n_px_tr, scale=(0.25,0.3), ratio=(0.95, 1.05), interpolation=InterpolationMode.BICUBIC)
                              CenterCrop(n_px_val),
                              ])
        elif preprocess == "downsize":
            resize = Compose([
                              Resize(n_px_val, interpolation=InterpolationMode.BICUBIC),
                              CenterCrop(n_px_val),
                              ])
    if normalize:
        return Compose([
            ToTensor(),
            resize,
            normalize,
        ])
    else:
        return Compose([
            ToTensor(),
            resize,
        ])



def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())
