import os
import torch 
import json 
import numpy as np 
from PIL import Image

from src.clip.clip import _transform
from src.clip.model import CLIPGeneral


class CLOOME:

    def __init__(self, checkpoint, config, image_res=[520, 696]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform = self._load(checkpoint, config, image_res)
        

    
    def encode_image(self, input, normalize=True):
        """Encode image given a list of tiff files. The order of the channels should be Mito, ERSyto, ERSytoBleed, Ph_golgi, Hoechst"""
        if isinstance(input[0], str):
            images = self._read_img(input)
            images = images.unsqueeze(0)

        if isinstance(input[0], list):
            images = [self._read_img(i) for i in input]
            images = torch.stack(images, dim=0)

        with torch.no_grad():
            embedding = self.model.encode_image(images.to(self.device))
            
            if normalize:
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding



    def encode_molecule(self, input, normalize=True):
        input = torch.from_numpy(input)

        with torch.no_grad():
            embedding = self.model.encode_text(input.to(self.device))
            
            if normalize:
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                
        return embedding
    
    

    def _read_img(self, lst):
        assert len(lst) == 5, f"Input has {len(lst)} channels, it should have 5"

        images = [np.array(Image.open(f)) for f in lst]
        thres = [self._illumination_threshold(i) for i in images]
        images = [self._sixteen_to_eight_bit(i, t) for i, t in zip(images, thres)]

        image = np.stack(images, axis=2)
        image = self.transform(image)

        return image


    def _load(self, checkpoint, config, image_res):
        checkpoint = torch.load(checkpoint, weights_only=True)
        state_dict = checkpoint["state_dict"]

        assert os.path.exists(config)

        with open(config, 'r') as f:
            model_info = json.load(f)

        model = CLIPGeneral(**model_info)

        if str(self.device) == "cpu":
            model.float()

        new_state_dict = {k[len('module.'):]: v for k,v in state_dict.items()}

        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()

        return model, _transform(image_res, image_res, is_train=False)
    


    def _sixteen_to_eight_bit(self, arr, display_max, display_min=0):
        threshold_image = ((arr.astype(float) - display_min) * (arr > display_min))

        scaled_image = (threshold_image * (255 / (display_max - display_min)))
        scaled_image[scaled_image > 255] = 255

        scaled_image = scaled_image.astype(np.uint8)

        return scaled_image
    


    def _illumination_threshold(self, arr, perc=0.01):
        """ Return threshold value to not display a percentage of highest pixels"""

        perc = perc/100

        h = arr.shape[0]
        w = arr.shape[1]

        # find n pixels to delete
        total_pixels = h * w
        n_pixels = total_pixels * perc
        n_pixels = int(np.around(n_pixels))

        # find indexes of highest pixels
        flat_inds = np.argpartition(arr, -n_pixels, axis=None)[-n_pixels:]
        inds = np.array(np.unravel_index(flat_inds, arr.shape)).T

        max_values = [arr[i, j] for i, j in inds]

        threshold = min(max_values)

        return threshold