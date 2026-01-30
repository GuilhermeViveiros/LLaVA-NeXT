import json
import os
import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoImageProcessor, AutoConfig
from llava.utils import rank0_print



class MoonVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.config = AutoConfig.from_pretrained(vision_tower, trust_remote_code=True)

        self.vision_tower_name = vision_tower
        
        
        self.image_processor = AutoImageProcessor.from_pretrained(vision_tower, trust_remote_code=True)

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            if self._checkpoint_has_vision_tower_weights(vision_tower_cfg):
                rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
                self.load_model()
            else:
                self.cfg_only = self.config
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def _checkpoint_has_vision_tower_weights(self, vision_tower_cfg):
        if hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            return True

        ckpt_dir = self._get_local_checkpoint_dir(vision_tower_cfg)
        if not ckpt_dir:
            return False

        index_files = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        for filename in index_files:
            path = os.path.join(ckpt_dir, filename)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    weight_map = json.load(handle).get("weight_map", {})
            except Exception:
                return False
            for key in weight_map.keys():
                if key.startswith("vision_tower.") or key.startswith("model.vision_tower.") or ".vision_tower." in key:
                    return True
            return False

        weight_files = ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt")
        return any(os.path.isfile(os.path.join(ckpt_dir, filename)) for filename in weight_files)

    @staticmethod
    def _get_local_checkpoint_dir(vision_tower_cfg):
        candidate_attrs = ("model_name_or_path", "_name_or_path", "name_or_path", "pretrained_model_name_or_path", "model_path")
        for attr in candidate_attrs:
            path = getattr(vision_tower_cfg, attr, None)
            if path and os.path.isdir(path):
                return path
        return None

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, device_map=device_map, trust_remote_code=True)

        #del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.head = nn.Identity()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images, image_grids):
        # if type(images) is list:
        #     image_features = []
        #     for image, image_grid in zip(images, image_grids):
        #         image_forward_out = self.vision_tower(images.to(device=self.device, dtype=self.dtype), images_processed.image_grid_hws)
        #         image_feature = image_forward_out[-1].to(image.dtype)
        #         #assert image_features.shape[-2] == 729
        #         image_features.append(image_feature)
        # else:
        image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype), image_grids)
        for img_feat in image_features:
            assert img_feat.shape[-1] == 1152

        return image_features

    # def preprocess(
    #     self,
    #     images: ImageInput,
    #     return_tensors: Optional[Union[str, TensorType]] = None,
    # ) -> BatchFeature:
    #     images = make_list_of_images(images)

    #     if not valid_images(images):
    #         raise ValueError(
    #             "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
    #             "torch.Tensor, tf.Tensor or jax.ndarray."
    #         )

    #     pixel_values, image_grid_hws = [], []
    #     for image in images:
    #         patches, image_grid_hw = self._preprocess(image)
    #         pixel_values.append(patches)
    #         image_grid_hws.append(image_grid_hw)
    #     pixel_values = torch.concat(pixel_values, dim=0)
    #     image_grid_hws = np.array(image_grid_hws)
    #     data = {"pixel_values": pixel_values, "image_grid_hws": image_grid_hws}

    #     return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size


if __name__=="__main__":
    ve = MoonVisionTower("moonshotai/MoonViT-SO-400M", {})
    # Load a random image and process it through the vision tower
    from PIL import Image
    image_path = "images/Art_test_Art_113.png"
    import torchvision.transforms as T
    img = Image.open(image_path).convert("RGB")
    inp = image_processor(img, return_tensors="pt").to(dtype=ve.dtype, device=ve.device)
    output = ve(inp)
    print("Output:", output)