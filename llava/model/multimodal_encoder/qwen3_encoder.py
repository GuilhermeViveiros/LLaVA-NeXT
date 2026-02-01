from transformers import Qwen3VLForConditionalGeneration, Qwen2VLImageProcessorFast, Qwen3VLPreTrainedModel, AutoProcessor
from transformers import PretrainedConfig
import torch
from torch import nn
from llava.utils import rank0_print
import torch.nn.functional as F

class Qwen3VLVisionConfig(PretrainedConfig):
    model_type = "qwen3_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=24,
        hidden_size=1024,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4096,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=2560,
        num_position_embeddings=2304,
        #deepstack_visual_indexes=[5, 11, 17],
        #initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        #self.initializer_range = initializer_range
        #self.deepstack_visual_indexes = deepstack_visual_indexes

class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        #import pdb; pdb.set_trace()
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        #import pdb; pdb.set_trace()
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock, Qwen3VLVisionPatchMerger

# class Qwen3VLVisionBlock(GradientCheckpointingLayer):
#     def __init__(self, config, attn_implementation: str = "sdpa") -> None:
#         super().__init__()
#         self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
#         self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
#         self.attn = Qwen3VLVisionAttention(config=config)
#         self.mlp = Qwen3VLVisionMLP(config=config)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         cu_seqlens: torch.Tensor,
#         rotary_pos_emb: Optional[torch.Tensor] = None,
#         position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
#         **kwargs,
#     ) -> torch.Tensor:
#         hidden_states = hidden_states + self.attn(
#             self.norm1(hidden_states),
#             cu_seqlens=cu_seqlens,
#             rotary_pos_emb=rotary_pos_emb,
#             position_embeddings=position_embeddings,
#             **kwargs,
#         )
#         hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
#         return hidden_states

# class Qwen3VLVisionPatchMerger(nn.Module):
#     def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
#         super().__init__()
#         self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
#         self.use_postshuffle_norm = use_postshuffle_norm
#         self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
#         self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.act_fn = nn.GELU()
#         self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
#         x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
#         return x


class Qwen3VLVisionModel(Qwen3VLPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config=config)

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        #deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # if layer_num in self.deepstack_visual_indexes:
            #     deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
            #         hidden_states
            #     )
            #     deepstack_feature_lists.append(deepstack_feature)

        # hidden_states = self.merger(hidden_states)

        return hidden_states #, deepstack_feature_lists



class Qwen3VisionTower(nn.Module):

    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        # FIXME: this only works for the small model < 9B parameters (qwen3VL 30B uses a different vision encoder)
        self.config = Qwen3VLVisionConfig()
        
        # TODO: improve this later and add it into the Qwen3 Folder
        self.image_processor = Qwen2VLImageProcessorFast.from_pretrained(vision_tower)
        self.image_processor.min_pixels = 100 * 16 * 16
        self.image_processor.max_pixels = 4096 * 16 * 16
        
        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config
        
    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return
        
        # load model
        self.vision_tower = Qwen3VLVisionModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        del self.vision_tower.merger # qwen3VL merger will not be used for now
        del self.vision_tower.deepstack_merger_list # deepstack will not be used for now

        #self.vision_tower.head = nn.Identity()
        self.vision_tower.requires_grad_(False)

    def forward(self, images, image_grids):
        image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype), image_grids)
        assert image_features.shape[-1] == 1024
        return image_features

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

    # @property
    # def num_patches(self): # num_patches is computed dynnamically
    #     return (self.config.image_size // self.config.patch_size) ** 2

    # @property
    # def num_patches_per_side(self):
    #     return self.config.image_size // self.config.patch_size
    #     # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    # @property
    # def image_size(self):
    #     return self.config.image_size

    
if __name__=="__main__":
    # load Qwen3-VL VisionEncoder
    ve = Qwen3VisionTower(
        "GuilhermeNunes/Qwen3-VE-300M",
        {
        }
    )


# # default: Load the model on the available device(s)
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
# )

# import pdb; pdb.set_trace()

# processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
# model.visual
# image_processor=processor.image_processor
# video_processor=processor.video_processor
# # localir - "Qwen3-VE-300M/"
# image_processor.save_pretrained("Qwen3-VE-300M")
# video_processor.save_pretrained("Qwen3-VE-300M")
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# # Preparation for inference
# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_dict=True,
#     return_tensors="pt"
# )
# inputs = inputs.to(model.device)

# import pdb; pdb.set_trace()

# # config = Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

# # processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

# # # access image features
# # pixel_values = inputs["pixel_values"]
# # grid_thw = inputs["image_grid_thw"]

# # vision_outputs = model.visual(
# #     hidden_states=pixel_values,
# #     grid_thw=grid_thw,
# #     #output_hidden_states=True,
# #     #return_dict=True,
# # )



# # # Inference: Generation of the output
# # generated_ids = model.generate(**inputs, max_new_tokens=128)
# # generated_ids_trimmed = [
# #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# # ]
# # output_text = processor.batch_decode(
# #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# # )
# # print(output_text)

# # import pdb; pdb.set_trace()
