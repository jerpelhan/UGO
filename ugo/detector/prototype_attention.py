# Prototype aggregation adapted from GECO2 commit ef6954a59d22620b9a3052f8500530a799c0379b:
# https://github.com/jerpelhan/GECO2/blob/ef6954a59d22620b9a3052f8500530a799c0379b/models/query_generator.py

from typing import Tuple

import torch
from torch import nn

from .ops.modules.ms_deform_attn import MSDeformAttn
from .layers import PrototypeAttentionBlock, UpsamplingLayer


class PrototypeAttention(nn.Module):

    def __init__(
        self,
        *,
        transformer_dim: int,
        num_prototype_attn_steps: int,
        num_image_attn_steps: int,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.image_attention = nn.ModuleList()
        self.image_attention_l1 = nn.ModuleList()
        self.image_attention_l2 = nn.ModuleList()
        self.prototype_attention = nn.ModuleList()
        self.prototype_attention_l1 = nn.ModuleList()
        self.prototype_attention_l2 = nn.ModuleList()

        for _ in range(num_prototype_attn_steps):
            self.prototype_attention.append(
                PrototypeAttentionBlock(
                    embedding_dim=transformer_dim,
                    num_heads=8,
                ))
            self.prototype_attention_l1.append(
                PrototypeAttentionBlock(
                    embedding_dim=transformer_dim,
                    num_heads=8,
                ))
            self.prototype_attention_l2.append(
                PrototypeAttentionBlock(
                    embedding_dim=transformer_dim,
                    num_heads=8,
                ))

        for _ in range(num_image_attn_steps):
            self.image_attention.append(
                MSDeformAttn(
                    d_model=256,
                    n_levels=1,
                    n_heads=8,
                    n_points=8,
                ))
            self.image_attention_l1.append(
                MSDeformAttn(
                    d_model=256,
                    n_levels=1,
                    n_heads=8,
                    n_points=8,
                ))
            self.image_attention_l2.append(
                MSDeformAttn(
                    d_model=256,
                    n_levels=1,
                    n_heads=8,
                    n_points=8,
                ))

        self.up1 = UpsamplingLayer(transformer_dim, transformer_dim)
        self.up2 = UpsamplingLayer(transformer_dim, transformer_dim)
        self.up3 = UpsamplingLayer(transformer_dim, transformer_dim)
        self.up_aux = UpsamplingLayer(transformer_dim, transformer_dim)

        height, width = 64, 64
        self.spatial_shapes = torch.tensor([[height, width]])
        self.valid_ratios = torch.tensor([[1.0, 1.0]])
        self.level_start_index = torch.tensor([[0]])
        self.spatial_shapes2 = torch.tensor([[height * 2, width * 2]])
        self.valid_ratios2 = torch.tensor([[1.0, 1.0]])
        self.level_start_index2 = torch.tensor([[0]])
        self.spatial_shapes1 = torch.tensor([[height * 4, width * 4]])
        self.valid_ratios1 = torch.tensor([[1.0, 1.0]])
        self.level_start_index1 = torch.tensor([[0]])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device="cpu"):
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5,
                    height - 0.5,
                    height,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.linspace(
                    0.5,
                    width - 0.5,
                    width,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[level, 0] * width)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        prototype_embeddings: torch.Tensor,
        hq_features: torch.Tensor,
        hq_prototypes: torch.Tensor,
        hq_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.spatial_shapes.device != image_embeddings.device:
            self.spatial_shapes = self.spatial_shapes.to(image_embeddings.device)
            self.spatial_shapes1 = self.spatial_shapes1.to(image_embeddings.device)
            self.spatial_shapes2 = self.spatial_shapes2.to(image_embeddings.device)
            self.level_start_index = self.level_start_index.to(image_embeddings.device)
            self.level_start_index1 = self.level_start_index1.to(image_embeddings.device)
            self.level_start_index2 = self.level_start_index2.to(image_embeddings.device)
            self.valid_ratios = self.valid_ratios.to(image_embeddings.device)
            self.valid_ratios1 = self.valid_ratios1.to(image_embeddings.device)
            self.valid_ratios2 = self.valid_ratios2.to(image_embeddings.device)
            self.reference_points1 = self.get_reference_points(
                self.spatial_shapes1,
                self.valid_ratios1,
                device=image_embeddings.device,
            )
            self.reference_points2 = self.get_reference_points(
                self.spatial_shapes2,
                self.valid_ratios2,
                device=image_embeddings.device,
            )
            self.reference_points = self.get_reference_points(
                self.spatial_shapes,
                self.valid_ratios,
                device=image_embeddings.device,
            )

        batch_size, channels, height, width = image_embeddings.shape
        image_pe = torch.repeat_interleave(
            image_pe,
            image_embeddings.shape[0],
            dim=0,
        )
        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        src = image_embeddings

        hq_features_l1_pos = hq_pos[0].flatten(2).permute(0, 2, 1)
        hq_features_l2_pos = hq_pos[1].flatten(2).permute(0, 2, 1)
        hq_features_l1 = hq_features[0].flatten(2).permute(0, 2, 1)
        hq_features_l2 = hq_features[1].flatten(2).permute(0, 2, 1)

        for layer in self.prototype_attention:
            src = layer(
                image_f=src,
                prototypes=prototype_embeddings,
            )
        for layer in self.prototype_attention_l1:
            hq_features_l1 = layer(
                image_f=hq_features_l1,
                prototypes=hq_prototypes[0],
            )
        for layer in self.prototype_attention_l2:
            hq_features_l2 = layer(
                image_f=hq_features_l2,
                prototypes=hq_prototypes[1],
            )

        for layer in self.image_attention:
            src = layer(
                src + image_pe,
                self.reference_points,
                src,
                self.spatial_shapes,
                self.level_start_index,
            )
        for layer in self.image_attention_l1:
            hq_features_l1 = layer(
                hq_features_l1 + hq_features_l1_pos,
                self.reference_points1,
                hq_features_l1,
                self.spatial_shapes1,
                self.level_start_index1,
            )
        for layer in self.image_attention_l2:
            hq_features_l2 = layer(
                hq_features_l2 + hq_features_l2_pos,
                self.reference_points2,
                hq_features_l2,
                self.spatial_shapes2,
                self.level_start_index2,
            )

        src = src.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )
        hq_features_l2 = hq_features_l2.transpose(1, 2).view(
            batch_size,
            channels,
            height * 2,
            width * 2,
        )
        hq_features_l1 = hq_features_l1.transpose(1, 2).view(
            batch_size,
            channels,
            height * 4,
            width * 4,
        )
        src = self.up1(src) + hq_features_l2
        src = self.up2(src) + hq_features_l1
        src = self.up3(src)
        src_aux = self.up_aux(hq_features_l1)
        return src, src_aux
