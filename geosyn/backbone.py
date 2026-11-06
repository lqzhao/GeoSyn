"""Camera-aware DINOv2 features, joint queries and calibrated keypoint rays."""
from __future__ import annotations
import math
from pathlib import Path
import timm
import torch
from torch import nn
import torch.nn.functional as F
from .kinematics import PANDA_LIMITS


class CameraAwarePredictor(nn.Module):
    def __init__(
        self,
        pretrained_dino: bool = False,
        dino_weight: str | Path = None,
        input_size: int = 448,
        joint_width: int = 384,
        trainable_last_blocks: int = 4,
    ) -> None:
        super().__init__()
        if input_size % 14:
            raise ValueError("DINO input_size must be divisible by patch size 14")
        self.input_size = int(input_size)
        self.heatmap_temperature = 0.22
        self.backbone = timm.create_model(
            "vit_large_patch14_dinov2", pretrained=False, img_size=518, dynamic_img_size=True
        )
        if pretrained_dino:
            if dino_weight is None:
                raise ValueError("pretrained_dino requires a public DINOv2 checkpoint path")
            state = torch.load(dino_weight, map_location="cpu", weights_only=False)
            incompatible = self.backbone.load_state_dict(state, strict=False)
            if incompatible.missing_keys or set(incompatible.unexpected_keys) != {"mask_token"}:
                raise RuntimeError(
                    f"DINOv2-L load mismatch: missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
                )
        self.intermediate_blocks = (5, 11, 17, 23)
        self.layer_logits = nn.Parameter(torch.zeros(len(self.intermediate_blocks)))
        self.token_projection = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, joint_width), nn.GELU(), nn.LayerNorm(joint_width)
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, joint_width), nn.GELU(), nn.LayerNorm(joint_width)
        )
        self.patch_ray_projection = nn.Sequential(
            nn.Linear(4, joint_width), nn.GELU(), nn.Linear(joint_width, joint_width)
        )
        self.joint_queries = nn.Parameter(torch.empty(7, joint_width))
        self.joint_positions = nn.Parameter(torch.empty(7, joint_width))
        decoder_layer = nn.TransformerDecoderLayer(
            joint_width,
            8,
            4 * joint_width,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_decoder = nn.TransformerDecoder(
            decoder_layer, 3, norm=nn.LayerNorm(joint_width)
        )
        self.heatmap_query = nn.Sequential(
            nn.LayerNorm(joint_width), nn.Linear(joint_width, joint_width)
        )
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(joint_width), nn.Linear(joint_width, joint_width)
        )
        chain_layer = nn.TransformerEncoderLayer(
            joint_width,
            8,
            4 * joint_width,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.ordered_chain = nn.TransformerEncoder(chain_layer, 3, norm=nn.LayerNorm(joint_width))
        self.q_head = nn.Sequential(
            nn.LayerNorm(joint_width),
            nn.Linear(joint_width, joint_width),
            nn.GELU(),
            nn.Linear(joint_width, 2),
        )
        self.visibility_head = nn.Sequential(nn.LayerNorm(joint_width), nn.Linear(joint_width, 1))
        self.joint_ray_projection = nn.Sequential(
            nn.Linear(4, joint_width), nn.GELU(), nn.Linear(joint_width, joint_width)
        )
        pose_layer = nn.TransformerEncoderLayer(
            joint_width,
            8,
            4 * joint_width,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pose_graph = nn.TransformerEncoder(pose_layer, 3, norm=nn.LayerNorm(joint_width))
        self.node_log_depth = nn.Linear(joint_width, 1)
        self.global_log_depth = nn.Sequential(
            nn.LayerNorm(joint_width),
            nn.Linear(joint_width, joint_width),
            nn.GELU(),
            nn.Linear(joint_width, 1),
        )
        self.register_buffer("joint_limits", PANDA_LIMITS.clone(), persistent=True)
        self._initialize_new_parameters()
        self.configure_backbone_trainability(trainable_last_blocks)

    def _initialize_new_parameters(self) -> None:
        nn.init.trunc_normal_(self.joint_queries, std=0.02)
        nn.init.trunc_normal_(self.joint_positions, std=0.02)
        nn.init.zeros_(self.q_head[-1].weight)
        with torch.no_grad():
            self.q_head[-1].bias[0] = 0.0
            self.q_head[-1].bias[1] = -1.0
        nn.init.zeros_(self.node_log_depth.weight)
        nn.init.zeros_(self.node_log_depth.bias)
        nn.init.zeros_(self.global_log_depth[-1].weight)
        nn.init.zeros_(self.global_log_depth[-1].bias)

    def configure_backbone_trainability(self, last_blocks: int) -> None:
        if not 0 <= last_blocks <= len(self.backbone.blocks):
            raise ValueError(f"invalid trainable_last_blocks={last_blocks}")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if last_blocks:
            for block in self.backbone.blocks[-last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad_(True)
        self.trainable_last_blocks = int(last_blocks)

    @staticmethod
    def soft_coordinates(logits: torch.Tensor, temperature: float):
        (batch, nodes, height, width) = logits.shape
        probability = torch.softmax(logits.flatten(2) / temperature, -1)
        x = (torch.arange(width, device=logits.device, dtype=logits.dtype) + 0.5) / width
        y = (torch.arange(height, device=logits.device, dtype=logits.dtype) + 0.5) / height
        (yy, xx) = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xx, yy), -1).reshape(-1, 2)
        uv = probability @ grid
        entropy = -(probability * probability.clamp_min(1e-09).log()).sum(-1)
        return (uv, probability, entropy / math.log(height * width))

    @staticmethod
    def _ray_descriptor(
        camera: torch.Tensor, height: int, width: int, image_height: int, image_width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.arange(width, device=camera.device, dtype=torch.float32) + 0.5
        y = torch.arange(height, device=camera.device, dtype=torch.float32) + 0.5
        x = x * (float(image_width) / width)
        y = y * (float(image_height) / height)
        (yy, xx) = torch.meshgrid(y, x, indexing="ij")
        pixel = torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(-1, 3)
        pixel = pixel[None].expand(len(camera), -1, -1)
        rays = torch.einsum("bij,bnj->bni", torch.linalg.inv(camera.float()), pixel)
        ray_xy = rays[..., :2] / rays[..., 2:3].clamp_min(1e-06)
        centered = ray_xy - ray_xy.mean(1, keepdim=True)
        return (torch.cat((ray_xy, centered), -1), ray_xy)

    def forward(
        self, image: torch.Tensor, camera: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if image.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"expected {self.input_size}x{self.input_size}, got {tuple(image.shape[-2:])}"
            )
        stages = self.backbone.get_intermediate_layers(
            image, n=self.intermediate_blocks, reshape=True, return_prefix_tokens=True, norm=True
        )
        maps = torch.stack([value[0] for value in stages], 0)
        prefixes = torch.stack([value[1][:, 0] for value in stages], 0)
        layer_weight = torch.softmax(self.layer_logits, 0).to(maps)
        fused_map = torch.einsum("l,lbchw->bchw", layer_weight, maps)
        global_dino = torch.einsum("l,lbc->bc", layer_weight, prefixes)
        (batch, _, height, width) = fused_map.shape
        memory = self.token_projection(fused_map.flatten(2).transpose(1, 2))
        global_token = self.global_projection(global_dino)
        patch_ray_xy = None
        if camera is not None:
            if camera.shape != (batch, 3, 3):
                raise ValueError(f"camera must be [B,3,3], got {tuple(camera.shape)}")
            (descriptor, patch_ray_xy) = self._ray_descriptor(
                camera, height, width, image.shape[-2], image.shape[-1]
            )
            memory = memory + self.patch_ray_projection(descriptor).to(memory.dtype)
        queries = self.joint_queries[None].expand(batch, -1, -1)
        queries = queries + self.joint_positions[None] + global_token[:, None]
        decoded = self.visual_decoder(queries, memory)
        heat_query = F.normalize(self.heatmap_query(decoded).float(), dim=-1)
        heat_memory = F.normalize(memory.float(), dim=-1)
        logits = torch.einsum("bnc,bpc->bnp", heat_query, heat_memory)
        logits = (
            (logits * math.sqrt(heat_memory.shape[-1]))
            .reshape(batch, self.joint_queries.shape[0], height, width)
            .to(memory.dtype)
        )
        (uv, probability, entropy) = self.soft_coordinates(logits, self.heatmap_temperature)
        semantic = probability @ memory
        chain = self.ordered_chain(
            decoded + self.semantic_projection(semantic) + self.joint_positions[None]
        )
        raw = self.q_head(chain[:, self.joint_indices])
        normalized = torch.sigmoid(raw[..., 0])
        (lower, upper) = (self.joint_limits[:, 0], self.joint_limits[:, 1])
        q = lower[None] + normalized * (upper - lower)[None]
        visibility_logits = self.visibility_head(chain)[..., 0]
        confidence = torch.sigmoid(visibility_logits) * torch.exp(-2.5 * entropy)
        result = {
            "q": q,
            "q_normalized": normalized,
            "q_log_variance": raw[..., 1].clamp(-5.0, 3.0),
            "heatmap_logits": logits,
            "uv_normalized": uv,
            "visibility_logits": visibility_logits,
            "confidence": confidence.clamp(0.02, 1.0),
            "entropy": entropy,
            "feature": fused_map,
            "dino_layer_weights": layer_weight,
            "joint_tokens": chain,
            "visual_memory_mean": memory.mean(1),
            "global_visual_token": global_token,
        }
        if camera is not None:
            pixel = uv.float() * uv.new_tensor((float(image.shape[-1]), float(image.shape[-2])))
            homogeneous = torch.cat((pixel, torch.ones_like(pixel[..., :1])), -1)
            rays = torch.einsum("bij,bnj->bni", torch.linalg.inv(camera.float()), homogeneous)
            ray_xy = rays[..., :2] / rays[..., 2:3].clamp_min(1e-06)
            ray_feature = torch.cat((ray_xy, ray_xy - ray_xy.mean(1, keepdim=True)), -1)
            pose_token = self.pose_graph(
                chain + self.joint_ray_projection(ray_feature).to(chain.dtype)
            )
            global_depth = self.global_log_depth(global_token + memory.mean(1))[:, None]
            local_depth = self.node_log_depth(pose_token)
            result.update(
                {
                    "ray_xy": ray_xy,
                    "pose_tokens": pose_token,
                    "log_depth_residual": (0.55 * torch.tanh(global_depth + local_depth))[..., 0],
                }
            )
        return result
