import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from classifier import Head, Allocated_Head  # Custom heads for classification
from pytorchvideo.models.hub import i3d_r50, csn_r101, mvit_base_16x4  # Pretrained video models
import i3d
import inception
import r50
import numpy as np
from math import ceil
from ptflops import get_model_complexity_info  # FLOPs calculator

# -----------------------------
# SLOT ATTENTION MODULE
# -----------------------------
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, num_actor_class=64, eps=1e-8, input_dim=64, resolution=[16, 8, 24], allocated_slot=True):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_actor_class = num_actor_class
        self.allocated_slot = allocated_slot
        self.eps = eps
        self.scale = dim ** -0.5
        self.resolution = resolution

        # Slot initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim).cuda())  # Mean of initial slot distribution
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim).cuda()))  # Absolute value of std deviation

        # Feedforward projection layers
        self.FC1 = nn.Linear(dim, dim)
        self.FC2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)

        # Attention query/key/value projection layers
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # Feedforward network after attention
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        # Optional GRU-based slot updater (commented)
        self.gru = nn.GRUCell(dim, dim)

        # Normalization layers
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        # Initialize slots
        mu = self.slots_mu.expand(1, self.num_slots, -1)
        sigma = self.slots_sigma.expand(1, self.num_slots, -1)
        slots = torch.normal(mu, sigma)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pe = SoftPositionEmbed3D(dim, resolution)  # Positional embeddings for 3D inputs

        # Register the initial slot buffer
        self.register_buffer("slots", slots.contiguous())

    def extend_slots(self):
        # Adds new slots by sampling a new set and concatenating with old ones
        mu = self.slots_mu.expand(1, 29, -1)
        sigma = self.slots_sigma.expand(1, 29, -1)
        new_slots = torch.normal(mu, sigma).contiguous()
        new_slots = torch.cat((self.slots[:, :-1, :], new_slots, self.slots[:, -1:, :]), 1)
        self.register_buffer("slots", new_slots)

    def extract_slots_for_oats(self):
        # Extracts predefined slot indices for the OATS dataset
        oats_slot_idx = [13, 12, 50, 6, 3, 55, 1, 0, 5, 10, 8, 51, 9, 53, 2,
                         4, 48, 59, 52, 61, 63, 49, 60, 7, 30, 11, 57, 22, 62, 58,
                         18, 54, 29, 17, 25]
        selected = [self.slots[:, idx:idx+1, :] for idx in oats_slot_idx]
        self.register_buffer("slots", torch.cat(selected, 1))

    def get_3d_slot(self, slots, inputs):
        # Applies slot attention mechanism on 3D inputs
        b, l, h, w, d = inputs.shape
        inputs = self.pe(inputs)  # Add positional encoding
        inputs = inputs.view(b, -1, d)  # Flatten spatial dims

        # Initial feedforward processing of inputs
        inputs = self.LN(inputs)
        inputs = F.relu(self.FC1(inputs))
        inputs = self.FC2(inputs)

        # Attention projection
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        # Scaled dot-product attention
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn_ori = dots.softmax(dim=-1) + self.eps
        attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)

        # Slot update (commented GRU)
        slots = torch.einsum('bjd,bij->bid', v, attn)
        slots = slots.view(b, -1, d)

        # Slot selection: allocated slots vs general slots
        if self.allocated_slot:
            slots = slots[:, :self.num_actor_class, :]
        else:
            slots = slots[:, :self.num_slots, :]

        # Final FFN update to slots
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori

    def forward(self, inputs, num_slots=None):
        # Entry point: apply slot attention and return slots + attention maps
        b, nf, h, w, d = inputs.shape
        slots = self.slots.expand(b, -1, -1)  # Duplicate slot templates for batch
        return self.get_3d_slot(slots, inputs)


# -----------------------------
# POSITIONAL ENCODING FOR 3D GRIDS
# -----------------------------
def build_3d_grid(resolution):
    # Builds a 3D grid of normalized coordinates [T, H, W, 6]
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [*resolution, -1]).unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)  # Add symmetry (forward + reverse)

class SoftPositionEmbed3D(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.embedding = nn.Linear(6, hidden_size, bias=True)
        self.register_buffer("grid", build_3d_grid(resolution))

    def forward(self, inputs):
        # Add learned positional encodings to input
        return inputs + self.embedding(self.grid)
