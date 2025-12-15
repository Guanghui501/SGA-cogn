"""Connectivity Optimized Graph Network (coGN) - PyTorch Implementation.

 

A PyTorch implementation of coGN for crystal property prediction with multi-modal fusion.

Based on the original TensorFlow implementation from kgcnn.

 

Reference: https://arxiv.org/abs/2302.14102

"""

from typing import Tuple, Union, List, Optional

 

import dgl

import dgl.function as fn

import numpy as np

import torch

from dgl.nn import AvgPooling, SumPooling

from pydantic.typing import Literal

from torch import nn

from torch.nn import functional as F

from models.utils import RBFExpansion

from utils import BaseSettings

 

from transformers import AutoTokenizer

from transformers import AutoModel

from tokenizers.normalizers import BertNormalizer

 

"""**VoCab Mapping and Normalizer**"""

 

f = open('vocab_mappings.txt', 'r')

mappings = f.read().strip().split('\n')

f.close()

 

mappings = {m[0]: m[2:] for m in mappings}

 

norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

 

def normalize(text):

    text = [norm.normalize_str(s) for s in text.split('\n')]

    out = []

    for s in text:

        norm_s = ''

        for c in s:

            norm_s += mappings.get(c, ' ')

        out.append(norm_s)

    return '\n'.join(out)

 

"""**Custom Dataset**"""

 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert', model_max_length=512)

text_model = AutoModel.from_pretrained('m3rg-iitd/matscibert')

text_model.to(device)

 

 

# ==================== Periodic Table Data ====================

 

# Atomic masses (index 0 is placeholder, indices 1-118 are elements)

ATOMIC_MASSES = [

    0.0,  # placeholder for index 0

    1.008, 4.003, 6.941, 9.012, 10.81, 12.01, 14.01, 16.00, 19.00, 20.18,  # H-Ne

    22.99, 24.31, 26.98, 28.09, 30.97, 32.07, 35.45, 39.95, 39.10, 40.08,  # Na-Ca

    44.96, 47.87, 50.94, 52.00, 54.94, 55.85, 58.93, 58.69, 63.55, 65.38,  # Sc-Zn

    69.72, 72.63, 74.92, 78.97, 79.90, 83.80, 85.47, 87.62, 88.91, 91.22,  # Ga-Zr

    92.91, 95.95, 98.00, 101.1, 102.9, 106.4, 107.9, 112.4, 114.8, 118.7,  # Nb-Sn

    121.8, 127.6, 126.9, 131.3, 132.9, 137.3, 138.9, 140.1, 140.9, 144.2,  # Sb-Nd

    145.0, 150.4, 152.0, 157.3, 158.9, 162.5, 164.9, 167.3, 168.9, 173.0,  # Pm-Yb

    175.0, 178.5, 180.9, 183.8, 186.2, 190.2, 192.2, 195.1, 197.0, 200.6,  # Lu-Hg

    204.4, 207.2, 209.0, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0,  # Tl-Th

    231.0, 238.0, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0,  # Pa-Fm

    258.0, 259.0, 262.0, 267.0, 270.0, 269.0, 270.0, 270.0, 278.0, 281.0,  # Md-Ds

    281.0, 285.0, 286.0, 289.0, 289.0, 293.0, 293.0, 294.0,  # Rg-Og

]

 

# Atomic radii in Angstroms (van der Waals radii)

ATOMIC_RADII = [

    0.0,  # placeholder

    1.20, 1.40, 1.82, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54,  # H-Ne

    2.27, 1.73, 1.84, 2.10, 1.80, 1.80, 1.75, 1.88, 2.75, 2.31,  # Na-Ca

    2.11, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.63, 1.40, 1.39,  # Sc-Zn

    1.87, 2.11, 1.85, 1.90, 1.85, 2.02, 3.03, 2.49, 2.00, 2.00,  # Ga-Zr

    2.00, 2.00, 2.00, 2.00, 2.00, 1.63, 1.72, 1.58, 1.93, 2.17,  # Nb-Sn

    2.06, 2.06, 1.98, 2.16, 3.43, 2.68, 2.00, 2.00, 2.00, 2.00,  # Sb-Nd

    2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,  # Pm-Yb

    2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.75, 1.66, 1.55,  # Lu-Hg

    1.96, 2.02, 2.07, 1.97, 2.02, 2.20, 3.48, 2.83, 2.00, 2.00,  # Tl-Th

    2.00, 1.86, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,  # Pa-Fm

    2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,  # Md-Ds

    2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,  # Rg-Og

]

 

# Electronegativities (Pauling scale)

ELECTRONEGATIVITIES = [

    0.0,  # placeholder

    2.20, 0.00, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0.00,  # H-Ne

    0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 0.00, 0.82, 1.00,  # Na-Ca

    1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90, 1.65,  # Sc-Zn

    1.81, 2.01, 2.18, 2.55, 2.96, 3.00, 0.82, 0.95, 1.22, 1.33,  # Ga-Zr

    1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93, 1.69, 1.78, 1.96,  # Nb-Sn

    2.05, 2.10, 2.66, 2.60, 0.79, 0.89, 1.10, 1.12, 1.13, 1.14,  # Sb-Nd

    1.13, 1.17, 1.20, 1.20, 1.22, 1.23, 1.24, 1.25, 1.10, 1.27,  # Pm-Yb

    1.30, 1.50, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54, 2.00, 1.62,  # Lu-Hg

    1.87, 2.33, 2.02, 2.00, 2.20, 2.20, 0.70, 0.90, 1.10, 1.30,  # Tl-Th

    1.50, 1.38, 1.36, 1.28, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30,  # Pa-Fm

    1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30,  # Md-Ds

    1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30,  # Rg-Og

]

 

# First ionization energies (eV)

IONIZATION_ENERGIES = [

    0.0,  # placeholder

    13.60, 24.59, 5.39, 9.32, 8.30, 11.26, 14.53, 13.62, 17.42, 21.56,  # H-Ne

    5.14, 7.65, 5.99, 8.15, 10.49, 10.36, 12.97, 15.76, 4.34, 6.11,  # Na-Ca

    6.56, 6.83, 6.75, 6.77, 7.43, 7.90, 7.88, 7.64, 7.73, 9.39,  # Sc-Zn

    6.00, 7.90, 9.79, 9.75, 11.81, 14.00, 4.18, 5.69, 6.22, 6.63,  # Ga-Zr

    6.76, 7.09, 7.28, 7.36, 7.46, 8.34, 7.58, 8.99, 5.79, 7.34,  # Nb-Sn

    8.64, 9.01, 10.45, 12.13, 3.89, 5.21, 5.58, 5.54, 5.47, 5.53,  # Sb-Nd

    5.58, 5.64, 5.67, 6.15, 5.86, 5.94, 6.02, 6.11, 6.18, 6.25,  # Pm-Yb

    5.43, 6.83, 7.55, 7.86, 7.83, 8.44, 8.97, 8.96, 9.23, 10.44,  # Lu-Hg

    6.11, 7.42, 7.29, 8.42, 9.30, 10.75, 4.07, 5.28, 5.17, 6.31,  # Tl-Th

    5.89, 6.19, 6.27, 6.03, 5.97, 6.02, 6.20, 6.28, 6.42, 6.50,  # Pa-Fm

    6.58, 6.65, 4.90, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00,  # Md-Ds

    6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00, 6.00,  # Rg-Og

]

 

 

# ==================== Projection and Fusion Modules ====================

 

class ProjectionHead(nn.Module):

    def __init__(self, embedding_dim, projection_dim=64, dropout=0.1):

        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)

        self.gelu = nn.GELU()

        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(projection_dim)

 

    def forward(self, x):

        projected = self.projection(x)

        x = self.gelu(projected)

        x = self.fc(x)

        x = self.dropout(x)

        x = x + projected

        x = self.layer_norm(x)

        return x

 

 

class ContrastiveLoss(nn.Module):

    """Contrastive loss for aligning graph and text representations."""

 

    def __init__(self, temperature=0.1):

        super().__init__()

        self.temperature = temperature

 

    def forward(self, graph_features, text_features):

        batch_size = graph_features.size(0)

        graph_features = F.normalize(graph_features, dim=1)

        text_features = F.normalize(text_features, dim=1)

        similarity_matrix = torch.matmul(graph_features, text_features.T) / self.temperature

        labels = torch.arange(batch_size, device=graph_features.device)

        loss_g2t = F.cross_entropy(similarity_matrix, labels)

        loss_t2g = F.cross_entropy(similarity_matrix.T, labels)

        loss = (loss_g2t + loss_t2g) / 2.0

        return loss

 

 

class MiddleFusionModule(nn.Module):

    """Middle fusion module for injecting text information into graph encoding."""

 

    def __init__(self, node_dim=64, text_dim=64, hidden_dim=128, num_heads=2, dropout=0.1,

                 use_gate_norm=False, use_learnable_scale=False, initial_scale=1.0):

        super().__init__()

        self.node_dim = node_dim

        self.text_dim = text_dim

        self.hidden_dim = hidden_dim

        self.use_gate_norm = use_gate_norm

        self.use_learnable_scale = use_learnable_scale

 

        self.text_transform = nn.Sequential(

            nn.Linear(text_dim, hidden_dim),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, node_dim)

        )

 

        if use_learnable_scale:

            self.text_scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))

        else:

            self.register_buffer('text_scale', torch.tensor(1.0, dtype=torch.float32))

 

        if use_gate_norm:

            self.gate_norm = nn.LayerNorm(node_dim * 2)

 

        self.gate = nn.Sequential(

            nn.Linear(node_dim + node_dim, node_dim),

            nn.Sigmoid()

        )

 

        self.layer_norm = nn.LayerNorm(node_dim)

        self.dropout = nn.Dropout(dropout)

        self.stored_alphas = None

 

    def forward(self, node_feat, text_feat, batch_num_nodes=None):

        batch_size = text_feat.size(0)

        num_nodes = node_feat.size(0)

 

        text_transformed = self.text_transform(text_feat)

        text_transformed = text_transformed * self.text_scale

 

        if num_nodes != batch_size:

            if batch_num_nodes is not None:

                text_expanded = []

                for i, num in enumerate(batch_num_nodes):

                    text_expanded.append(text_transformed[i].unsqueeze(0).repeat(num, 1))

                text_broadcasted = torch.cat(text_expanded, dim=0)

            else:

                text_pooled = text_transformed.mean(dim=0, keepdim=True)

                text_broadcasted = text_pooled.repeat(num_nodes, 1)

        else:

            text_broadcasted = text_transformed

 

        gate_input = torch.cat([node_feat, text_broadcasted], dim=-1)

 

        if self.use_gate_norm:

            gate_input = self.gate_norm(gate_input)

 

        gate_values = self.gate(gate_input)

        self.stored_alphas = gate_values.mean(dim=1).detach().cpu()

 

        enhanced = node_feat + gate_values * text_broadcasted

        enhanced = self.layer_norm(enhanced)

        enhanced = self.dropout(enhanced)

 

        return enhanced

 

 

class CrossModalAttention(nn.Module):

    """Cross-modal attention between graph and text features."""

 

    def __init__(self, graph_dim=256, text_dim=64, hidden_dim=256, num_heads=4, dropout=0.1):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.num_heads = num_heads

        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

 

        self.g2t_query = nn.Linear(graph_dim, hidden_dim)

        self.g2t_key = nn.Linear(text_dim, hidden_dim)

        self.g2t_value = nn.Linear(text_dim, hidden_dim)

 

        self.t2g_query = nn.Linear(text_dim, hidden_dim)

        self.t2g_key = nn.Linear(graph_dim, hidden_dim)

        self.t2g_value = nn.Linear(graph_dim, hidden_dim)

 

        self.graph_output = nn.Linear(hidden_dim, graph_dim)

        self.text_output = nn.Linear(hidden_dim, text_dim)

 

        self.dropout = nn.Dropout(dropout)

        self.layer_norm_graph = nn.LayerNorm(graph_dim)

        self.layer_norm_text = nn.LayerNorm(text_dim)

 

        self.scale = self.head_dim ** -0.5

 

    def split_heads(self, x, batch_size):

        x = x.view(batch_size, -1, self.num_heads, self.head_dim)

        return x.permute(0, 2, 1, 3)

 

    def forward(self, graph_feat, text_feat, return_attention=False):

        batch_size = graph_feat.size(0)

        attention_weights = {} if return_attention else None

 

        if graph_feat.dim() == 2:

            graph_feat_seq = graph_feat.unsqueeze(1)

        else:

            graph_feat_seq = graph_feat

 

        if text_feat.dim() == 2:

            text_feat_seq = text_feat.unsqueeze(1)

        else:

            text_feat_seq = text_feat

 

        Q_g2t = self.g2t_query(graph_feat_seq)

        K_g2t = self.g2t_key(text_feat_seq)

        V_g2t = self.g2t_value(text_feat_seq)

 

        Q_g2t = self.split_heads(Q_g2t, batch_size)

        K_g2t = self.split_heads(K_g2t, batch_size)

        V_g2t = self.split_heads(V_g2t, batch_size)

 

        attn_g2t = torch.matmul(Q_g2t, K_g2t.transpose(-2, -1)) * self.scale

        attn_g2t = F.softmax(attn_g2t, dim=-1)

        if return_attention:

            attention_weights['graph_to_text'] = attn_g2t.detach()

        attn_g2t = self.dropout(attn_g2t)

 

        context_g2t = torch.matmul(attn_g2t, V_g2t)

        context_g2t = context_g2t.permute(0, 2, 1, 3).contiguous()

        context_g2t = context_g2t.view(batch_size, 1, self.hidden_dim)

        context_g2t = self.graph_output(context_g2t).squeeze(1)

 

        Q_t2g = self.t2g_query(text_feat_seq)

        K_t2g = self.t2g_key(graph_feat_seq)

        V_t2g = self.t2g_value(graph_feat_seq)

 

        Q_t2g = self.split_heads(Q_t2g, batch_size)

        K_t2g = self.split_heads(K_t2g, batch_size)

        V_t2g = self.split_heads(V_t2g, batch_size)

 

        attn_t2g = torch.matmul(Q_t2g, K_t2g.transpose(-2, -1)) * self.scale

        attn_t2g = F.softmax(attn_t2g, dim=-1)

        if return_attention:

            attention_weights['text_to_graph'] = attn_t2g.detach()

        attn_t2g = self.dropout(attn_t2g)

 

        context_t2g = torch.matmul(attn_t2g, V_t2g)

        context_t2g = context_t2g.permute(0, 2, 1, 3).contiguous()

        context_t2g = context_t2g.view(batch_size, 1, self.hidden_dim)

        context_t2g = self.text_output(context_t2g).squeeze(1)

 

        enhanced_graph = self.layer_norm_graph(graph_feat + context_g2t)

        enhanced_text = self.layer_norm_text(text_feat + context_t2g)

 

        if return_attention:

            return enhanced_graph, enhanced_text, attention_weights

        else:

            return enhanced_graph, enhanced_text

 

 

class GatedFusion(nn.Module):

    """Gated fusion module - learns dynamic weights to balance two modalities."""

 

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, dropout=0.1):

        super().__init__()

        self.graph_dim = graph_dim

        self.text_dim = text_dim

 

        self.gate_graph = nn.Sequential(

            nn.Linear(graph_dim, graph_dim // 2),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(graph_dim // 2, 1),

            nn.Sigmoid()

        )

 

        self.gate_text = nn.Sequential(

            nn.Linear(text_dim, text_dim // 2),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(text_dim // 2, 1),

            nn.Sigmoid()

        )

 

        self.graph_transform = nn.Linear(graph_dim, output_dim)

        self.text_transform = nn.Linear(text_dim, output_dim)

 

        self.fusion_transform = nn.Sequential(

            nn.Linear(output_dim, output_dim),

            nn.LayerNorm(output_dim),

            nn.ReLU(),

            nn.Dropout(dropout)

        )

 

    def forward(self, graph_feat, text_feat):

        gate_g = self.gate_graph(graph_feat)

        gate_t = self.gate_text(text_feat)

 

        gate_sum = gate_g + gate_t + 1e-8

        gate_g = gate_g / gate_sum

        gate_t = gate_t / gate_sum

 

        graph_transformed = self.graph_transform(graph_feat)

        text_transformed = self.text_transform(text_feat)

 

        fused = gate_g * graph_transformed + gate_t * text_transformed

        fused = self.fusion_transform(fused)

 

        return fused

 

 

class BilinearFusion(nn.Module):

    """Bilinear pooling fusion - captures second-order interactions."""

 

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, rank=16, dropout=0.1):

        super().__init__()

        self.rank = rank

        self.graph_proj = nn.Linear(graph_dim, rank * output_dim, bias=False)

        self.text_proj = nn.Linear(text_dim, rank * output_dim, bias=False)

        self.output_dim = output_dim

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(output_dim)

 

    def forward(self, graph_feat, text_feat):

        batch_size = graph_feat.size(0)

        graph_proj = self.graph_proj(graph_feat)

        text_proj = self.text_proj(text_feat)

        graph_proj = graph_proj.view(batch_size, self.rank, self.output_dim)

        text_proj = text_proj.view(batch_size, self.rank, self.output_dim)

        fused = torch.sum(graph_proj * text_proj, dim=1)

        fused = self.layer_norm(fused)

        fused = self.dropout(fused)

        return fused

 

 

class AdaptiveFusion(nn.Module):

    """Adaptive fusion - dynamically selects fusion strategy based on content."""

 

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, dropout=0.1):

        super().__init__()

        self.graph_align = nn.Linear(graph_dim, output_dim)

        self.text_align = nn.Linear(text_dim, output_dim)

 

        fusion_input_dim = output_dim * 2

        self.fusion_selector = nn.Sequential(

            nn.Linear(fusion_input_dim, output_dim),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(output_dim, 3),

            nn.Softmax(dim=-1)

        )

 

        self.gate_net = nn.Sequential(

            nn.Linear(output_dim * 2, output_dim),

            nn.Sigmoid()

        )

 

        self.layer_norm = nn.LayerNorm(output_dim)

        self.dropout = nn.Dropout(dropout)

 

    def forward(self, graph_feat, text_feat):

        graph_aligned = self.graph_align(graph_feat)

        text_aligned = self.text_align(text_feat)

 

        concat_feat = torch.cat([graph_aligned, text_aligned], dim=-1)

        fusion_weights = self.fusion_selector(concat_feat)

 

        fusion_add = graph_aligned + text_aligned

        fusion_mul = graph_aligned * text_aligned

 

        gate = self.gate_net(concat_feat)

        fusion_gate = gate * graph_aligned + (1 - gate) * text_aligned

 

        fused = (fusion_weights[:, 0:1] * fusion_add +

                 fusion_weights[:, 1:2] * fusion_mul +

                 fusion_weights[:, 2:3] * fusion_gate)

 

        fused = self.layer_norm(fused)

        fused = self.dropout(fused)

 

        return fused

 

 

class TuckerFusion(nn.Module):

    """Tucker decomposition fusion - efficient high-order tensor decomposition."""

 

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, rank=8, dropout=0.1):

        super().__init__()

        self.rank = rank

        self.graph_factor = nn.Linear(graph_dim, rank, bias=False)

        self.text_factor = nn.Linear(text_dim, rank, bias=False)

        self.core_factor = nn.Linear(rank * rank, output_dim)

        self.layer_norm = nn.LayerNorm(output_dim)

        self.dropout = nn.Dropout(dropout)

 

    def forward(self, graph_feat, text_feat):

        batch_size = graph_feat.size(0)

        graph_compressed = self.graph_factor(graph_feat)

        text_compressed = self.text_factor(text_feat)

        core_tensor = torch.bmm(

            graph_compressed.unsqueeze(2),

            text_compressed.unsqueeze(1)

        )

        core_flat = core_tensor.view(batch_size, -1)

        fused = self.core_factor(core_flat)

        fused = self.layer_norm(fused)

        fused = self.dropout(fused)

        return fused

 

 

class FineGrainedCrossModalAttention(nn.Module):

    """Fine-grained cross-modal attention between atoms and text tokens."""

 

    def __init__(self, node_dim=256, token_dim=768, hidden_dim=256,

                 num_heads=8, dropout=0.1, use_projection=True):

        super().__init__()

        self.node_dim = node_dim

        self.token_dim = token_dim

        self.hidden_dim = hidden_dim

        self.num_heads = num_heads

        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

 

        self.use_projection = use_projection

 

        if use_projection:

            self.node_proj_in = nn.Linear(node_dim, hidden_dim)

            self.token_proj_in = nn.Linear(token_dim, hidden_dim)

 

        self.a2t_query = nn.Linear(hidden_dim if use_projection else node_dim, hidden_dim)

        self.a2t_key = nn.Linear(hidden_dim if use_projection else token_dim, hidden_dim)

        self.a2t_value = nn.Linear(hidden_dim if use_projection else token_dim, hidden_dim)

 

        self.t2a_query = nn.Linear(hidden_dim if use_projection else token_dim, hidden_dim)

        self.t2a_key = nn.Linear(hidden_dim if use_projection else node_dim, hidden_dim)

        self.t2a_value = nn.Linear(hidden_dim if use_projection else node_dim, hidden_dim)

 

        self.node_output = nn.Linear(hidden_dim, node_dim)

        self.token_output = nn.Linear(hidden_dim, token_dim)

 

        self.dropout = nn.Dropout(dropout)

        self.layer_norm_node = nn.LayerNorm(node_dim)

        self.layer_norm_token = nn.LayerNorm(token_dim)

 

        self.scale = self.head_dim ** -0.5

 

    def split_heads(self, x):

        batch_size, seq_len, _ = x.size()

        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        return x.permute(0, 2, 1, 3)

 

    def forward(self, node_feat, token_feat, node_mask=None, token_mask=None,

                return_attention=False):

        batch_size = node_feat.size(0)

        num_atoms = node_feat.size(1)

        seq_len = token_feat.size(1)

 

        node_feat_orig = node_feat

        token_feat_orig = token_feat

 

        if self.use_projection:

            node_feat = self.node_proj_in(node_feat)

            token_feat = self.token_proj_in(token_feat)

 

        attention_weights = {} if return_attention else None

 

        Q_a2t = self.a2t_query(node_feat)

        K_a2t = self.a2t_key(token_feat)

        V_a2t = self.a2t_value(token_feat)

 

        Q_a2t = self.split_heads(Q_a2t)

        K_a2t = self.split_heads(K_a2t)

        V_a2t = self.split_heads(V_a2t)

 

        attn_a2t = torch.matmul(Q_a2t, K_a2t.transpose(-2, -1)) * self.scale

 

        if token_mask is not None:

            token_mask_expanded = token_mask.unsqueeze(1).unsqueeze(2)

            attn_a2t = attn_a2t.masked_fill(~token_mask_expanded, float('-inf'))

 

        attn_a2t = F.softmax(attn_a2t, dim=-1)

 

        if return_attention:

            attention_weights['atom_to_text'] = attn_a2t.detach()

 

        attn_a2t = self.dropout(attn_a2t)

 

        context_a2t = torch.matmul(attn_a2t, V_a2t)

        context_a2t = context_a2t.permute(0, 2, 1, 3).contiguous()

        context_a2t = context_a2t.view(batch_size, num_atoms, self.hidden_dim)

        context_a2t = self.node_output(context_a2t)

 

        Q_t2a = self.t2a_query(token_feat)

        K_t2a = self.t2a_key(node_feat)

        V_t2a = self.t2a_value(node_feat)

 

        Q_t2a = self.split_heads(Q_t2a)

        K_t2a = self.split_heads(K_t2a)

        V_t2a = self.split_heads(V_t2a)

 

        attn_t2a = torch.matmul(Q_t2a, K_t2a.transpose(-2, -1)) * self.scale

 

        if node_mask is not None:

            node_mask_expanded = node_mask.unsqueeze(1).unsqueeze(2)

            attn_t2a = attn_t2a.masked_fill(~node_mask_expanded, float('-inf'))

 

        attn_t2a = F.softmax(attn_t2a, dim=-1)

 

        if return_attention:

            attention_weights['text_to_atom'] = attn_t2a.detach()

 

        attn_t2a = self.dropout(attn_t2a)

 

        context_t2a = torch.matmul(attn_t2a, V_t2a)

        context_t2a = context_t2a.permute(0, 2, 1, 3).contiguous()

        context_t2a = context_t2a.view(batch_size, seq_len, self.hidden_dim)

        context_t2a = self.token_output(context_t2a)

 

        enhanced_nodes = self.layer_norm_node(node_feat_orig + context_a2t)

        enhanced_tokens = self.layer_norm_token(token_feat_orig + context_t2a)

 

        if return_attention:

            return enhanced_nodes, enhanced_tokens, attention_weights

        else:

            return enhanced_nodes, enhanced_tokens

 

 

# ==================== coGN Core Modules ====================

 

class GaussBasisExpansion(nn.Module):

    """Gaussian Basis Expansion for distance encoding."""

 

    def __init__(self, n_gaussians: int = 32, vmin: float = 0.0, vmax: float = 8.0):

        super().__init__()

        self.n_gaussians = n_gaussians

        offset = torch.linspace(vmin, vmax, n_gaussians)

        self.register_buffer('offset', offset)

        self.width = (vmax - vmin) / n_gaussians

 

    def forward(self, distances):

        """

        Args:

            distances: [N] or [N, 1] tensor of distances

        Returns:

            [N, n_gaussians] tensor of gaussian expansions

        """

        if distances.dim() == 1:

            distances = distances.unsqueeze(-1)

        return torch.exp(-((distances - self.offset) ** 2) / (2 * self.width ** 2))

 

 

class AtomEmbedding(nn.Module):

    """Atom embedding layer including element properties."""

 

    def __init__(self, embedding_dim: int = 128, use_atom_mass: bool = True,

                 use_atom_radius: bool = True, use_electronegativity: bool = True,

                 use_ionization_energy: bool = True):

        super().__init__()

        self.embedding_dim = embedding_dim

 

        # Atomic number embedding

        self.atom_embedding = nn.Embedding(119, embedding_dim)

 

        # Register element properties as buffers

        self.use_atom_mass = use_atom_mass

        self.use_atom_radius = use_atom_radius

        self.use_electronegativity = use_electronegativity

        self.use_ionization_energy = use_ionization_energy

 

        # Pad to 119 elements

        def pad_property(prop_list):

            padded = prop_list + [0.0] * (119 - len(prop_list))

            return torch.tensor(padded, dtype=torch.float32)

 

        if use_atom_mass:

            self.register_buffer('atomic_masses', pad_property(ATOMIC_MASSES))

        if use_atom_radius:

            self.register_buffer('atomic_radii', pad_property(ATOMIC_RADII))

        if use_electronegativity:

            self.register_buffer('electronegativities', pad_property(ELECTRONEGATIVITIES))

        if use_ionization_energy:

            self.register_buffer('ionization_energies', pad_property(IONIZATION_ENERGIES))

 

        # Calculate total feature dimension

        extra_features = sum([use_atom_mass, use_atom_radius, use_electronegativity, use_ionization_energy])

        total_dim = embedding_dim + extra_features

 

        # Project to output dimension

        self.output_proj = nn.Linear(total_dim, embedding_dim)

 

    def forward(self, atomic_numbers):

        """

        Args:

            atomic_numbers: [N] tensor of atomic numbers (1-indexed)

        Returns:

            [N, embedding_dim] tensor of atom embeddings

        """

        # Clamp to valid range

        atomic_numbers = torch.clamp(atomic_numbers, 0, 118)

 

        # Get embedding

        embeddings = self.atom_embedding(atomic_numbers)  # [N, embedding_dim]

 

        # Gather element properties

        features = [embeddings]

 

        if self.use_atom_mass:

            mass = self.atomic_masses[atomic_numbers].unsqueeze(-1)

            features.append(mass)

        if self.use_atom_radius:

            radius = self.atomic_radii[atomic_numbers].unsqueeze(-1)

            features.append(radius)

        if self.use_electronegativity:

            en = self.electronegativities[atomic_numbers].unsqueeze(-1)

            features.append(en)

        if self.use_ionization_energy:

            ie = self.ionization_energies[atomic_numbers].unsqueeze(-1)

            features.append(ie)

 

        # Concatenate and project

        combined = torch.cat(features, dim=-1)

        output = self.output_proj(combined)

 

        return output

 

 

class EdgeEmbedding(nn.Module):

    """Edge embedding using Gaussian basis expansion."""

 

    def __init__(self, edge_dim: int = 128, n_gaussians: int = 32,

                 vmin: float = 0.0, vmax: float = 8.0):

        super().__init__()

        self.gauss_expansion = GaussBasisExpansion(n_gaussians, vmin, vmax)

        self.edge_proj = nn.Linear(n_gaussians, edge_dim)

 

    def forward(self, distances):

        """

        Args:

            distances: [E] tensor of edge distances

        Returns:

            [E, edge_dim] tensor of edge embeddings

        """

        gauss_features = self.gauss_expansion(distances)

        edge_features = self.edge_proj(gauss_features)

        return edge_features

 

 

class AttentionAggregation(nn.Module):

    """Attention-based aggregation for messages."""

 

    def __init__(self, input_dim: int, hidden_dim: int = 32):

        super().__init__()

        self.attention_mlp = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),

            nn.SiLU(),

            nn.Linear(hidden_dim, 1)

        )

 

    def forward(self, messages, index, num_nodes):

        """

        Args:

            messages: [E, D] edge messages

            index: [E] destination node indices

            num_nodes: number of nodes

        Returns:

            [N, D] aggregated node features

        """

        # Compute attention weights

        attn_scores = self.attention_mlp(messages)  # [E, 1]

 

        # Softmax over edges going to same node

        attn_weights = torch.zeros_like(attn_scores)

        max_scores = torch.zeros(num_nodes, 1, device=messages.device)

        max_scores.scatter_reduce_(0, index.unsqueeze(-1), attn_scores, reduce='amax', include_self=False)

        attn_scores = attn_scores - max_scores[index]

        exp_scores = torch.exp(attn_scores)

        sum_exp = torch.zeros(num_nodes, 1, device=messages.device)

        sum_exp.scatter_add_(0, index.unsqueeze(-1), exp_scores)

        attn_weights = exp_scores / (sum_exp[index] + 1e-8)

 

        # Weighted aggregation

        weighted_messages = messages * attn_weights

        aggregated = torch.zeros(num_nodes, messages.size(1), device=messages.device)

        aggregated.scatter_add_(0, index.unsqueeze(-1).expand_as(weighted_messages), weighted_messages)

 

        return aggregated

 

 

class coGNConvLayer(nn.Module):

    """Single coGN convolution layer.

 

    Implements the Graph Network update:

    1. Edge update: e' = MLP([e, v_i, v_j])

    2. Node update: v' = MLP(aggregate(e')) + v (residual)

    """

 

    def __init__(self, node_dim: int = 128, edge_dim: int = 128,

                 edge_mlp_layers: int = 5, node_mlp_layers: int = 1,

                 use_attention: bool = True, residual_node: bool = True,

                 dropout: float = 0.0):

        super().__init__()

        self.node_dim = node_dim

        self.edge_dim = edge_dim

        self.use_attention = use_attention

        self.residual_node = residual_node

 

        # Edge MLP: takes concatenated [edge, src_node, dst_node]

        edge_input_dim = edge_dim + node_dim * 2

        edge_layers = []

        for i in range(edge_mlp_layers):

            if i == 0:

                edge_layers.append(nn.Linear(edge_input_dim, edge_dim))

            else:

                edge_layers.append(nn.Linear(edge_dim, edge_dim))

            edge_layers.append(nn.SiLU())

            if dropout > 0:

                edge_layers.append(nn.Dropout(dropout))

        self.edge_mlp = nn.Sequential(*edge_layers)

 

        # Node MLP: takes aggregated edge features

        node_layers = []

        for i in range(node_mlp_layers):

            node_layers.append(nn.Linear(edge_dim if i == 0 else node_dim, node_dim))

            node_layers.append(nn.SiLU())

            if dropout > 0:

                node_layers.append(nn.Dropout(dropout))

        self.node_mlp = nn.Sequential(*node_layers)

 

        # Attention aggregation

        if use_attention:

            self.attention = AttentionAggregation(edge_dim, hidden_dim=32)

 

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, edge_feats: torch.Tensor):

        """

        Args:

            g: DGL graph

            node_feats: [N, node_dim] node features

            edge_feats: [E, edge_dim] edge features

        Returns:

            updated_node_feats: [N, node_dim]

            updated_edge_feats: [E, edge_dim]

        """

        with g.local_scope():

            # Get source and destination node features for each edge

            src, dst = g.edges()

            src_feats = node_feats[src]  # [E, node_dim]

            dst_feats = node_feats[dst]  # [E, node_dim]

 

            # Edge update

            edge_input = torch.cat([edge_feats, src_feats, dst_feats], dim=-1)

            new_edge_feats = self.edge_mlp(edge_input)  # [E, edge_dim]

 

            # Aggregate edges to nodes

            num_nodes = node_feats.size(0)

            if self.use_attention:

                aggregated = self.attention(new_edge_feats, dst, num_nodes)

            else:

                aggregated = torch.zeros(num_nodes, self.edge_dim, device=node_feats.device)

                aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(new_edge_feats), new_edge_feats)

 

            # Node update

            new_node_feats = self.node_mlp(aggregated)

 

            # Residual connection

            if self.residual_node:

                new_node_feats = new_node_feats + node_feats

 

            return new_node_feats, new_edge_feats

 

 

class MLPLayer(nn.Module):

    """Multilayer perceptron layer helper."""

 

    def __init__(self, in_features: int, out_features: int):

        super().__init__()

        self.layer = nn.Sequential(

            nn.Linear(in_features, out_features),

            nn.BatchNorm1d(out_features),

            nn.SiLU(),

        )

 

    def forward(self, x):

        return self.layer(x)

 

 

# ==================== coGN Configuration ====================

 

class coGNConfig(BaseSettings):

    """Hyperparameter schema for coGN model."""

 

    name: Literal["cogn"]

    # coGN specific parameters

    cogn_layers: int = 5  # Number of coGN convolution layers

    node_dim: int = 128  # Node feature dimension

    edge_dim: int = 128  # Edge feature dimension

    edge_mlp_layers: int = 5  # Number of layers in edge MLP

    node_mlp_layers: int = 1  # Number of layers in node MLP

    use_attention: bool = True  # Whether to use attention aggregation

    residual_node: bool = True  # Whether to use residual node updates

 

    # Input features

    atom_input_features: int = 92  # For CGCNN atom features

    edge_input_features: int = 32  # Number of Gaussian basis functions

    cutoff: float = 8.0  # Distance cutoff for edges

 

    # Embedding features

    embedding_features: int = 64

    hidden_features: int = 128  # Same as node_dim for coGN

 

    # Output

    output_features: int = 1

 

    # Dropout for graph layers

    graph_dropout: float = 0.0

 

    # Cross-modal attention settings (late fusion)

    use_cross_modal_attention: bool = True

    cross_modal_hidden_dim: int = 256

    cross_modal_num_heads: int = 4

    cross_modal_dropout: float = 0.1

 

    # Late fusion type

    late_fusion_type: Literal["concat", "gated", "bilinear", "adaptive", "tucker"] = "concat"

    late_fusion_rank: int = 16

    late_fusion_output_dim: int = 64

 

    # Fine-grained attention settings

    use_fine_grained_attention: bool = False

    fine_grained_hidden_dim: int = 256

    fine_grained_num_heads: int = 8

    fine_grained_dropout: float = 0.1

    fine_grained_use_projection: bool = True

    mask_stopwords: bool = False

    remove_stopwords: bool = False

    stopwords_dir: str = ""

 

    # Middle fusion settings

    use_middle_fusion: bool = False

    middle_fusion_layers: str = "2"

    middle_fusion_hidden_dim: int = 128

    middle_fusion_num_heads: int = 2

    middle_fusion_dropout: float = 0.1

    middle_fusion_use_gate_norm: bool = False

    middle_fusion_use_learnable_scale: bool = False

    middle_fusion_initial_scale: float = 1.0

 

    # Contrastive learning settings

    use_contrastive_loss: bool = False

    contrastive_loss_weight: float = 0.1

    contrastive_temperature: float = 0.1

 

    link: Literal["identity", "log", "logit"] = "identity"

    zero_inflated: bool = False

    classification: bool = False

 

    class Config:

        """Configure model settings behavior."""

        env_prefix = "jv_model"

 

 

# ==================== coGN Main Model ====================

 

class coGN(nn.Module):

    """Connectivity Optimized Graph Network for crystal property prediction.

 

    This is a PyTorch implementation of coGN with multi-modal fusion support.

    """

 

    def __init__(self, config: coGNConfig = coGNConfig(name="cogn")):

        """Initialize coGN model."""

        super().__init__()

        self.config = config

        self.classification = config.classification

 

        # Atom embedding: use either CGCNN features or custom embedding

        if config.atom_input_features == 92:

            # CGCNN features - project to node_dim

            self.atom_embedding = MLPLayer(config.atom_input_features, config.node_dim)

        else:

            # Use custom atom embedding with element properties

            self.atom_embedding = AtomEmbedding(

                embedding_dim=config.node_dim,

                use_atom_mass=True,

                use_atom_radius=True,

                use_electronegativity=True,

                use_ionization_energy=True

            )

 

        # Edge embedding

        self.edge_embedding = nn.Sequential(

            GaussBasisExpansion(

                n_gaussians=config.edge_input_features,

                vmin=0.0,

                vmax=config.cutoff

            ),

            MLPLayer(config.edge_input_features, config.edge_dim),

        )

 

        # coGN convolution layers

        self.cogn_layers = nn.ModuleList([

            coGNConvLayer(

                node_dim=config.node_dim,

                edge_dim=config.edge_dim,

                edge_mlp_layers=config.edge_mlp_layers,

                node_mlp_layers=config.node_mlp_layers,

                use_attention=config.use_attention,

                residual_node=config.residual_node,

                dropout=config.graph_dropout

            )

            for _ in range(config.cogn_layers)

        ])

 

        # Readout: average pooling

        self.readout = AvgPooling()

 

        # Projection heads

        self.graph_projection = ProjectionHead(embedding_dim=config.node_dim)

        self.text_projection = ProjectionHead(embedding_dim=768)

 

        # Middle fusion modules

        self.use_middle_fusion = config.use_middle_fusion

        self.middle_fusion_modules = nn.ModuleDict()

        if self.use_middle_fusion:

            fusion_layers = [int(x.strip()) for x in config.middle_fusion_layers.split(',')]

            for layer_idx in fusion_layers:

                self.middle_fusion_modules[f'layer_{layer_idx}'] = MiddleFusionModule(

                    node_dim=config.node_dim,

                    text_dim=64,

                    hidden_dim=config.middle_fusion_hidden_dim,

                    num_heads=config.middle_fusion_num_heads,

                    dropout=config.middle_fusion_dropout,

                    use_gate_norm=config.middle_fusion_use_gate_norm,

                    use_learnable_scale=config.middle_fusion_use_learnable_scale,

                    initial_scale=config.middle_fusion_initial_scale

                )

            self.middle_fusion_layer_indices = fusion_layers

 

        # Fine-grained cross-modal attention

        self.use_fine_grained_attention = config.use_fine_grained_attention

        if self.use_fine_grained_attention:

            self.fine_grained_attention = FineGrainedCrossModalAttention(

                node_dim=config.node_dim,

                token_dim=768,

                hidden_dim=config.fine_grained_hidden_dim,

                num_heads=config.fine_grained_num_heads,

                dropout=config.fine_grained_dropout,

                use_projection=config.fine_grained_use_projection

            )

 

        # Cross-modal attention (global level)

        self.use_cross_modal_attention = config.use_cross_modal_attention

        if self.use_cross_modal_attention:

            self.cross_modal_attention = CrossModalAttention(

                graph_dim=64,

                text_dim=64,

                hidden_dim=config.cross_modal_hidden_dim,

                num_heads=config.cross_modal_num_heads,

                dropout=config.cross_modal_dropout

            )

 

        # Late fusion module

        self.late_fusion_type = config.late_fusion_type

        print(f"\n{'='*80}")

        print(f"coGN Late Fusion Configuration")

        print(f"{'='*80}")

        print(f"Fusion Type: {config.late_fusion_type}")

 

        if config.late_fusion_type == "concat":

            self.fusion_module = None

            self.fc1 = nn.Linear(128, 64)

            self.fc = nn.Linear(64, config.output_features)

            print(f"Parameters: Simple concat, 128 -> 64 -> {config.output_features}")

 

        elif config.late_fusion_type == "gated":

            self.fusion_module = GatedFusion(

                graph_dim=64,

                text_dim=64,

                output_dim=config.late_fusion_output_dim,

                dropout=config.cross_modal_dropout

            )

            self.fc = nn.Linear(config.late_fusion_output_dim, config.output_features)

            print(f"Parameters: Gated fusion, output dim {config.late_fusion_output_dim}")

 

        elif config.late_fusion_type == "bilinear":

            self.fusion_module = BilinearFusion(

                graph_dim=64,

                text_dim=64,

                output_dim=config.late_fusion_output_dim,

                rank=config.late_fusion_rank,

                dropout=config.cross_modal_dropout

            )

            self.fc = nn.Linear(config.late_fusion_output_dim, config.output_features)

            print(f"Parameters: Bilinear fusion, Rank={config.late_fusion_rank}, output dim {config.late_fusion_output_dim}")

 

        elif config.late_fusion_type == "adaptive":

            self.fusion_module = AdaptiveFusion(

                graph_dim=64,

                text_dim=64,

                output_dim=config.late_fusion_output_dim,

                dropout=config.cross_modal_dropout

            )

            self.fc = nn.Linear(config.late_fusion_output_dim, config.output_features)

            print(f"Parameters: Adaptive fusion, output dim {config.late_fusion_output_dim}")

 

        elif config.late_fusion_type == "tucker":

            self.fusion_module = TuckerFusion(

                graph_dim=64,

                text_dim=64,

                output_dim=config.late_fusion_output_dim,

                rank=config.late_fusion_rank,

                dropout=config.cross_modal_dropout

            )

            self.fc = nn.Linear(config.late_fusion_output_dim, config.output_features)

            print(f"Parameters: Tucker fusion, Rank={config.late_fusion_rank}, output dim {config.late_fusion_output_dim}")

 

        else:

            raise ValueError(f"Unknown late_fusion_type: {config.late_fusion_type}")

 

        print(f"{'='*80}\n")

 

        # Contrastive learning module

        self.use_contrastive_loss = config.use_contrastive_loss

        if self.use_contrastive_loss:

            self.contrastive_loss_fn = ContrastiveLoss(temperature=config.contrastive_temperature)

            self.contrastive_loss_weight = config.contrastive_loss_weight

 

        # Link function

        self.link = None

        self.link_name = config.link

        if config.link == "identity":

            self.link = lambda x: x

        elif config.link == "log":

            self.link = torch.exp

            avg_gap = 0.7

            self.fc.bias.data = torch.tensor(np.log(avg_gap), dtype=torch.float)

        elif config.link == "logit":

            self.link = torch.sigmoid

 

    
    def forward(self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],

                return_features=False, return_attention=False, return_intermediate_features=False):

        """coGN forward pass.

 

        Args:

            g: Tuple of (graph, line_graph, text) - line_graph is ignored for coGN

            return_features: If True, return dict with predictions and intermediate features

            return_attention: If True, include attention weights in returned dict

            return_intermediate_features: If True, return features at each fusion stage

 

        Returns:

            If return_features=False: predictions [batch_size]

            If return_features=True: dict with predictions and features

        """

        # Unpack input - coGN doesn't need line graph

        if isinstance(g, (tuple, list)) and len(g) == 3:

            g, lg, text = g  # lg is ignored

        elif isinstance(g, (tuple, list)) and len(g) == 2:

            g, text = g

        else:

            raise ValueError("Expected (graph, line_graph, text) or (graph, text) tuple")

 

        g = g.local_var()

 

        # Text Encoding

        norm_sents = [normalize(s) for s in text]

        encodings = tokenizer(norm_sents, return_tensors='pt', padding=True, truncation=True)

        if torch.cuda.is_available():

            encodings.to(device)

        with torch.no_grad():

            last_hidden_state = text_model(**encodings)[0]

 

        text_tokens = last_hidden_state

        attention_mask = encodings['attention_mask']

        cls_emb = last_hidden_state[:, 0, :]

        text_emb = self.text_projection(cls_emb)

 

        text_emb_base = text_emb.clone() if return_intermediate_features else None

 

        # Initial node features

        x = g.ndata.pop("atom_features")

        x = self.atom_embedding(x)

 

        # Initial edge features

        bondlength = torch.norm(g.edata.pop("r"), dim=1)

        edge_feats = self.edge_embedding(bondlength)

 

        # coGN convolution layers

        for idx, cogn_layer in enumerate(self.cogn_layers):

            x, edge_feats = cogn_layer(g, x, edge_feats)

 

            # Apply middle fusion if configured

            if self.use_middle_fusion and idx in self.middle_fusion_layer_indices:

                batch_num_nodes = g.batch_num_nodes().tolist()

                x = self.middle_fusion_modules[f'layer_{idx}'](x, text_emb, batch_num_nodes)

 

        # Save features after middle fusion

        graph_emb_after_middle = None

        if return_intermediate_features and self.use_middle_fusion:

            temp_graph_emb = self.readout(g, x)

            graph_emb_after_middle = self.graph_projection(temp_graph_emb).clone()

 

        # Save BASE graph features BEFORE any attention

        graph_emb_base = None

        if return_intermediate_features:

            temp_graph_emb = self.readout(g, x)

            graph_emb_base = self.graph_projection(temp_graph_emb).clone()

 

        # Fine-grained cross-modal attention

        fine_grained_attention_weights = None

        if self.use_fine_grained_attention:

            batch_num_nodes = g.batch_num_nodes().tolist()

            batch_size = len(batch_num_nodes)

            max_atoms = max(batch_num_nodes)

            node_dim = x.size(1)

 

            node_features_batched = torch.zeros(batch_size, max_atoms, node_dim,

                                                 device=x.device, dtype=x.dtype)

            node_mask = torch.zeros(batch_size, max_atoms, device=x.device, dtype=torch.bool)

 

            offset = 0

            for i, num_nodes in enumerate(batch_num_nodes):

                node_features_batched[i, :num_nodes] = x[offset:offset+num_nodes]

                node_mask[i, :num_nodes] = True

                offset += num_nodes

 

            if return_attention:

                enhanced_nodes, enhanced_tokens, fine_grained_attention_weights = self.fine_grained_attention(

                        node_features_batched,

                        text_tokens,

                        node_mask=node_mask,

                        token_mask=attention_mask.bool(),

                        return_attention=True

                    )

            else:

                enhanced_nodes, enhanced_tokens = self.fine_grained_attention(

                    node_features_batched,

                    text_tokens,

                    node_mask=node_mask,

                    token_mask=attention_mask.bool()

                )

 

            x_enhanced = torch.zeros_like(x)

            offset = 0

            for i, num_nodes in enumerate(batch_num_nodes):

                x_enhanced[offset:offset+num_nodes] = enhanced_nodes[i, :num_nodes]

                offset += num_nodes

 

            x = x_enhanced

 

        # Save features after fine-grained attention

        graph_emb_after_fine = None

        text_emb_after_fine = None

        if return_intermediate_features and self.use_fine_grained_attention:

            temp_graph_emb = self.readout(g, x)

            graph_emb_after_fine = self.graph_projection(temp_graph_emb).clone()

            text_emb_after_fine = enhanced_tokens[:, 0, :]

            text_emb_after_fine = self.text_projection(text_emb_after_fine).clone()

 

        # Readout

        graph_emb = self.readout(g, x)

        h = self.graph_projection(graph_emb)

 

        # Multi-Modal Representation Fusion

        attention_weights = None

        if self.use_cross_modal_attention:

            if return_attention:

                enhanced_graph, enhanced_text, attention_weights = self.cross_modal_attention(

                    h, text_emb, return_attention=True

                )

            else:

                enhanced_graph, enhanced_text = self.cross_modal_attention(h, text_emb)

 

            if self.late_fusion_type == "concat":

                h = torch.cat([enhanced_graph, enhanced_text], dim=-1)

                h = F.relu(self.fc1(h))

                out = self.fc(h)

            else:

                fused = self.fusion_module(enhanced_graph, enhanced_text)

                out = self.fc(fused)

        else:

            if self.late_fusion_type == "concat":

                h = torch.cat((h, text_emb), 1)

                h = F.relu(self.fc1(h))

                out = self.fc(h)

            else:

                fused = self.fusion_module(h, text_emb)

                out = self.fc(fused)

 

        if self.link:

            out = self.link(out)

 

        if self.classification:

            out = torch.sigmoid(out)

 

        predictions = torch.squeeze(out)

 

        # Return intermediate features if requested

        if return_features or self.use_contrastive_loss or return_attention or return_intermediate_features:

            output_dict = {

                'predictions': predictions,

                'graph_features': h if not self.use_cross_modal_attention else enhanced_graph,

                'text_features': text_emb if not self.use_cross_modal_attention else enhanced_text,

            }

 

            if return_intermediate_features:

                output_dict['graph_base'] = graph_emb_base

                output_dict['text_base'] = text_emb_base

                if self.use_middle_fusion and graph_emb_after_middle is not None:

                    output_dict['graph_middle'] = graph_emb_after_middle

                if self.use_fine_grained_attention and graph_emb_after_fine is not None:

                    output_dict['graph_fine'] = graph_emb_after_fine

                    output_dict['text_fine'] = text_emb_after_fine

                if self.use_cross_modal_attention:

                    output_dict['graph_cross'] = enhanced_graph

                    output_dict['text_cross'] = enhanced_text

 

            if return_attention:

                if attention_weights is not None:

                    output_dict['attention_weights'] = attention_weights

                if fine_grained_attention_weights is not None:

                    output_dict['fine_grained_attention_weights'] = fine_grained_attention_weights

 

            if self.use_contrastive_loss and self.training:

                graph_feat = h if not self.use_cross_modal_attention else enhanced_graph

                text_feat = text_emb if not self.use_cross_modal_attention else enhanced_text

                contrastive_loss = self.contrastive_loss_fn(graph_feat, text_feat)

                output_dict['contrastive_loss'] = contrastive_loss

 

            return output_dict if return_features or return_attention or return_intermediate_features or (self.use_contrastive_loss and self.training) else predictions

 

        return predictions
