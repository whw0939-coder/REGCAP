"""Multi-view REGCAP network and regional importance aggregation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv

from .rdp_schema import RDP_DIM

class HierAST(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim,
                 stopgrad_between_levels=False):
        super().__init__()
        self.tok = GCNConv(in_dim, hid_dim)
        self.stm = GCNConv(in_dim, hid_dim)
        self.blk = GCNConv(in_dim, hid_dim)

        self.n_t = nn.LayerNorm(hid_dim)
        self.n_s = nn.LayerNorm(hid_dim)
        self.n_b = nn.LayerNorm(hid_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(hid_dim * 3, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim),
        )
        self.stopgrad_between_levels = stopgrad_between_levels

    def forward(self, x, token_ei, stmt_ei, block_ei):
        xt = self.n_t(F.relu(self.tok(x, token_ei)))
        x1 = xt.detach() if self.stopgrad_between_levels else xt
        xs = self.n_s(F.relu(self.stm(x1, stmt_ei)))
        x2 = xs.detach() if self.stopgrad_between_levels else xs
        xb = self.n_b(F.relu(self.blk(x2, block_ei)))
        h = torch.cat([xt, xs, xb], dim=-1)
        out = self.out_proj(h)
        return out

class GGNNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_steps=6, out_dim=None):
        super().__init__()
        self.input = nn.Linear(in_dim, hid_dim) if in_dim != hid_dim else nn.Identity()
        self.ggnn = GatedGraphConv(out_channels=hid_dim, num_layers=n_steps)
        self.norm = nn.LayerNorm(hid_dim)
        self.out = nn.Identity() if (out_dim is None or out_dim == hid_dim) else nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.input(x)
        x = self.ggnn(x, edge_index)
        x = self.norm(F.relu(x))
        return self.out(x)

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, heads=4, out_dim=None):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hid_dim) if in_dim != hid_dim else nn.Identity()
        self.gat1 = GATConv(hid_dim, hid_dim // heads, heads=heads, concat=True)
        self.gat2 = GATConv(hid_dim, hid_dim // heads, heads=heads, concat=True)
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        self.out = nn.Identity() if (out_dim is None or out_dim == hid_dim) else nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.proj_in(x)
        x = self.norm1(F.elu(self.gat1(x, edge_index)))
        x = self.norm2(F.elu(self.gat2(x, edge_index)))
        return self.out(x)

class AutoMaskGenerator(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.eps = epsilon
    def forward(self, emb):
        norms = torch.norm(emb, p=2, dim=-1)
        mask = norms < self.eps
        all_pad = mask.all(dim=1)
        mask[all_pad, 0] = False
        return mask

def masked_region_pool(node_feat, region_nodes, node_mask, reduce="mean"):
    B, K, M = region_nodes.shape
    D = node_feat.size(-1)
    safe_idx = torch.where(node_mask, region_nodes, torch.zeros_like(region_nodes))
    flat = safe_idx.view(-1)
    picked = node_feat[flat].view(B, K, M, D)
    picked = picked * node_mask[..., None].to(picked.dtype)
    cnt = node_mask.sum(dim=2, keepdim=True).clamp(min=1)
    if reduce == "mean":
        pooled = picked.sum(dim=2) / cnt
    elif reduce == "max":
        neg_inf = torch.finfo(picked.dtype).min
        masked = picked.masked_fill(~node_mask[..., None], neg_inf)
        pooled = masked.max(dim=2).values
    else:
        raise ValueError("reduce must be mean/max")
    return pooled

def pool_region_lines(encoded_lines, region_line_indices, region_line_mask):
    B, L, D = encoded_lines.shape
    _, K, l = region_line_indices.shape
    safe_idx = torch.where(region_line_mask, region_line_indices, torch.zeros_like(region_line_indices))
    safe_idx = safe_idx.clamp(min=0, max=L-1)
    b_idx = torch.arange(B, device=encoded_lines.device)[:, None, None].expand(B, K, l)
    gathered = encoded_lines[b_idx, safe_idx]
    gathered = gathered * region_line_mask[..., None].to(gathered.dtype)
    cnt = region_line_mask.sum(dim=2, keepdim=True).clamp(min=1)
    return gathered.sum(dim=2) / cnt

class VulDetectionModel(nn.Module):
    """Fuse regional views and use learned importance to classify functions."""
    def __init__(self, hidden_dim, input_dim, heads=4, ggnn_steps=6, attr_dim=RDP_DIM,
                 disabled_views=(), use_rdp=True, pooling="weighted"):
        super().__init__()
        valid_views = {"ast", "cfg", "pdg", "sem"}
        disabled_views = set(disabled_views)
        if not disabled_views <= valid_views:
            raise ValueError("Unknown disabled view")
        if pooling not in {"weighted", "mean", "gated_attention"}:
            raise ValueError("Unknown pooling mode")
        self.enabled_views = valid_views - disabled_views
        self.use_rdp = bool(use_rdp)
        self.pooling = pooling
        self.hidden_dim = hidden_dim
        self.attr_dim = int(attr_dim)
        if self.use_rdp and self.attr_dim != RDP_DIM:
            raise ValueError("REGCAP requires the canonical 31-feature RDP")
        if not self.enabled_views and not self.use_rdp:
            raise ValueError("At least one region view is required")
        self.ast_enc = HierAST(input_dim, hidden_dim, hidden_dim) if "ast" in self.enabled_views else None
        self.cfg_enc = GGNNEncoder(input_dim, hidden_dim, ggnn_steps, hidden_dim) if "cfg" in self.enabled_views else None
        self.pdg_enc = GATEncoder(input_dim, hidden_dim, heads, hidden_dim) if "pdg" in self.enabled_views else None
        if "sem" in self.enabled_views:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=4 * hidden_dim,
                dropout=0.2, activation="gelu", batch_first=True
            )
            self.line_masker = AutoMaskGenerator()
            self.sem_enc = nn.TransformerEncoder(enc_layer, num_layers=3)
        else:
            self.line_masker = None
            self.sem_enc = None
        if self.use_rdp:
            if self.attr_dim < 1:
                raise ValueError("RDP feature count must be positive")
            self.attr_enc = nn.Sequential(
                nn.Linear(self.attr_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
            )
        else:
            self.attr_enc = None
        self.view_order = tuple(name for name in ("token", "ast", "cfg", "pdg", "prior")
                                if (name == "token" and "sem" in self.enabled_views)
                                or (name in self.enabled_views)
                                or (name == "prior" and self.use_rdp))
        feature_count = len(self.view_order)
        self.view_projections = nn.ModuleDict({
            name: nn.Linear(hidden_dim, hidden_dim) for name in self.view_order
        })
        self.view_gate = nn.Linear(hidden_dim * feature_count, feature_count)
        self.region_fuse = nn.Sequential(
            nn.Linear(hidden_dim * feature_count, hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.region_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        if pooling == "gated_attention":
            self.gate_v = nn.Linear(hidden_dim, hidden_dim)
            self.gate_u = nn.Linear(hidden_dim, hidden_dim)
            self.gate_w = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        region_mask = self._get_region_mask(data)
        view_embeddings = {}
        if self.ast_enc is not None:
            ast_h = self.ast_enc(data.ast_x, data.token_edge_index,
                                 data.stmt_edge_index, data.block_edge_index)
            view_embeddings["ast"] = masked_region_pool(
                ast_h, data.region_ast_nodes, data.region_ast_node_mask)
        if self.cfg_enc is not None:
            cfg_h = self.cfg_enc(data.cfg_x, data.cfg_edge_index)
            view_embeddings["cfg"] = masked_region_pool(
                cfg_h, data.region_cfg_nodes, data.region_cfg_node_mask)
        if self.pdg_enc is not None:
            pdg_h = self.pdg_enc(data.pdg_x, data.pdg_edge_index)
            view_embeddings["pdg"] = masked_region_pool(
                pdg_h, data.region_pdg_nodes, data.region_pdg_node_mask)
        if self.sem_enc is not None:
            lines = data.global_code_embeddings
            sem = self.sem_enc(lines, src_key_padding_mask=self.line_masker(lines))
            view_embeddings["token"] = pool_region_lines(
                sem, data.region_line_numbers, data.region_line_mask)
        if self.attr_enc is not None:
            attrs = data.region_attr
            if attrs.ndim != 3 or attrs.size(-1) != self.attr_dim:
                raise ValueError("region_attr dimension differs from attr_dim")
            view_embeddings["prior"] = self.attr_enc(attrs) * region_mask.unsqueeze(-1)
        projected = [self.view_projections[name](view_embeddings[name])
                     for name in self.view_order]
        gate_input = torch.cat(projected, dim=-1)
        view_gate_weights = F.softmax(self.view_gate(gate_input), dim=-1)
        gated_views = torch.cat([
            weight * representation
            for weight, representation in zip(view_gate_weights.split(1, dim=-1), projected)
        ], dim=-1)
        region_feat = self.region_fuse(gated_views)
        region_logits = self.region_scorer(region_feat).squeeze(-1)
        region_probs = torch.sigmoid(region_logits)
        graph_feat = self._pool(region_feat, region_probs, region_mask)
        cls_logits = self.graph_classifier(graph_feat).squeeze(-1)
        return {
            "region_logits": region_logits,
            "region_mask": region_mask,
            "region_feat": region_feat,
            "view_gate_weights": view_gate_weights,
            "graph_feat": graph_feat,
            "cls_logits": cls_logits,
        }

    @staticmethod
    def _get_region_mask(data):
        for name in ("region_ast_region_mask", "region_cfg_region_mask",
                     "region_pdg_region_mask", "region_attr_mask"):
            mask = getattr(data, name, None)
            if mask is not None:
                return mask.bool()
        raise ValueError("Missing region mask")

    def _pool(self, region_feat, region_probs, region_mask):
        mask = region_mask.to(region_feat.dtype)
        if self.pooling == "weighted":
            weight = region_probs * mask
        elif self.pooling == "mean":
            weight = mask
        else:
            gate = torch.tanh(self.gate_v(region_feat))
            gate = gate * torch.sigmoid(self.gate_u(region_feat))
            logits = self.gate_w(gate).squeeze(-1).masked_fill(~region_mask, -1e9)
            weight = torch.softmax(logits, dim=1) * mask
        denominator = weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (region_feat * weight.unsqueeze(-1)).sum(dim=1) / denominator
