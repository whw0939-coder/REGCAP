"""Batch processed function graphs and aligned regional data."""

import os
import torch
import random
import pickle
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch_geometric.data import Batch
from typing import Optional, List
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import json

from .rdp_schema import RDP_DIM, RDP_FEATURES

class GlobalDataModule(pl.LightningDataModule):
    """Load processed samples and align every view to retained region slots."""
    def __init__(
            self,
            data_path: str,
            batch_size: int = 32,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            max_regions: int = 17,
            emb_dim: int = 128,
            seed: int = 42,
            cwe_mode: bool = False,
            max_global_lines: int = 4096,
            drop_long_global: bool = True,
            save_lists_dir: Optional[str] = None,
            cross_data_path: Optional[str] = None,
            cross_test_ratio: float = 1.0,

            train_data_path: Optional[str] = None,
            val_data_path: Optional[str] = None,
            test_data_path: Optional[str] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.max_regions = max_regions
        self.emb_dim = emb_dim
        self.seed = seed
        self.cwe_mode    = cwe_mode

        self.max_global_lines = max_global_lines
        self.drop_long_global = drop_long_global

        self.global_train = None
        self.global_val = None
        self.global_test = None
        self.contrast_train = None
        self.contrast_val = None

        self.save_lists_dir = save_lists_dir

        self.train_file_list: List[str] = []
        self.val_file_list: List[str] = []
        self.test_file_list: List[str] = []

        self.cross_data_path = cross_data_path
        self.cross_test_ratio = cross_test_ratio

        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path

    def _use_pre_split_mode(self) -> bool:
        return any([
            self.train_data_path is not None,
            self.val_data_path is not None,
            self.test_data_path is not None,
        ])

    def get_train_filenames(self) -> List[str]:
        return list(self.train_file_list)

    def get_val_filenames(self) -> List[str]:
        return list(self.val_file_list)

    def get_test_filenames(self) -> List[str]:
        return list(self.test_file_list)

    def prepare_data(self):

        if self._use_pre_split_mode():
            if self.cwe_mode:
                raise ValueError("pre-split 模式不能与 cwe_mode 同时使用。")

            if self.cross_data_path is not None:
                raise ValueError("pre-split 模式不能与 cross_data_path 同时使用。")

            if not self.train_data_path or not self.val_data_path or not self.test_data_path:
                raise ValueError("使用官方划分模式时，必须同时提供 train_data_path / val_data_path / test_data_path。")

            for p, name in [
                (self.train_data_path, "train_data_path"),
                (self.val_data_path, "val_data_path"),
                (self.test_data_path, "test_data_path"),
            ]:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"{name} 不存在: {p}")

            return

        if self.cwe_mode:
            if not os.path.isdir(self.data_path):
                raise FileNotFoundError(f"CWE 模式下，请指定包含多个 .pkl 的目录，而不是单文件：{self.data_path}")
        else:
            if not os.path.isfile(self.data_path):
                raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        if self.cross_data_path is not None:
            if not os.path.isfile(self.cross_data_path):
                raise FileNotFoundError(f"跨数据集 测试数据文件不存在: {self.cross_data_path}")

    def setup(self, stage: str = None):

        if self._use_pre_split_mode():
            with open(self.train_data_path, "rb") as f:
                self.global_train = pickle.load(f)
            with open(self.val_data_path, "rb") as f:
                self.global_val = pickle.load(f)
            with open(self.test_data_path, "rb") as f:
                self.global_test = pickle.load(f)

            rank_zero_info(
                f"[PreSplit] train={len(self.global_train)}  "
                f"val={len(self.global_val)}  test={len(self.global_test)}"
            )

            self.train_file_list = self._extract_filenames(self.global_train)
            self.val_file_list = self._extract_filenames(self.global_val)
            self.test_file_list = self._extract_filenames(self.global_test)

            if self.save_lists_dir:
                self._save_split_lists(self.save_lists_dir)
            return

        if self.cwe_mode:
            if self.cross_data_path is not None:
                raise ValueError("当前实现不支持 cwe_mode 与 cross_data_path 同时使用。请只启用其一。")

            train_list, val_list, test_list = [], [], []
            g = torch.Generator().manual_seed(self.seed)

            for fname in sorted(os.listdir(self.data_path)):
                rank_zero_info(f"Loading {fname}")
                if not fname.endswith(".pkl"):
                    continue
                full_path = os.path.join(self.data_path, fname)
                with open(full_path, "rb") as f:
                    data_i = pickle.load(f)

                n = len(data_i)
                n_train = int(self.train_ratio * n)
                n_val = int(self.val_ratio * n)
                n_test = n - n_train - n_val

                subt = random_split(data_i, [n_train, n_val, n_test], generator=g)
                train_list.extend([subt[0].dataset[i] for i in subt[0].indices])
                val_list.extend([subt[1].dataset[i] for i in subt[1].indices])
                test_list.extend([subt[2].dataset[i] for i in subt[2].indices])

            self.global_train = train_list
            self.global_val = val_list
            self.global_test = test_list

            rank_zero_info(
                f"[CWE] train={len(self.global_train)}  val={len(self.global_val)}  test={len(self.global_test)}")

        else:

            with open(self.data_path, "rb") as f:
                dbd_full = pickle.load(f)

            if self.cross_data_path is not None:

                total = len(dbd_full)
                n_train = int(self.train_ratio * total)
                n_val = total - n_train
                g = torch.Generator().manual_seed(self.seed)
                dbd_train, dbd_val = random_split(dbd_full, [n_train, n_val], generator=g)
                self.global_train, self.global_val = dbd_train, dbd_val

                with open(self.cross_data_path, "rb") as f:
                    reveal_full = pickle.load(f)

                n_reveal = len(reveal_full)
                n_test = max(1, int(round(self.cross_test_ratio * n_reveal)))
                rng = random.Random(self.seed)
                idxs = list(range(n_reveal))
                rng.shuffle(idxs)
                pick = idxs[:n_test]
                self.global_test = [reveal_full[i] for i in pick]

                rank_zero_info(f"[DBD] train={len(self.global_train)}  val={len(self.global_val)}")
                rank_zero_info(f"[Reveal] total={n_reveal}  sampled_test={len(self.global_test)} "
                               f"(ratio={self.cross_test_ratio})")
            else:

                self._safe_data_split(dbd_full)
                rank_zero_info(
                    f"[Single] train={len(self.global_train)}  val={len(self.global_val)}  test={len(self.global_test)}")

        self.train_file_list = self._extract_filenames(self.global_train)
        self.val_file_list = self._extract_filenames(self.global_val)
        self.test_file_list = self._extract_filenames(self.global_test)

        if self.save_lists_dir:
            self._save_split_lists(self.save_lists_dir)

    def _safe_data_split(self, full_dataset):

        total = len(full_dataset)
        train_size = int(self.train_ratio * total)
        val_size = int(self.val_ratio * total)
        test_size = total - train_size - val_size

        generator = torch.Generator().manual_seed(self.seed)
        self.global_train, self.global_val, self.global_test = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

    def _extract_filenames(self, subset) -> List[str]:

        if hasattr(subset, "indices") and hasattr(subset, "dataset"):
            items = [subset.dataset[i] for i in subset.indices]
        else:
            items = list(subset) if subset is not None else []

        names = []
        for s in items:
            fn = getattr(s, "file_name", None)
            if fn is not None:
                names.append(str(fn))
        return names

    def _save_split_lists(self, out_dir: str):

        os.makedirs(out_dir, exist_ok=True)

        payload = {
            "train": self.train_file_list,
            "val": self.val_file_list,
            "test": self.test_file_list,
        }

        for split, lst in payload.items():
            with open(os.path.join(out_dir, f"{split}_filenames.txt"), "w", encoding="utf-8") as f:
                for fn in lst:
                    f.write(fn + "\n")

        with open(os.path.join(out_dir, "split_filenames.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        rank_zero_info(f"Saved filename lists to: {out_dir}")

    def _create_weighted_sampler(self, subset):
        labels = [1 if p.y == 1 else 0 for p in subset]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        sample_weights = [class_weights[label] for label in labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights))

    def train_dataloader(self):
        return DataLoader(
            self.global_train,
            batch_size=self.batch_size,
            sampler=self._create_weighted_sampler(self.global_train),
            shuffle=False,
            collate_fn=lambda batch: self._classification_collate(batch, split="train"),
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.global_val,
            batch_size=self.batch_size,
            collate_fn=lambda batch: self._classification_collate(batch, split="val"),
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.global_test,
            batch_size=self.batch_size,
            collate_fn=lambda batch: self._classification_collate(batch, split="test"),
            num_workers=0
        )

    def _classification_collate(self, batch, split="train"):
        try:
            return self._build_cls_batch(batch, split=split)
        except Exception as e:
            print(f"Collate error: {str(e)}")
            torch.save(batch, "error_batch.pt")
            raise

    def _select_region_indices_uniform(self, num_regions: int, max_regions: int):

        if max_regions is None or max_regions <= 0 or num_regions <= max_regions:
            return list(range(num_regions))

        idx = torch.linspace(0, num_regions - 1, steps=max_regions)
        idx = torch.round(idx).long().tolist()

        selected = []
        seen = set()
        for i in idx:
            if i not in seen:
                selected.append(i)
                seen.add(i)

        if len(selected) < max_regions:
            for i in range(num_regions):
                if i not in seen:
                    selected.append(i)
                    seen.add(i)
                if len(selected) == max_regions:
                    break

        return selected

    def _select_regions_for_sample(self, sample, selected_idx):

        def _select_list(lst):
            if lst is None:
                return []
            return [lst[i] for i in selected_idx if i < len(lst)]

        def _select_tensor_rows(t):
            if t is None or t.numel() == 0:
                return torch.zeros((0,), dtype=torch.float32)
            idx = [i for i in selected_idx if i < t.size(0)]
            if len(idx) == 0:
                shape = list(t.shape)
                shape[0] = 0
                return t.new_zeros(shape)
            idx_t = torch.tensor(idx, dtype=torch.long)
            return t.index_select(0, idx_t)

        out = {
            "ast_regions": _select_list(getattr(sample, "region_ast_nodes_list", [])),
            "cfg_regions": _select_list(getattr(sample, "region_cfg_nodes_list", [])),
            "pdg_regions": _select_list(getattr(sample, "region_pdg_nodes_list", [])),
            "line_regions": _select_list(getattr(sample, "region_line_numbers_lists", [])),
            "region_score": _select_tensor_rows(getattr(sample, "region_score_gt", None)),
            "region_attr": _select_tensor_rows(getattr(sample, "region_attr", None)),
            "selected_idx": selected_idx,
        }
        return out

    def _build_cls_batch(self, samples, split="train"):
        max_regions = int(self.max_regions)
        d = int(self.emb_dim)

        drop_long = getattr(self, "drop_long_global", True)
        L_max = int(getattr(self, "max_global_lines", 512))

        if drop_long:
            kept = []
            for s in samples:
                num_lines = int(s.global_code_embedding.size(0)) if hasattr(s, "global_code_embedding") else 0
                if num_lines <= L_max:
                    kept.append(s)
            samples = kept
            if len(samples) == 0:

                return Batch(
                    ast_x=torch.empty((0, d)), num_nodes=0,
                    token_edge_index=torch.empty((2, 0), dtype=torch.long),
                    stmt_edge_index=torch.empty((2, 0), dtype=torch.long),
                    block_edge_index=torch.empty((2, 0), dtype=torch.long),
                    ast_batch=torch.empty((0,), dtype=torch.long),
                    y=torch.empty((0,), dtype=torch.long),
                    file_name=[],

                    global_code_embeddings=torch.empty((0, 0, d)),
                    global_emb_mask=torch.empty((0, 0), dtype=torch.bool),

                    region_ast_nodes=torch.empty((0, max_regions, 0), dtype=torch.long),
                    region_ast_region_mask=torch.empty((0, max_regions), dtype=torch.bool),
                    region_ast_node_mask=torch.empty((0, max_regions, 0), dtype=torch.bool),

                    cfg_x=torch.empty((0, d)), cfg_batch=torch.empty((0,), dtype=torch.long),
                    cfg_edge_index=torch.empty((2, 0), dtype=torch.long),
                    region_cfg_nodes=torch.empty((0, max_regions, 0), dtype=torch.long),
                    region_cfg_region_mask=torch.empty((0, max_regions), dtype=torch.bool),
                    region_cfg_node_mask=torch.empty((0, max_regions, 0), dtype=torch.bool),

                    pdg_x=torch.empty((0, d)), pdg_batch=torch.empty((0,), dtype=torch.long),
                    pdg_edge_index=torch.empty((2, 0), dtype=torch.long),
                    region_pdg_nodes=torch.empty((0, max_regions, 0), dtype=torch.long),
                    region_pdg_region_mask=torch.empty((0, max_regions), dtype=torch.bool),
                    region_pdg_node_mask=torch.empty((0, max_regions, 0), dtype=torch.bool),

                    region_line_numbers=torch.empty((0, max_regions, 0), dtype=torch.long),
                    region_line_mask=torch.empty((0, max_regions, 0), dtype=torch.bool),
                    region_score=torch.empty((0, max_regions)),

                    region_attr=torch.empty((0, max_regions, RDP_DIM), dtype=torch.float32),
                    region_attr_mask=torch.empty((0, max_regions), dtype=torch.bool),
                )

        selected_region_data = []
        for s in samples:

            num_regions = len(getattr(s, "region_ast_nodes_list", []))

            selected_idx = self._select_region_indices_uniform(
                num_regions=num_regions,
                max_regions=max_regions
            )

            selected = self._select_regions_for_sample(s, selected_idx)
            selected_region_data.append(selected)

        for sample in samples:
            if tuple(getattr(sample, "region_attr_names", ())) != RDP_FEATURES:
                raise ValueError("Processed sample uses a different RDP schema")
            if sample.region_attr.ndim != 2 or sample.region_attr.size(1) != RDP_DIM:
                raise ValueError("Processed sample must contain 31 RDP columns")

        B = len(samples)

        max_ast_region_nodes = 0
        max_cfg_region_nodes = 0
        max_pdg_region_nodes = 0
        max_region_lines = 0

        for sel in selected_region_data:
            for t in sel["ast_regions"]:
                max_ast_region_nodes = max(max_ast_region_nodes, int(t.size(0)))
            for t in sel["cfg_regions"]:
                max_cfg_region_nodes = max(max_cfg_region_nodes, int(t.size(0)))
            for t in sel["pdg_regions"]:
                max_pdg_region_nodes = max(max_pdg_region_nodes, int(t.size(0)))
            for t in sel["line_regions"]:
                max_region_lines = max(max_region_lines, int(t.size(0)))

        ast_x_list = []
        token_eis, stmt_eis, block_eis = [], [], []
        ast_batch_list = []

        cfg_x_list, cfg_eis, cfg_batch_list = [], [], []
        pdg_x_list, pdg_eis, pdg_batch_list = [], [], []

        ast_region_lists = []
        cfg_region_lists = []
        pdg_region_lists = []
        region_line_lists = []
        region_score_lists = []
        file_names = []
        ys = []

        ast_node_offset = 0
        cfg_node_offset = 0
        pdg_node_offset = 0

        for i, s in enumerate(samples):

            ast_x_list.append(s.ast_x)
            N_ast = int(s.ast_x.size(0))

            if s.token_edge_index.numel() > 0:
                token_eis.append(s.token_edge_index + ast_node_offset)
            if s.stmt_edge_index.numel() > 0:
                stmt_eis.append(s.stmt_edge_index + ast_node_offset)
            if s.block_edge_index.numel() > 0:
                block_eis.append(s.block_edge_index + ast_node_offset)

            ast_batch_list.append(torch.full((N_ast,), i, dtype=torch.long))
            ast_node_offset += N_ast

            if hasattr(s, "cfg_x") and s.cfg_x is not None:
                x = s.cfg_x
                ei = s.cfg_edge_index
                cfg_x_list.append(x)
                cfg_eis.append(ei + cfg_node_offset if ei.numel() > 0 else ei)
                cfg_batch_list.append(torch.full((x.size(0),), i, dtype=torch.long))
                cfg_node_offset += int(x.size(0))
            else:
                cfg_x_list.append(torch.empty((0, d)))
                cfg_eis.append(torch.empty((2, 0), dtype=torch.long))
                cfg_batch_list.append(torch.empty((0,), dtype=torch.long))

            if hasattr(s, "pdg_x") and s.pdg_x is not None:
                x = s.pdg_x
                ei = s.pdg_edge_index
                pdg_x_list.append(x)
                pdg_eis.append(ei + pdg_node_offset if ei.numel() > 0 else ei)
                pdg_batch_list.append(torch.full((x.size(0),), i, dtype=torch.long))
                pdg_node_offset += int(x.size(0))
            else:
                pdg_x_list.append(torch.empty((0, d)))
                pdg_eis.append(torch.empty((2, 0), dtype=torch.long))
                pdg_batch_list.append(torch.empty((0,), dtype=torch.long))

            sel = selected_region_data[i]

            ast_region_lists.append(sel["ast_regions"])
            cfg_region_lists.append(sel["cfg_regions"])
            pdg_region_lists.append(sel["pdg_regions"])
            region_line_lists.append(sel["line_regions"])

            region_score_lists.append(sel["region_score"])

            ys.append(s.y)
            file_names.append(s.file_name)

        ast_x = torch.cat(ast_x_list, dim=0) if len(ast_x_list) > 0 else torch.empty((0, d))
        token_edge_index = (torch.cat(token_eis, dim=1)
                            if len(token_eis) > 0 else torch.empty((2, 0), dtype=torch.long))
        stmt_edge_index = (torch.cat(stmt_eis, dim=1)
                           if len(stmt_eis) > 0 else torch.empty((2, 0), dtype=torch.long))
        block_edge_index = (torch.cat(block_eis, dim=1)
                            if len(block_eis) > 0 else torch.empty((2, 0), dtype=torch.long))
        ast_batch = torch.cat(ast_batch_list, dim=0) if len(ast_batch_list) > 0 else torch.empty((0,), dtype=torch.long)

        cfg_x = torch.cat(cfg_x_list, dim=0) if len(cfg_x_list) > 0 else torch.empty((0, d))
        cfg_edge_index = (torch.cat(cfg_eis, dim=1)
                          if len(cfg_eis) > 0 else torch.empty((2, 0), dtype=torch.long))
        cfg_batch = torch.cat(cfg_batch_list, dim=0) if len(cfg_batch_list) > 0 else torch.empty((0,), dtype=torch.long)

        pdg_x = torch.cat(pdg_x_list, dim=0) if len(pdg_x_list) > 0 else torch.empty((0, d))
        pdg_edge_index = (torch.cat(pdg_eis, dim=1)
                          if len(pdg_eis) > 0 else torch.empty((2, 0), dtype=torch.long))
        pdg_batch = torch.cat(pdg_batch_list, dim=0) if len(pdg_batch_list) > 0 else torch.empty((0,), dtype=torch.long)

        region_score = torch.zeros((B, max_regions), dtype=torch.float32)
        for i, s in enumerate(region_score_lists):
            L = min(int(s.size(0)), max_regions)
            if L > 0:
                region_score[i, :L] = s[:L]

        ast_offsets = []
        c = 0
        for s in samples:
            ast_offsets.append(c)
            c += int(s.ast_x.size(0))

        cfg_offsets, pdg_offsets = [], []
        c = 0
        for s in samples:
            cfg_offsets.append(c)
            c += int(s.cfg_x.size(0)) if hasattr(s, "cfg_x") and s.cfg_x is not None else 0
        c = 0
        for s in samples:
            pdg_offsets.append(c)
            c += int(s.pdg_x.size(0)) if hasattr(s, "pdg_x") and s.pdg_x is not None else 0

        ast_region_nodes, ast_region_mask, ast_node_mask = self._pad_2d_indices(
            [ast_region_lists[i] for i in range(B)],
            max_regions=max_regions,
            pad_nodes=max_ast_region_nodes if max_ast_region_nodes > 0 else 0,
            offset=0
        )
        cfg_region_nodes, cfg_region_mask, cfg_node_mask = self._pad_2d_indices(
            [cfg_region_lists[i] for i in range(B)],
            max_regions=max_regions,
            pad_nodes=max_cfg_region_nodes if max_cfg_region_nodes > 0 else 0,
            offset=0
        )
        pdg_region_nodes, pdg_region_mask, pdg_node_mask = self._pad_2d_indices(
            [pdg_region_lists[i] for i in range(B)],
            max_regions=max_regions,
            pad_nodes=max_pdg_region_nodes if max_pdg_region_nodes > 0 else 0,
            offset=0
        )

        for i in range(B):
            if ast_region_nodes.size(2) > 0 and ast_region_mask[i].any():
                ast_region_nodes[i, ast_region_mask[i]] += ast_offsets[i]
            if cfg_region_nodes.size(2) > 0 and cfg_region_mask[i].any():
                cfg_region_nodes[i, cfg_region_mask[i]] += cfg_offsets[i]
            if pdg_region_nodes.size(2) > 0 and pdg_region_mask[i].any():
                pdg_region_nodes[i, pdg_region_mask[i]] += pdg_offsets[i]

        F = RDP_DIM
        region_attr = torch.zeros((B, max_regions, F), dtype=torch.float32)

        region_attr_mask = ast_region_mask.clone() if ast_region_mask.numel() > 0 else torch.empty((B, max_regions),
                                                                                                   dtype=torch.bool)

        for i, sel in enumerate(selected_region_data):
            attr_i = sel["region_attr"]
            if attr_i is None or attr_i.numel() == 0:
                continue

            Ri = int(attr_i.size(0))
            Fi = int(attr_i.size(1)) if attr_i.ndim == 2 else 0
            L = min(Ri, max_regions)
            C = min(F, Fi)
            if L > 0 and C > 0:
                region_attr[i, :L, :C] = attr_i[:L, :C]

        region_line_numbers, region_line_region_mask, region_line_mask = self._pad_2d_lines(
            [region_line_lists[i] for i in range(B)],
            max_regions=max_regions,
            pad_lines=max_region_lines if max_region_lines > 0 else 0
        )

        global_embeddings, global_mask = self._pad_global_embeddings(
            [s.global_code_embedding for s in samples],
            pad_lines=L_max,
            d=d
        )
        line_limits = global_mask.sum(dim=1)[:, None, None]
        region_line_mask &= region_line_numbers < line_limits
        region_line_numbers = region_line_numbers.masked_fill(~region_line_mask, 0)

        y = torch.cat(ys, dim=0) if len(ys) > 0 else torch.empty((0,), dtype=torch.long)

        batch = Batch(

            ast_x=ast_x,
            num_nodes=ast_x.size(0),
            token_edge_index=token_edge_index,
            stmt_edge_index=stmt_edge_index,
            block_edge_index=block_edge_index,
            ast_batch=ast_batch,

            y=y,
            file_name=file_names,

            global_code_embeddings=global_embeddings,
            global_emb_mask=global_mask,

            region_ast_nodes=ast_region_nodes,
            region_ast_region_mask=ast_region_mask,
            region_ast_node_mask=ast_node_mask,

            cfg_x=cfg_x,
            cfg_edge_index=cfg_edge_index,
            cfg_batch=cfg_batch,
            region_cfg_nodes=cfg_region_nodes,
            region_cfg_region_mask=cfg_region_mask,
            region_cfg_node_mask=cfg_node_mask,

            pdg_x=pdg_x,
            pdg_edge_index=pdg_edge_index,
            pdg_batch=pdg_batch,
            region_pdg_nodes=pdg_region_nodes,
            region_pdg_region_mask=pdg_region_mask,
            region_pdg_node_mask=pdg_node_mask,

            region_line_numbers=region_line_numbers,
            region_line_mask=region_line_mask,
            region_score=region_score,

            region_attr=region_attr,
            region_attr_mask=region_attr_mask,

        )
        batch.num_graphs = batch.y.size(0)
        return batch

    def _pad_2d_indices(self, lists_per_sample, max_regions, pad_nodes, offset=0):
        B = len(lists_per_sample)
        idx_tensor = torch.zeros((B, max_regions, pad_nodes), dtype=torch.long)
        region_mask = torch.zeros((B, max_regions), dtype=torch.bool)
        node_mask = torch.zeros((B, max_regions, pad_nodes), dtype=torch.bool)

        for i, region_list in enumerate(lists_per_sample):

            assert len(region_list) <= max_regions, \
                f"region_list length {len(region_list)} exceeds max_regions={max_regions}"

            for r, ids in enumerate(region_list):
                if isinstance(ids, torch.Tensor):
                    ids = ids.detach().cpu().tolist()
                region_mask[i, r] = True
                length = min(len(ids), pad_nodes)
                if length > 0:
                    idx_tensor[i, r, :length] = torch.tensor(ids[:length], dtype=torch.long) + offset
                    node_mask[i, r, :length] = True
        return idx_tensor, region_mask, node_mask

    def _pad_2d_lines(self, lists_per_sample, max_regions, pad_lines):
        B = len(lists_per_sample)
        line_tensor = torch.zeros((B, max_regions, pad_lines), dtype=torch.long)
        region_mask = torch.zeros((B, max_regions), dtype=torch.bool)
        line_mask = torch.zeros((B, max_regions, pad_lines), dtype=torch.bool)

        for i, region_list in enumerate(lists_per_sample):
            assert len(region_list) <= max_regions, \
                f"region_list length {len(region_list)} exceeds max_regions={max_regions}"

            for r, lines in enumerate(region_list):
                if isinstance(lines, torch.Tensor):
                    lines = lines.detach().cpu().tolist()
                region_mask[i, r] = True
                length = min(len(lines), pad_lines)
                if length > 0:
                    source_lines = torch.tensor(lines[:length], dtype=torch.long)
                    valid = source_lines > 0
                    line_tensor[i, r, :length] = (source_lines - 1).clamp_min(0)
                    line_mask[i, r, :length] = valid
        return line_tensor, region_mask, line_mask

    def _pad_global_embeddings(self, emb_list, pad_lines, d):

        B = len(emb_list)
        out = torch.zeros((B, pad_lines, d), dtype=emb_list[0].dtype if B > 0 else torch.float32)
        mask = torch.zeros((B, pad_lines), dtype=torch.bool)
        for i, e in enumerate(emb_list):
            if e is None or e.numel() == 0:
                continue
            L = min(int(e.size(0)), pad_lines)
            out[i, :L] = e[:L]
            mask[i, :L] = True
        return out, mask
