"""Joern graph preprocessing and canonical regional feature extraction."""

import os
import gensim
from nltk.tokenize import RegexpTokenizer
import logging
import torch
import json
from torch_geometric.data import Data
import traceback
import pickle
import re
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F
from typing import List
from collections import defaultdict
import math
from collections import deque

import warnings

from .rdp_schema import RDP_DIM, RDP_FEATURES
from .regions import construct_regions, weak_region_target

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegionFeatureBuilderMixin:
    """Extract the canonical 31-dimensional RDP from each AST-guided region."""

    def _init_region_attr_stats(self):
        return {
            "feature_names": None,
            "total_files": 0,
            "total_regions": 0,
            "sum": None,
            "sumsq": None,
            "min": None,
            "max": None,
            "nonzero": None,
            "eps": 1e-12,
            "per_file": []
        }

    def _update_region_attr_stats(self, stats, region_attr_tensor, feature_names, file_name):

        if region_attr_tensor is None:
            return

        arr = region_attr_tensor.detach().cpu().numpy()
        if arr.ndim == 1:
            arr = arr[None, :]

        if arr.shape[0] == 0:
            return

        F = arr.shape[1]
        eps = stats["eps"]

        if stats["feature_names"] is None:
            stats["feature_names"] = list(feature_names)
            stats["sum"] = np.zeros(F, dtype=np.float64)
            stats["sumsq"] = np.zeros(F, dtype=np.float64)
            stats["min"] = np.full(F, np.inf, dtype=np.float64)
            stats["max"] = np.full(F, -np.inf, dtype=np.float64)
            stats["nonzero"] = np.zeros(F, dtype=np.int64)
        else:
            if len(feature_names) != len(stats["feature_names"]):
                raise ValueError(
                    f"region_attr feature dim mismatch: current={len(feature_names)}, "
                    f"expected={len(stats['feature_names'])}"
                )

        stats["total_files"] += 1
        stats["total_regions"] += arr.shape[0]

        stats["sum"] += arr.sum(axis=0)
        stats["sumsq"] += np.square(arr).sum(axis=0)
        stats["min"] = np.minimum(stats["min"], arr.min(axis=0))
        stats["max"] = np.maximum(stats["max"], arr.max(axis=0))
        stats["nonzero"] += (np.abs(arr) > eps).sum(axis=0)

        file_mean = arr.mean(axis=0)
        file_max = arr.max(axis=0)
        file_min = arr.min(axis=0)
        file_nonzero_ratio = (np.abs(arr) > eps).mean(axis=0)

        stats["per_file"].append({
            "file_name": file_name,
            "num_regions": int(arr.shape[0]),
            "feature_mean": {feature_names[i]: float(file_mean[i]) for i in range(F)},
            "feature_min": {feature_names[i]: float(file_min[i]) for i in range(F)},
            "feature_max": {feature_names[i]: float(file_max[i]) for i in range(F)},
            "feature_nonzero_ratio": {feature_names[i]: float(file_nonzero_ratio[i]) for i in range(F)},
        })

    def _finalize_region_attr_stats(self, stats, near_zero_ratio_threshold=0.001):

        if stats["feature_names"] is None or stats["total_regions"] == 0:
            return {
                "total_files": 0,
                "total_regions": 0,
                "feature_summary": [],
                "all_zero_dims": [],
                "near_zero_dims": []
            }

        names = stats["feature_names"]
        total_regions = stats["total_regions"]

        mean = stats["sum"] / total_regions
        var = stats["sumsq"] / total_regions - np.square(mean)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)

        nonzero_ratio = stats["nonzero"] / total_regions

        feature_summary = []
        all_zero_dims = []
        near_zero_dims = []

        for i, name in enumerate(names):
            item = {
                "index": i,
                "name": name,
                "mean": float(mean[i]),
                "std": float(std[i]),
                "min": float(stats["min"][i]),
                "max": float(stats["max"][i]),
                "nonzero_count": int(stats["nonzero"][i]),
                "nonzero_ratio": float(nonzero_ratio[i]),
            }
            feature_summary.append(item)

            if stats["nonzero"][i] == 0:
                all_zero_dims.append({
                    "index": i,
                    "name": name
                })

            if nonzero_ratio[i] < near_zero_ratio_threshold:
                near_zero_dims.append({
                    "index": i,
                    "name": name,
                    "nonzero_ratio": float(nonzero_ratio[i])
                })

        report = {
            "total_files": int(stats["total_files"]),
            "total_regions": int(total_regions),
            "feature_dim": int(len(names)),
            "feature_names": names,
            "feature_summary": feature_summary,
            "all_zero_dims": all_zero_dims,
            "near_zero_dims": near_zero_dims,
            "per_file": stats["per_file"]
        }
        return report

    def _save_region_attr_stats_report(self, report):
        save_name = os.path.join(getattr(self, "output_dir", "."), f"{self.dataset_name}-region_attr_stats.json")
        with open(save_name, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n================ Region Attr Stats Summary ================")
        print(f"Processed files   : {report['total_files']}")
        print(f"Processed regions : {report['total_regions']}")
        print(f"Feature dim       : {report['feature_dim']}")
        print(f"All-zero dims     : {[x['name'] for x in report['all_zero_dims']]}")
        print(f"Near-zero dims    : {[x['name'] for x in report['near_zero_dims']]}")
        print(f"Saved report to   : {save_name}")
        print("==========================================================\n")

    def _is_param_node(self, n):
        return self._node_label(n) in ("METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT")

    def _build_adj(self, edges):
        adj = defaultdict(list)
        for e in edges:
            if len(e) >= 2:
                u, v = e[0], e[1]
                adj[u].append(v)
        return adj

    def _build_pred(self, edges):
        pred = defaultdict(list)
        for e in edges:
            if len(e) >= 2:
                u, v = e[0], e[1]
                pred[v].append(u)
        return pred

    def _count_internal_edges(self, nodes, adj):
        node_set = set(nodes)
        cnt = 0
        for u in nodes:
            for v in adj.get(u, []):
                if v in node_set:
                    cnt += 1
        return cnt

    def _reachable_within_region(self, starts, region_nodes, adj, max_steps=None):
        node_set = set(region_nodes)
        q = deque()
        visited = set()

        for s in starts:
            if s in node_set:
                q.append((s, 0))
                visited.add(s)

        while q:
            u, d = q.popleft()
            if max_steps is not None and d >= max_steps:
                continue
            for v in adj.get(u, []):
                if v in node_set and v not in visited:
                    visited.add(v)
                    q.append((v, d + 1))
        return visited

    def _shortest_path_len_in_region(self, src, targets, region_nodes, adj, max_depth=12):
        node_set = set(region_nodes)
        targets = set(t for t in targets if t in node_set)
        if src not in node_set or not targets:
            return None
        q = deque([(src, 0)])
        visited = {src}
        while q:
            u, d = q.popleft()
            if u in targets:
                return d
            if d >= max_depth:
                continue
            for v in adj.get(u, []):
                if v in node_set and v not in visited:
                    visited.add(v)
                    q.append((v, d + 1))
        return None

    MEMORY_APIS = {
        "memcpy", "memmove", "memset", "strcpy", "strncpy", "strcat", "strncat",
        "sprintf", "snprintf", "vsprintf", "vsnprintf"
    }

    DANGEROUS_SINK_APIS = {
        "system", "popen", "execl", "execle", "execlp", "execv", "execvp", "execve",
        "printf", "fprintf", "sprintf", "snprintf", "scanf", "sscanf", "gets",
        "strcpy", "strcat", "memcpy", "realpath"
    }

    INPUT_SOURCE_APIS = {
        "scanf", "sscanf", "fscanf", "gets", "fgets", "getline", "read", "recv",
        "getenv", "argv", "argc"
    }

    ALLOC_APIS = {"malloc", "calloc", "realloc", "new", "fopen", "open", "socket"}
    RELEASE_APIS = {"free", "delete", "fclose", "close"}

    SIZE_HINT_WORDS = {"len", "length", "size", "sz", "cap", "count", "n", "num"}
    INPUT_HINT_WORDS = {
        "input", "buf", "buffer", "cmd", "path", "arg", "argv", "argc", "env",
        "data", "user", "name", "url", "src", "dst"
    }

    def _is_memory_call(self, n):
        return self._is_call_node(n) and self._call_name(n) in self.MEMORY_APIS

    def _is_dangerous_sink_call(self, n):
        return self._is_call_node(n) and self._call_name(n) in self.DANGEROUS_SINK_APIS

    def _is_alloc_call(self, n):
        return self._is_call_node(n) and self._call_name(n) in self.ALLOC_APIS

    def _is_release_call(self, n):
        return self._is_call_node(n) and self._call_name(n) in self.RELEASE_APIS

    def _is_input_source_node(self, n):
        if self._is_param_node(n):
            return True
        ids = [x.lower() for x in self._node_identifiers(n)]
        if any(x in self.INPUT_HINT_WORDS for x in ids):
            return True
        if self._is_call_node(n) and self._call_name(n) in self.INPUT_SOURCE_APIS:
            return True
        return False

    def _contains_array_access(self, code):

        return bool(re.search(r"[A-Za-z_]\w*\s*\[[^\]]+\]", code))

    def _contains_pointer_deref(self, code):

        return code.count("*")

    def _contains_pointer_arith(self, code):

        return bool(re.search(r"[A-Za-z_]\w*\s*[\+\-]\s*[A-Za-z0-9_]+", code))

    def _contains_cast(self, code):
        return len(re.findall(r"\([ \t]*[A-Za-z_][A-Za-z0-9_ \t\*]*\)", code))

    def _contains_arith(self, code):

        return len(re.findall(r"(?<!\+)\+(?!\+)|(?<!-)-(?!-)|\*|/|%", code))

    def _looks_like_size_expr_node(self, n):
        ids = [x.lower() for x in self._node_identifiers(n)]
        code = self._node_code(n).lower()
        if any(x in self.SIZE_HINT_WORDS for x in ids):
            return True
        if any(w in code for w in self.SIZE_HINT_WORDS):
            return True
        return False

    def _is_validation_condition(self, n):
        if not self._is_control_structure(n):
            return False
        code = self._node_code(n).replace(" ", "").lower()
        good_patterns = [
            r"[A-Za-z_]\w*<=[A-Za-z_]\w*",
            r"[A-Za-z_]\w*<[A-Za-z_]\w*",
            r"[A-Za-z_]\w*!=null",
            r"[A-Za-z_]\w*==null",
            r"[A-Za-z_]\w*>=0",
            r"[A-Za-z_]\w*>0",
        ]
        return any(re.search(p, code) for p in good_patterns)

    def _is_format_string_risky_call(self, n):
        if not self._is_call_node(n):
            return False
        name = self._call_name(n)
        if name not in {"printf", "fprintf", "sprintf", "snprintf", "scanf", "sscanf"}:
            return False
        code = self._node_code(n)

        return not bool(re.search(r'^\s*\w+\s*\(\s*"[^"]*"', code))

    def _approx_ast_nesting(self, nodes, ast_nodes_dict, ast_adj, max_depth_cap=10):
        node_set = set(nodes)

        def is_struct(nid):
            n = ast_nodes_dict[nid]
            if self._is_control_structure(n):
                return True
            return False

        children = {u: [v for v in ast_adj.get(u, []) if v in node_set] for u in nodes}
        best = 0

        def dfs(u, depth, visiting):
            nonlocal best
            if u in visiting:
                return
            visiting.add(u)
            d2 = depth + 1 if is_struct(u) else depth
            best = max(best, d2)
            for v in children.get(u, []):
                dfs(v, d2, visiting)
            visiting.remove(u)

        for u in nodes:
            dfs(u, 0, set())

        return self._clip01(best / max_depth_cap)

    def _cfg_cyclomatic(self, nodes, cfg_adj):
        n = max(1, len(nodes))
        m = self._count_internal_edges(nodes, cfg_adj)

        raw = max(0, m - n + 2)
        return self._log_norm(raw, scale=5.0)

    def build_region_attrs(
        self,
        region,
        ast_nodes_dict,
        ast_edges,
        cfg_edges,
        pdg_edges,
    ):
        """Return normalized RDP values in canonical topical-group order."""

        nodes = region["nodes"]
        lines = region.get("lines", [])
        n = max(1, len(nodes))
        node_set = set(nodes)

        ast_adj = self._build_adj(ast_edges)
        cfg_adj = self._build_adj(cfg_edges)
        pdg_adj = self._build_adj(pdg_edges)
        cfg_pred = self._build_pred(cfg_edges)
        pdg_pred = self._build_pred(pdg_edges)

        num_calls = 0
        num_identifiers = 0
        num_returns = 0
        num_ifs = 0
        num_loops = 0
        num_switch = 0

        pointer_deref_cnt = 0
        array_access_cnt = 0
        memory_call_cnt = 0
        pointer_arith_cnt = 0
        dangerous_sink_cnt = 0
        format_risky_cnt = 0

        alloc_cnt = 0
        release_cnt = 0

        arith_cnt = 0
        cast_cnt = 0

        input_id_hits = 0

        call_nodes = []
        memory_op_nodes = []
        dangerous_sink_nodes = []
        input_source_nodes = []
        alloc_nodes = []
        release_nodes = []
        return_nodes = []
        arith_nodes = []
        validation_nodes = []
        control_nodes = []

        for nid in nodes:
            node = ast_nodes_dict[nid]
            code = self._node_code(node)
            ids = self._node_identifiers(node)
            num_identifiers += len(ids)

            lbl = self._node_label(node)
            if self._is_call_node(node):
                num_calls += 1
                call_nodes.append(nid)

            if self._is_return_node(node):
                num_returns += 1
                return_nodes.append(nid)

            if self._is_control_structure(node):
                control_nodes.append(nid)
                kind = self._control_kind(node)
                if kind == "if":
                    num_ifs += 1
                elif kind in ("for", "while"):
                    num_loops += 1
                elif kind == "switch":
                    num_switch += 1

            if self._is_validation_condition(node):
                validation_nodes.append(nid)

            if self._contains_pointer_deref(code) > 0:
                pointer_deref_cnt += self._contains_pointer_deref(code)

            if self._contains_array_access(code):
                array_access_cnt += 1

            if self._contains_pointer_arith(code):
                pointer_arith_cnt += 1

            if self._contains_cast(code) > 0:
                cast_cnt += self._contains_cast(code)

            if self._contains_arith(code) > 0:
                arith_cnt += self._contains_arith(code)
                arith_nodes.append(nid)

            if self._is_memory_call(node):
                memory_call_cnt += 1
                memory_op_nodes.append(nid)

            if self._is_dangerous_sink_call(node):
                dangerous_sink_cnt += 1
                dangerous_sink_nodes.append(nid)

            if self._is_format_string_risky_call(node):
                format_risky_cnt += 1

            if self._is_alloc_call(node):
                alloc_cnt += 1
                alloc_nodes.append(nid)

            if self._is_release_call(node):
                release_cnt += 1
                release_nodes.append(nid)

            low_ids = [x.lower() for x in ids]
            input_id_hits += sum(1 for x in low_ids if x in self.INPUT_HINT_WORDS)

            if self._is_input_source_node(node):
                input_source_nodes.append(nid)

        num_calls_safe = max(1, num_calls)
        num_identifiers_safe = max(1, num_identifiers)
        num_memory_ops_safe = max(1, len(memory_op_nodes))
        num_input_sources_safe = max(1, len(input_source_nodes))
        num_arith_safe = max(1, len(arith_nodes))
        num_alloc_safe = max(1, alloc_cnt)
        num_ops_safe = max(1, num_calls + len(arith_nodes))

        M1_ptr_deref_density = self._clip01(pointer_deref_cnt / n / 4.0)
        M2_array_access_density = self._clip01(array_access_cnt / n)
        M3_memory_call_ratio = self._clip01(memory_call_cnt / num_calls_safe)
        M4_pointer_arith_density = self._clip01(pointer_arith_cnt / n)

        guarded_memory = 0
        unguarded_memory = 0
        for m in memory_op_nodes:
            preds = set(cfg_pred.get(m, [])) | set(pdg_pred.get(m, []))
            ok = any(p in node_set and p in validation_nodes for p in preds)
            if not ok:

                for p in list(preds):
                    preds2 = set(cfg_pred.get(p, [])) | set(pdg_pred.get(p, []))
                    if any(pp in node_set and pp in validation_nodes for pp in preds2):
                        ok = True
                        break
            if ok:
                guarded_memory += 1
            else:
                unguarded_memory += 1

        M5_memory_guard_ratio = self._clip01(guarded_memory / num_memory_ops_safe)
        M6_unguarded_memory_ratio = self._clip01(unguarded_memory / num_memory_ops_safe)

        I1_input_identifier_ratio = self._clip01(input_id_hits / num_identifiers_safe)
        I2_dangerous_sink_ratio = self._clip01(dangerous_sink_cnt / num_calls_safe)
        I3_format_string_risk_ratio = self._clip01(format_risky_cnt / num_calls_safe)

        validated_input_uses = 0
        sink_reachable_from_input = 0
        total_input_uses = 0

        pdg_reachable_from_input = self._reachable_within_region(input_source_nodes, nodes, pdg_adj, max_steps=8)

        for s in dangerous_sink_nodes:
            total_input_uses += 1

            if s in pdg_reachable_from_input:
                sink_reachable_from_input += 1
                preds = set(pdg_pred.get(s, [])) | set(cfg_pred.get(s, []))
                ok = any(p in node_set and p in validation_nodes for p in preds)
                if not ok:

                    for p in list(preds):
                        preds2 = set(pdg_pred.get(p, [])) | set(cfg_pred.get(p, []))
                        if any(pp in node_set and pp in validation_nodes for pp in preds2):
                            ok = True
                            break
                if ok:
                    validated_input_uses += 1

        total_input_uses_safe = max(1, total_input_uses)
        I4_input_validation_ratio = self._clip01(validated_input_uses / total_input_uses_safe)
        I5_input_to_sink_reachable_ratio = self._clip01(sink_reachable_from_input / total_input_uses_safe)

        input_fanout = len(pdg_reachable_from_input)
        I6_input_fanout = self._clip01(input_fanout / n)

        R1_alloc_call_ratio = self._clip01(alloc_cnt / num_calls_safe)
        R2_release_call_ratio = self._clip01(release_cnt / num_calls_safe)
        R3_alloc_release_imbalance = self._clip01(max(0, alloc_cnt - release_cnt) / num_alloc_safe)

        exits_wo_release = 0
        if alloc_cnt > release_cnt:
            exits_wo_release = len(return_nodes)
        R4_exit_without_release_ratio = self._clip01(exits_wo_release / max(1, len(return_nodes)))

        dep_depths = []
        combined_adj = defaultdict(list)
        for u in set(list(cfg_adj.keys()) + list(pdg_adj.keys())):
            combined_adj[u] = list(set(cfg_adj.get(u, [])) | set(pdg_adj.get(u, [])))

        for a in alloc_nodes:
            d = self._shortest_path_len_in_region(a, release_nodes, nodes, combined_adj, max_depth=10)
            if d is not None:
                dep_depths.append(d)

        avg_dep_depth = sum(dep_depths) / len(dep_depths) if dep_depths else 0.0
        R5_resource_dependency_depth = self._clip01(avg_dep_depth / 10.0)

        R6_early_return_ratio = self._clip01(num_returns / n)

        C1_branch_density = self._clip01(num_ifs / n)
        C2_loop_density = self._clip01(num_loops / n)
        C3_switch_density = self._clip01(num_switch / n)
        C4_cfg_cyclomatic = self._cfg_cyclomatic(nodes, cfg_adj)
        C5_max_ast_nesting = self._approx_ast_nesting(nodes, ast_nodes_dict, ast_adj, max_depth_cap=10)

        cfg_internal_edges = self._count_internal_edges(nodes, cfg_adj)
        C6_cfg_branch_fanout = self._clip01((cfg_internal_edges / n) / 3.0)

        guarded_ops = 0
        op_nodes = set(call_nodes) | set(arith_nodes)
        for op in op_nodes:
            preds = set(cfg_pred.get(op, [])) | set(pdg_pred.get(op, []))
            ok = any(p in node_set and p in validation_nodes for p in preds)
            if ok:
                guarded_ops += 1
        C7_guarded_operation_ratio = self._clip01(guarded_ops / max(1, len(op_nodes)))

        A1_arithmetic_density = self._clip01(arith_cnt / n / 4.0)
        A2_cast_density = self._clip01(cast_cnt / n / 3.0)

        size_comp_hits = 0
        size_related_nodes = [nid for nid in arith_nodes if self._looks_like_size_expr_node(ast_nodes_dict[nid])]
        size_targets = set(alloc_nodes) | set(memory_op_nodes)

        reachable_from_size = self._reachable_within_region(size_related_nodes, nodes, pdg_adj, max_steps=6)
        for t in size_targets:
            if t in reachable_from_size:
                size_comp_hits += 1

        A3_size_computation_pattern_ratio = self._clip01(size_comp_hits / max(1, len(size_targets)))

        guarded_arith = 0
        for a in arith_nodes:
            preds = set(cfg_pred.get(a, [])) | set(pdg_pred.get(a, []))
            ok = any(p in node_set and p in validation_nodes for p in preds)
            if ok:
                guarded_arith += 1
        A4_arithmetic_guard_ratio = self._clip01(guarded_arith / num_arith_safe)

        S1_region_size_score = self._clip01(len(nodes) / 30.0)

        if lines:
            span = max(lines) - min(lines) + 1
            S2_region_span_score = self._clip01(span / 50.0)
        else:
            S2_region_span_score = 0.0

        ast_internal_edges = self._count_internal_edges(nodes, ast_adj)
        S3_ast_internal_edge_density = self._clip01((ast_internal_edges / n) / 4.0)
        S4_call_density = self._clip01(num_calls / n)
        S5_identifier_density = self._clip01(num_identifiers / n / 3.0)
        S6_cfg_internal_edge_density = self._clip01((cfg_internal_edges / n) / 4.0)

        feats = [

            M1_ptr_deref_density,
            M2_array_access_density,
            M3_memory_call_ratio,
            M4_pointer_arith_density,

            M6_unguarded_memory_ratio,

            I1_input_identifier_ratio,
            I2_dangerous_sink_ratio,
            I3_format_string_risk_ratio,

            I5_input_to_sink_reachable_ratio,
            I6_input_fanout,

            R1_alloc_call_ratio,
            R2_release_call_ratio,
            R3_alloc_release_imbalance,
            R4_exit_without_release_ratio,
            R5_resource_dependency_depth,
            R6_early_return_ratio,

            C1_branch_density,
            C2_loop_density,
            C3_switch_density,
            C4_cfg_cyclomatic,
            C5_max_ast_nesting,
            C6_cfg_branch_fanout,

            A1_arithmetic_density,
            A2_cast_density,
            A3_size_computation_pattern_ratio,

            S1_region_size_score,
            S2_region_span_score,
            S3_ast_internal_edge_density,
            S4_call_density,
            S5_identifier_density,
            S6_cfg_internal_edge_density,
        ]

        names = list(RDP_FEATURES)

        feats = [self._clip01(x) for x in feats]
        if len(feats) != RDP_DIM:
            raise ValueError("RDP feature count does not match the schema")
        return feats, names

class DatasetBuilder(RegionFeatureBuilderMixin):
    """Build aligned graph, source-line, region, and RDP fields from prepared input."""
    def __init__(self,
                 c_dir: str,
                 js_dir: str,
                 vul_lines_path: str,
                 w2v_path: str,

                 seed: int = 42):
        c_regexp = (
            r'\w+|->|\+\+|--|<=|>=|==|!=|'
            r'<<|>>|&&|\|\||-=|\+=|\*=|/=|%=|'
            r'&=|<<=|>>=|^=|\|=|::|'
            r'[!@#$%^&*()_+\-=\[\]{};\':"\|,.<>/?]'
        )
        self.tokenizer = RegexpTokenizer(c_regexp)

        self.original_c = c_dir
        self.original_js = js_dir
        self.vul_lines = vul_lines_path
        self.w2v_path = w2v_path

        if not os.path.exists(self.w2v_path):
            raise FileNotFoundError(f"Word2Vec file not found: {self.w2v_path}")
        self.word_vectors   = gensim.models.KeyedVectors.load(self.w2v_path, mmap='r')
        self.embedding_size = self.word_vectors.vector_size

        self.rng = random.Random(seed)


    def build_classify_dataset(self):
        path_config = {
            "original_c": self.original_c,
            "original_js": self.original_js,
            "vul_lines": self.vul_lines
        }

        with open(path_config["vul_lines"], 'r') as f:
            vul_lines_data = json.load(f)

        dataset = []
        error_log = []
        c_files = sorted([
            f for f in os.listdir(path_config["original_c"])
            if f.endswith(".c")
        ])

        region_attr_stats = self._init_region_attr_stats()

        pbar = tqdm(c_files, desc="Processing C-JSON pairs")

        for c_file in pbar:
            try:
                sample_id = c_file.split(".")[0]
                js_file = f"{sample_id}.json"
                c_path = os.path.join(path_config["original_c"], c_file)
                js_path = os.path.join(path_config["original_js"], js_file)

                if not os.path.exists(js_path):
                    raise FileNotFoundError(f"Missing JSON file: {js_file}")

                vul_lines = vul_lines_data.get(c_file, [])
                data = self.prepare_torch_graph(
                    js_path=js_path,
                    c_path=c_path,
                    vul_lines=vul_lines
                )

                dataset.append(data)

                self._update_region_attr_stats(
                    stats=region_attr_stats,
                    region_attr_tensor=data.region_attr,
                    feature_names=data.region_attr_names,
                    file_name=data.file_name
                )

            except Exception as e:
                Traceback = traceback.format_exc()
                print(f"Traceback:{Traceback}")
                error_msg = (
                    f"[{c_file}] processing error\n"
                    f"  C path: {c_path}\n"
                    f"  JSON path: {js_path}\n"
                    f"  Error: {repr(e)}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                error_log.append(error_msg)
                continue

        self._save_error_log(error_log)

        report = self._finalize_region_attr_stats(
            region_attr_stats,
            near_zero_ratio_threshold=0.001
        )
        self._save_region_attr_stats_report(report)

        return dataset

    def _save_error_log(self, error_log: List[str]):
        if error_log:
            with open(os.path.join(getattr(self, "output_dir", "."), f"{self.dataset_name}_error.log"), "w") as f:
                f.write("\n".join(error_log))
            logger.warning(f"Encountered {len(error_log)} errors, see log file for details")

    def build_regions(self, ast_nodes, ast_edges, vul_lines):
        """Build syntax-guided regions and Eq. (11) weak scores."""
        regions = construct_regions(ast_nodes, ast_edges)
        if not regions:
            raise ValueError("The AST contains no usable structural regions")
        for region in regions:
            region["score"] = weak_region_target(region["lines"], vul_lines)
        return regions

    def prepare_torch_graph(self, js_path: str, c_path: str, vul_lines) -> Data:
        """Convert one matched source/Joern pair into a PyG sample."""
        with open(js_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        file_name = os.path.basename(js_path)

        ast_nodes = json_data['ast_nodes']
        ast_edges = json_data['ast_edges']
        ast_nodes, ast_edges = self.filter_nodes(ast_nodes, ast_edges)

        ast_nodes_dict = {node['id']: node for node in ast_nodes}

        final_regions = self.build_regions(ast_nodes, ast_edges, vul_lines)

        final_nodes_index_lists = [region['nodes'] for region in final_regions]

        normalized_node_index_lists = self.normalize_lists(ast_nodes, final_nodes_index_lists)
        normalized_node_index_lists_tensors = [
            torch.tensor(sublist, dtype=torch.long)
            for sublist in normalized_node_index_lists
        ]

        final_score_lists = [region['score'] for region in final_regions]
        region_score_gt_tensor = torch.tensor(final_score_lists, dtype=torch.float32)
        region_score_gt_tensor = torch.nan_to_num(
            region_score_gt_tensor, nan=0.0, posinf=1.0, neginf=0.0
        )
        region_score_gt_tensor = region_score_gt_tensor.clamp_(0.0, 1.0)

        final_region_line_numbers = [region['lines'] for region in final_regions]
        region_line_numbers_lists_tensors = [
            torch.tensor(sublist, dtype=torch.long)
            for sublist in final_region_line_numbers
        ]

        cfg_edges = json_data.get('cfg_edges', [])
        pdg_edges = json_data.get('cdg_edges', []) + json_data.get('ddg_edges', [])

        region_attr_list_all = []
        region_attr_names = None
        for region in final_regions:
            feats, names = self.build_region_attrs(
                region=region,
                ast_nodes_dict=ast_nodes_dict,
                ast_edges=ast_edges,
                cfg_edges=cfg_edges,
                pdg_edges=pdg_edges,
            )
            if len(feats) != RDP_DIM or tuple(names) != RDP_FEATURES:
                raise ValueError("RDP output does not match the canonical schema")
            region_attr_list_all.append(feats)
            if region_attr_names is None:
                region_attr_names = names
        region_attr_tensor = torch.tensor(region_attr_list_all, dtype=torch.float32).reshape(-1, RDP_DIM)
        region_attr_tensor = torch.nan_to_num(region_attr_tensor, nan=0.0, posinf=1.0, neginf=0.0)

        global_code_embeddings = self.global_code_embedding(c_path)

        token_edges, stmt_edges, block_edges = self.build_hierarchical_edges(ast_nodes, ast_edges)

        normalized_token_edges = self.normalize_graph(ast_nodes, token_edges)
        normalized_stmt_edges = self.normalize_graph(ast_nodes, stmt_edges)
        normalized_block_edges = self.normalize_graph(ast_nodes, block_edges)
        normalized_ast_edges = self.normalize_graph(ast_nodes, ast_edges)

        ast_x = self.generate_node_embeddings(ast_nodes, "AST")
        ast_edge_index = torch.tensor(normalized_ast_edges, dtype=torch.int64).t().contiguous()
        token_edge_index = torch.tensor(normalized_token_edges, dtype=torch.int64).t().contiguous()
        stmt_edge_index = torch.tensor(normalized_stmt_edges, dtype=torch.int64).t().contiguous()
        block_edge_index = torch.tensor(normalized_block_edges, dtype=torch.int64).t().contiguous()

        subgraphs = self.extract_region_feature(
            normalized_node_index_lists_tensors,
            ast_x,
            ast_edge_index
        )
        region_nodes_list = [region['nodes'] for region in subgraphs]

        normalized_cfg_edges = self.normalize_graph(ast_nodes, cfg_edges)
        normalized_pdg_edges = self.normalize_graph(ast_nodes, pdg_edges)

        cfg_x, cfg_edge_index, cfg_id_map = self._build_subgraph_x_and_edge_index(
            edges_base_on_ast=normalized_cfg_edges,
            ast_nodes=ast_nodes,
            ast_x=ast_x,
            graph_type="CFG",
            use_regen_embedding=False
        )

        pdg_x, pdg_edge_index, pdg_id_map = self._build_subgraph_x_and_edge_index(
            edges_base_on_ast=normalized_pdg_edges,
            ast_nodes=ast_nodes,
            ast_x=ast_x,
            graph_type="PDG",
            use_regen_embedding=False
        )

        region_cfg_nodes_list = self._map_regions_to_subgraph_indices(region_nodes_list, cfg_id_map)
        region_pdg_nodes_list = self._map_regions_to_subgraph_indices(region_nodes_list, pdg_id_map)

        assert len(region_nodes_list) == len(region_line_numbers_lists_tensors) == len(region_cfg_nodes_list) == \
               len(region_pdg_nodes_list) == region_attr_tensor.size(0) == region_score_gt_tensor.size(0), \
            f"Region-aligned fields mismatch in file {file_name}"

        data = Data(

            ast_x=ast_x,
            token_edge_index=token_edge_index,
            stmt_edge_index=stmt_edge_index,
            block_edge_index=block_edge_index,

            global_code_embedding=global_code_embeddings,

            region_line_numbers_lists=region_line_numbers_lists_tensors,

            region_ast_nodes_list=region_nodes_list,
            region_cfg_nodes_list=region_cfg_nodes_list,
            region_pdg_nodes_list=region_pdg_nodes_list,

            cfg_x=cfg_x,
            cfg_edge_index=cfg_edge_index,
            pdg_x=pdg_x,
            pdg_edge_index=pdg_edge_index,

            region_score_gt=region_score_gt_tensor,

            region_attr=region_attr_tensor,
            region_attr_names=region_attr_names,

            num_regions=torch.tensor([len(region_nodes_list)], dtype=torch.long),

            y=torch.tensor([self.get_target(file_name)], dtype=torch.long),

            file_name=file_name[:-5]
        )

        return data

    def _node_label(self, n):

        return (n.get("_label") or n.get("label") or "").upper()

    def _node_code(self, n):

        for k in ("code", "snippet", "text"):
            v = n.get(k)
            if isinstance(v, str) and v.strip():
                return v

        for k in ("name", "value"):
            v = n.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    def _is_call_node(self, n):
        return self._node_label(n) == "CALL"

    def _call_name(self, n):

        if self._node_label(n) != "CALL":
            return ""

        for k in ("methodFullName", "name", "fullName"):
            v = n.get(k)
            if isinstance(v, str) and v:

                base = v.split(":")[0]
                base = base.split("::")[-1].split(".")[-1]
                return base.lower()

        code = self._node_code(n)
        m = re.match(r"\s*([A-Za-z_]\w*)\s*\(", code)
        return m.group(1).lower() if m else ""

    def _is_identifier_node(self, n):
        return self._node_label(n) == "IDENTIFIER"

    def _identifier_name(self, n):

        v = n.get("name")
        if isinstance(v, str) and v:
            return v

        return self._node_code(n)

    def _node_identifiers(self, n):

        if self._is_identifier_node(n):
            name = self._identifier_name(n)
            return [name] if name else []
        code = self._node_code(n)
        return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code)

    def _is_control_structure(self, n):

        return self._node_label(n) == "CONTROL_STRUCTURE"

    def _control_kind(self, n):

        if not self._is_control_structure(n):
            return ""
        c = self._node_code(n).strip().lower()
        if c.startswith("if"):
            return "if"
        if c.startswith("for"):
            return "for"
        if c.startswith("while"):
            return "while"
        if c.startswith("switch"):
            return "switch"
        if c.startswith("try"):
            return "try"
        if c.startswith("catch"):
            return "catch"
        return ""

    def _is_return_node(self, n):
        return self._node_label(n) == "RETURN"

    def _clip01(self, x):
        try:
            return float(max(0.0, min(1.0, x)))
        except Exception:
            return 0.0

    def _log_norm(self, x, scale=5.0):

        return self._clip01(math.log1p(max(0.0, x)) / scale)

    def _unique_preserve_order(self, seq_iterable):

        seen = set()
        out = []
        for x in seq_iterable:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _map_regions_to_subgraph_indices(self, region_nodes_list, id_map):

        out = []
        for nodes in region_nodes_list:
            if isinstance(nodes, torch.Tensor):
                nodes = nodes.detach().cpu().tolist()

            mapped = [id_map[n] for n in nodes if n in id_map]
            mapped = self._unique_preserve_order(mapped)
            out.append(torch.tensor(mapped, dtype=torch.long))
        return out

    def _build_subgraph_x_and_edge_index(
            self,
            edges_base_on_ast,
            ast_nodes,
            ast_x,
            graph_type: str,
            use_regen_embedding=True,
    ):

        if not edges_base_on_ast:

            empty_x = torch.empty((0, ast_x.size(1)), dtype=ast_x.dtype, device=ast_x.device)
            empty_e = torch.empty((2, 0), dtype=torch.int64, device=ast_x.device)
            return empty_x, empty_e, {}

        used_old_ids = []
        for u, v in edges_base_on_ast:
            used_old_ids.append(int(u))
            used_old_ids.append(int(v))
        used_old_ids = sorted(set(used_old_ids))

        old2new = {old_id: new_id for new_id, old_id in enumerate(used_old_ids)}

        if use_regen_embedding:

            sub_nodes = [ast_nodes[old_id] for old_id in used_old_ids]
            sub_x = self.generate_node_embeddings(sub_nodes, graph_type)
        else:
            idx = torch.tensor(used_old_ids, dtype=torch.long, device=ast_x.device)
            sub_x = ast_x.index_select(0, idx)

        remapped = [(old2new[int(u)], old2new[int(v)]) for u, v in edges_base_on_ast]
        sub_edge_idx = torch.tensor(remapped, dtype=torch.int64, device=ast_x.device).t().contiguous()

        id_map = old2new
        return sub_x, sub_edge_idx, id_map

    def extract_region_feature(self, region_nodes_list, node_features, edge_index):
        subgraphs = []

        for region_nodes in region_nodes_list:

            region_set = set(region_nodes.tolist())

            sub_node_features = node_features[region_nodes]

            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(region_nodes.tolist())}

            src, dst = edge_index
            mask = torch.tensor([(s.item() in region_set and d.item() in region_set)
                                 for s, d in zip(src, dst)], dtype=torch.bool)

            sub_edge_index = edge_index[:, mask]

            sub_edge_index = torch.stack([
                torch.tensor([node_mapping[idx.item()] for idx in sub_edge_index[0]]),
                torch.tensor([node_mapping[idx.item()] for idx in sub_edge_index[1]])
            ])

            subgraphs.append({
                'nodes': region_nodes,
                'features': sub_node_features,
                'edges': sub_edge_index
            })

        return subgraphs

    def global_code_embedding(self, c_path: str) -> torch.Tensor:

        with open(c_path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = [line.rstrip('\n') for line in f]

        line_embeddings = []
        for line in all_lines:
            try:

                cleaned_line = line.strip()
                if not cleaned_line:
                    emb = np.zeros(self.word_vectors.vector_size)
                else:

                    cleaned_line = line.replace('\t', ' ').replace('\n', ' ')
                    tokens = self.tokenizer.tokenize(cleaned_line)
                    vectors = [self.word_vectors[t] for t in tokens if t in self.word_vectors]
                    emb = np.mean(vectors, axis=0) if vectors else np.zeros(self.word_vectors.vector_size)

                if emb.shape != (self.word_vectors.vector_size,):
                    emb = np.zeros(self.word_vectors.vector_size)

            except Exception as e:
                print(f"Error in line {len(line_embeddings) + 1}: {str(e)}")
                emb = np.zeros(self.word_vectors.vector_size)

            line_embeddings.append(emb)

        embeddings_tensor = torch.from_numpy(np.array(line_embeddings, dtype=np.float32))
        assert embeddings_tensor.ndim == 2, f"维度错误，应为(n, d)，实际得到{embeddings_tensor.shape}"
        return embeddings_tensor












    def filter_nodes(self, ast_nodes, ast_edges):

        node_ids = {node['id'] for node in ast_nodes}

        root_node_id = min(node_ids)

        nodes_to_remove = set()
        nodes_to_remove.add(root_node_id)

        first_child_id = None
        second_child_id = None

        for edge in ast_edges:
            if edge[0] == root_node_id:

                if first_child_id is None:
                    first_child_id = edge[1]
                    nodes_to_remove.add(first_child_id)
                else:
                    second_child_id = edge[1]
                    nodes_to_remove.add(second_child_id)
                    break

        filtered_nodes = [node for node in ast_nodes if node['id'] not in nodes_to_remove]

        filtered_edges = []
        for edge in ast_edges:
            if edge[0] not in nodes_to_remove and edge[1] not in nodes_to_remove:
                filtered_edges.append(edge)

        return filtered_nodes, filtered_edges

    def normalize_lists(self, nodes, node_index_lists):

        old_to_new_id_dict = {node['id']: new_id for new_id, node in enumerate(nodes)}

        normalized_lists = [[old_to_new_id_dict[node_id] for node_id in sublist] for sublist in node_index_lists]

        return normalized_lists


    def generate_node_embeddings(self, nodes, graph_type):
        node_embedding_dict = {}
        for n in nodes:

            if 'code' in n:
                n_code = n['code']
            else:
                n_code = ""
            try:
                n_code = n_code.replace('\\t', ' ')
                if not n_code:
                    code_embedding = np.zeros(self.word_vectors.vector_size)
                else:
                    code_embedding = np.nanmean([self.word_vectors[x] for x in self.tokenizer.tokenize(n_code)], axis=0)
            except KeyError as e:
                raise TorchGraphPrepError(f"Embedding error {e} in {graph_type} graph")

            node_embedding_dict[n['id']] = code_embedding

        node_embeddings = np.array([node_embedding_dict[node['id']] for node in nodes], dtype=np.float32)
        return torch.tensor(node_embeddings, dtype=torch.float)

    def normalize_graph(self, nodes, edges):

        old_to_new_id_dict = {node['id']: new_id for new_id, node in enumerate(nodes)}

        normalized_edges = [[old_to_new_id_dict[edge[0]], old_to_new_id_dict[edge[1]]] for edge in edges]

        return normalized_edges

    def _remap_rare_labels_to_unknown(self, ast_nodes):

        for n in ast_nodes:
            lab = n.get('_label', '')
            if lab in ('TYPE_DECL', 'MEMBER'):
                n['_label'] = 'UNKNOWN'

    def _node_level(self, label: str) -> str:

        if label in ('METHOD', 'BLOCK', 'CONTROL_STRUCTURE'):
            return 'block'
        if label in ('CALL', 'RETURN'):
            return 'statement'

        return 'token'

    def build_hierarchical_edges(self, ast_nodes, ast_edges):

        self._remap_rare_labels_to_unknown(ast_nodes)

        id2level = {}
        for n in ast_nodes:
            lab = n.get('_label', '')
            id2level[n['id']] = self._node_level(lab)

        token_e_set = set()
        stmt_e_set = set()
        block_e_set = set()

        for e in ast_edges:
            u, v = int(e[0]), int(e[1])
            lu = id2level.get(u, 'token')
            lv = id2level.get(v, 'token')

            if lu == 'block' and lv == 'block':
                block_e_set.add((u, v))
                continue

            if (lu != 'token' and lv != 'token') and ('statement' in (lu, lv)):
                stmt_e_set.add((u, v))
                continue

            if 'token' in (lu, lv):
                token_e_set.add((u, v))
                continue

            token_e_set.add((u, v))

        token_edges = [[u, v] for (u, v) in token_e_set]
        stmt_edges = [[u, v] for (u, v) in stmt_e_set]
        block_edges = [[u, v] for (u, v) in block_e_set]

        return token_edges, stmt_edges, block_edges

    def get_target(self, file_name: str) -> int:

        target_str = file_name.split("_")[1]
        non_num = re.compile(r'[^\d]')
        target = int(non_num.sub('', target_str))
        return target

    def save_processed_data(self, data, file_path: str) -> None:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Degrees of freedom.*")
