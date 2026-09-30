import unittest

import torch

from regcap.data_loading import GlobalDataModule
from regcap.model import VulDetectionModel
from regcap.rdp_schema import RDP_DIM, RDP_FEATURES, RDP_SE_FEATURES, RDP_VE_FEATURES
from regcap.regions import construct_regions, weak_region_target


class CoreChecks(unittest.TestCase):
    def test_weak_region_target(self):
        self.assertEqual(weak_region_target([2, 3, 4, 5], [3, 5, 8]), 0.5)
        self.assertEqual(weak_region_target([], [3]), 0.0)
        self.assertEqual(weak_region_target([1, 2], []), 0.0)
        self.assertEqual(weak_region_target([1, 1, 2], [1]), 0.5)

    def test_syntax_regions(self):
        nodes = [
            {"id": 1, "_label": "METHOD", "lineNumber": 1},
            {"id": 2, "_label": "BLOCK", "lineNumber": 2},
            {"id": 3, "_label": "CALL", "lineNumber": 3, "code": "dangerous_api()"},
            {"id": 4, "_label": "CONTROL_STRUCTURE", "lineNumber": 4},
            {"id": 5, "_label": "CALL", "lineNumber": 5, "code": "safe_api()"},
            {"id": 6, "_label": "RETURN", "lineNumber": 6},
        ]
        edges = [[1, 2], [2, 3], [2, 4], [4, 5], [2, 6]]
        regions = construct_regions(nodes, edges)
        self.assertEqual(sorted(n for region in regions for n in region["nodes"]), list(range(1, 7)))
        self.assertEqual([min(region["lines"]) for region in regions], sorted(min(region["lines"]) for region in regions))
        nodes[2]["code"] = "safe_api()"
        self.assertEqual(regions, construct_regions(nodes, edges))

    def test_schema_and_region_cap(self):
        expected = (
            "mem_ptr_deref_density", "mem_array_access_density", "mem_memory_call_ratio",
            "mem_pointer_arith_density", "mem_unguarded_ratio", "input_identifier_ratio",
            "input_dangerous_sink_ratio", "input_format_string_risk_ratio",
            "input_to_sink_reachable_ratio", "input_fanout_ratio", "res_alloc_call_ratio",
            "res_release_call_ratio", "res_alloc_release_imbalance", "res_exit_without_release_ratio",
            "res_dependency_depth", "res_early_return_ratio", "ctrl_branch_density",
            "ctrl_loop_density", "ctrl_switch_density", "ctrl_cfg_cyclomatic",
            "ctrl_max_ast_nesting", "ctrl_cfg_branch_fanout", "arith_density",
            "arith_cast_density", "arith_size_computation_pattern_ratio",
            "struct_region_size_score", "struct_region_span_score",
            "struct_ast_internal_edge_density", "struct_call_density",
            "struct_identifier_density", "struct_cfg_internal_edge_density",
        )
        self.assertEqual(RDP_FEATURES, expected)
        self.assertEqual((RDP_DIM, len(RDP_FEATURES)), (31, 31))
        self.assertEqual((len(RDP_VE_FEATURES), len(RDP_SE_FEATURES)), (24, 7))
        module = GlobalDataModule(data_path="", batch_size=1, max_regions=17, emb_dim=128, save_lists_dir=None)
        indices = module._select_region_indices_uniform(100, 17)
        self.assertEqual(len(indices), 17)
        self.assertEqual((indices[0], indices[-1]), (0, 99))
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(module._select_region_indices_uniform(5, 17), list(range(5)))

    def test_five_view_gate(self):
        model = VulDetectionModel(hidden_dim=128, input_dim=128, attr_dim=31)
        self.assertEqual(model.view_order, ("token", "ast", "cfg", "pdg", "prior"))
        projected = [model.view_projections[name](torch.randn(2, 3, 128))
                     for name in model.view_order]
        gates = torch.softmax(model.view_gate(torch.cat(projected, dim=-1)), dim=-1)
        self.assertEqual(gates.shape, (2, 3, 5))
        self.assertTrue(torch.allclose(gates.sum(dim=-1), torch.ones(2, 3), atol=1e-6))

    def test_source_line_index(self):
        module = GlobalDataModule(data_path="", batch_size=1, max_regions=17, emb_dim=128, save_lists_dir=None)
        values, regions, valid = module._pad_2d_lines([[[1, 3, 0]]], 1, 3)
        self.assertEqual(values[0, 0].tolist(), [0, 2, 0])
        self.assertEqual(valid[0, 0].tolist(), [True, True, False])
        self.assertTrue(regions[0, 0])


if __name__ == "__main__":
    unittest.main()
