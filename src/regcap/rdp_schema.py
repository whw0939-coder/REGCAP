"""Canonical feature schema for the 31-dimensional Regional Distributional Prior."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RDPFeature:
    name: str
    group: str
    orientation: str
    description: str


_FEATURES = (
    RDPFeature("mem_ptr_deref_density", "Memory Safety", "VE", "Density of pointer dereference operations within the region."),
    RDPFeature("mem_array_access_density", "Memory Safety", "VE", "Density of array-style accesses."),
    RDPFeature("mem_memory_call_ratio", "Memory Safety", "VE", "Ratio of memory-related API calls among all calls in the region."),
    RDPFeature("mem_pointer_arith_density", "Memory Safety", "VE", "Density of pointer arithmetic patterns."),
    RDPFeature("mem_unguarded_ratio", "Memory Safety", "VE", "Ratio of memory operations not protected by nearby validation conditions."),
    RDPFeature("input_identifier_ratio", "Input Validation & Injection", "VE", "Ratio of identifiers associated with external-input semantics."),
    RDPFeature("input_dangerous_sink_ratio", "Input Validation & Injection", "VE", "Ratio of dangerous sink calls among all calls."),
    RDPFeature("input_format_string_risk_ratio", "Input Validation & Injection", "VE", "Ratio of format-related calls lacking literal format strings."),
    RDPFeature("input_to_sink_reachable_ratio", "Input Validation & Injection", "VE", "Ratio of sinks reachable from input sources via dependencies."),
    RDPFeature("input_fanout_ratio", "Input Validation & Injection", "VE", "Influence scope of input sources within the region."),
    RDPFeature("res_alloc_call_ratio", "Resource Lifecycle", "VE", "Ratio of allocation calls among all calls."),
    RDPFeature("res_release_call_ratio", "Resource Lifecycle", "SE", "Ratio of release calls among all calls."),
    RDPFeature("res_alloc_release_imbalance", "Resource Lifecycle", "VE", "Imbalance between allocation and release operations."),
    RDPFeature("res_exit_without_release_ratio", "Resource Lifecycle", "VE", "Likelihood of exit without proper release."),
    RDPFeature("res_dependency_depth", "Resource Lifecycle", "VE", "Average dependency depth from allocation to release."),
    RDPFeature("res_early_return_ratio", "Resource Lifecycle", "VE", "Frequency of early returns."),
    RDPFeature("ctrl_branch_density", "Control-Flow Complexity", "VE", "Density of branching structures."),
    RDPFeature("ctrl_loop_density", "Control-Flow Complexity", "VE", "Density of loops."),
    RDPFeature("ctrl_switch_density", "Control-Flow Complexity", "VE", "Density of switch structures."),
    RDPFeature("ctrl_cfg_cyclomatic", "Control-Flow Complexity", "VE", "Approximate cyclomatic complexity."),
    RDPFeature("ctrl_max_ast_nesting", "Control-Flow Complexity", "VE", "Maximum AST nesting depth."),
    RDPFeature("ctrl_cfg_branch_fanout", "Control-Flow Complexity", "VE", "Branch fan-out in CFG."),
    RDPFeature("arith_density", "Arithmetic & Type Safety", "VE", "Density of arithmetic operations."),
    RDPFeature("arith_cast_density", "Arithmetic & Type Safety", "VE", "Density of type casts."),
    RDPFeature("arith_size_computation_pattern_ratio", "Arithmetic & Type Safety", "VE", "Size-related arithmetic flowing to memory operations."),
    RDPFeature("struct_region_size_score", "Structural Context", "SE", "Relative size of the region."),
    RDPFeature("struct_region_span_score", "Structural Context", "SE", "Code span covered by the region."),
    RDPFeature("struct_ast_internal_edge_density", "Structural Context", "SE", "Density of AST internal edges."),
    RDPFeature("struct_call_density", "Structural Context", "SE", "Density of call nodes."),
    RDPFeature("struct_identifier_density", "Structural Context", "SE", "Density of identifiers."),
    RDPFeature("struct_cfg_internal_edge_density", "Structural Context", "SE", "Density of CFG internal edges."),
)

RDP_DIM = 31
RDP_FEATURES = tuple(feature.name for feature in _FEATURES)
RDP_GROUPS = {
    group: tuple(feature.name for feature in _FEATURES if feature.group == group)
    for group in dict.fromkeys(feature.group for feature in _FEATURES)
}
RDP_VE_FEATURES = tuple(feature.name for feature in _FEATURES if feature.orientation == "VE")
RDP_SE_FEATURES = tuple(feature.name for feature in _FEATURES if feature.orientation == "SE")
RDP_FEATURE_METADATA = {feature.name: feature for feature in _FEATURES}

assert len(RDP_FEATURES) == len(set(RDP_FEATURES)) == RDP_DIM
assert len(RDP_VE_FEATURES) + len(RDP_SE_FEATURES) == RDP_DIM
