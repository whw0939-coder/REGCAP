# 31-dimensional Regional Distributional Prior

RDP adds explicit code-distribution cues to learned region representations. Each AST-guided region receives 31 normalized features grouped by topic. The feature names and order below are the serialized `region_attr` schema.

`RDP_DIM = 31`.

| Index | Feature | Topical Group | Orientation | Definition |
| ---: | --- | --- | :---: | --- |
| 1 | `mem_ptr_deref_density` | Memory Safety | VE | Density of pointer dereference operations within the region. |
| 2 | `mem_array_access_density` | Memory Safety | VE | Density of array-style accesses. |
| 3 | `mem_memory_call_ratio` | Memory Safety | VE | Ratio of memory-related API calls among all calls in the region. |
| 4 | `mem_pointer_arith_density` | Memory Safety | VE | Density of pointer arithmetic patterns. |
| 5 | `mem_unguarded_ratio` | Memory Safety | VE | Ratio of memory operations not protected by nearby validation conditions. |
| 6 | `input_identifier_ratio` | Input Validation & Injection | VE | Ratio of identifiers associated with external-input semantics. |
| 7 | `input_dangerous_sink_ratio` | Input Validation & Injection | VE | Ratio of dangerous sink calls among all calls. |
| 8 | `input_format_string_risk_ratio` | Input Validation & Injection | VE | Ratio of format-related calls lacking literal format strings. |
| 9 | `input_to_sink_reachable_ratio` | Input Validation & Injection | VE | Ratio of sinks reachable from input sources via dependencies. |
| 10 | `input_fanout_ratio` | Input Validation & Injection | VE | Influence scope of input sources within the region. |
| 11 | `res_alloc_call_ratio` | Resource Lifecycle | VE | Ratio of allocation calls among all calls. |
| 12 | `res_release_call_ratio` | Resource Lifecycle | SE | Ratio of release calls among all calls. |
| 13 | `res_alloc_release_imbalance` | Resource Lifecycle | VE | Imbalance between allocation and release operations. |
| 14 | `res_exit_without_release_ratio` | Resource Lifecycle | VE | Likelihood of exit without proper release. |
| 15 | `res_dependency_depth` | Resource Lifecycle | VE | Average dependency depth from allocation to release. |
| 16 | `res_early_return_ratio` | Resource Lifecycle | VE | Frequency of early returns. |
| 17 | `ctrl_branch_density` | Control-Flow Complexity | VE | Density of branching structures. |
| 18 | `ctrl_loop_density` | Control-Flow Complexity | VE | Density of loops. |
| 19 | `ctrl_switch_density` | Control-Flow Complexity | VE | Density of switch structures. |
| 20 | `ctrl_cfg_cyclomatic` | Control-Flow Complexity | VE | Approximate cyclomatic complexity. |
| 21 | `ctrl_max_ast_nesting` | Control-Flow Complexity | VE | Maximum AST nesting depth. |
| 22 | `ctrl_cfg_branch_fanout` | Control-Flow Complexity | VE | Branch fan-out in CFG. |
| 23 | `arith_density` | Arithmetic & Type Safety | VE | Density of arithmetic operations. |
| 24 | `arith_cast_density` | Arithmetic & Type Safety | VE | Density of type casts. |
| 25 | `arith_size_computation_pattern_ratio` | Arithmetic & Type Safety | VE | Size-related arithmetic flowing to memory operations. |
| 26 | `struct_region_size_score` | Structural Context | SE | Relative size of the region. |
| 27 | `struct_region_span_score` | Structural Context | SE | Code span covered by the region. |
| 28 | `struct_ast_internal_edge_density` | Structural Context | SE | Density of AST internal edges. |
| 29 | `struct_call_density` | Structural Context | SE | Density of call nodes. |
| 30 | `struct_identifier_density` | Structural Context | SE | Density of identifiers. |
| 31 | `struct_cfg_internal_edge_density` | Structural Context | SE | Density of CFG internal edges. |

For the released implementation, RDP dimensions are organized into VE-oriented and SE-oriented subsets according to their semantic roles. The serialized RDP vector preserves the stable topical-group ordering above. VE/SE labels describe semantic orientation and are provided as metadata.
