"""Task-agnostic AST region boundaries and diff-derived weak targets."""

from collections import defaultdict


STRUCTURAL_ANCHORS = frozenset({"METHOD", "BLOCK", "CONTROL_STRUCTURE"})


def weak_region_target(region_lines, vulnerable_lines):
    """Return Eq. (11): vulnerable lines divided by covered region lines."""
    covered = {int(line) for line in region_lines if line is not None and str(line).isdigit() and int(line) > 0}
    if not covered:
        return 0.0
    vulnerable = {int(line) for line in vulnerable_lines if line is not None and str(line).isdigit() and int(line) > 0}
    return len(covered & vulnerable) / len(covered)


def construct_regions(ast_nodes, ast_edges):
    """Partition an AST into source-ordered basic bodies and structural scopes."""
    nodes = {node["id"]: node for node in ast_nodes}
    if not nodes:
        return []
    position = {node["id"]: index for index, node in enumerate(ast_nodes)}
    children = defaultdict(list)
    parent_count = defaultdict(int)
    for source, target in ast_edges:
        if source in nodes and target in nodes:
            children[source].append(target)
            parent_count[target] += 1

    def number(value, default):
        try:
            result = int(value)
            return result if result > 0 else default
        except (TypeError, ValueError):
            return default

    def source_key(node_id):
        node = nodes[node_id]
        return (
            number(node.get("lineNumber"), 10 ** 9),
            number(node.get("columnNumber"), 10 ** 9),
            number(node.get("order"), 10 ** 9),
            position[node_id],
        )

    for parent in children:
        children[parent].sort(key=source_key)

    regions = []
    visited = set()

    def emit(node_ids):
        if not node_ids:
            return
        lines = sorted({number(nodes[node_id].get("lineNumber"), 0)
                        for node_id in node_ids} - {0})
        regions.append({"nodes": list(node_ids), "lines": lines})

    def visit(node_id, pending):
        if node_id in visited:
            return pending
        visited.add(node_id)
        if nodes[node_id].get("_label") in STRUCTURAL_ANCHORS:
            prefix = []
            if pending:
                if all(nodes[item].get("_label") in STRUCTURAL_ANCHORS for item in pending):
                    prefix = pending
                else:
                    emit(pending)
            current = prefix + [node_id]
            for child in children[node_id]:
                current = visit(child, current)
            emit(current)
            return []
        pending.append(node_id)
        for child in children[node_id]:
            pending = visit(child, pending)
        return pending

    roots = sorted((node_id for node_id in nodes if not parent_count[node_id]),
                   key=source_key)
    for root in roots:
        emit(visit(root, []))
    for orphan in sorted(nodes, key=source_key):
        if orphan not in visited:
            emit(visit(orphan, []))
    regions.sort(key=lambda region: min(source_key(node_id)
                                        for node_id in region["nodes"]))
    return regions
