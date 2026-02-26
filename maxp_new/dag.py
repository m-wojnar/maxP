"""DAG-based PM-to-PM data flow graph for per-op parametrization.

Traces the actual data flow between ParametrizedModule instances during a
forward pass, building a directed acyclic graph.  Each node corresponds to
one PM and carries its (a, b, alpha, omega, u) values.  Edges encode which
PM outputs feed into which PM inputs, with merge-type annotations (MIN for
addition/residual, SUM for elementwise multiplication).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class MergeType(Enum):
    """How multiple upstream signals combine before entering a PM.

    MIN — addition / residual: O(n^{-r1}) + O(n^{-r2}) = O(n^{-min(r1,r2)}).
    SUM — elementwise multiply: O(n^{-r1}) * O(n^{-r2}) = O(n^{-(r1+r2)}).
    """
    MIN = "min"
    SUM = "sum"


@dataclass
class DagNode:
    """One node in the PM data-flow graph."""
    name: str                          # module path, e.g. "blocks.0.mlp.gate"
    a: float
    b: float
    layer_type: str                    # "embedding", "hidden", "readout"
    has_weight: bool                   # pm.weight is not None
    width_dim: int
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    merge_type: MergeType = MergeType.MIN
    # Per-op alignment (preset for now, measured in Phase 2)
    alpha: float = 1.0
    omega: float = 0.5
    u: float = 1.0


class OpGraph:
    """Directed acyclic graph of PM-to-PM data flow."""

    def __init__(self, nodes: dict[str, DagNode] | None = None):
        self.nodes: dict[str, DagNode] = nodes or {}

    def topological_order(self) -> list[DagNode]:
        """Return nodes in topological order (Kahn's algorithm)."""
        in_degree: dict[str, int] = {name: 0 for name in self.nodes}
        for node in self.nodes.values():
            for succ in node.successors:
                in_degree[succ] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        result: list[DagNode] = []

        while queue:
            # Sort for deterministic ordering among nodes with same in-degree
            queue.sort()
            name = queue.pop(0)
            node = self.nodes[name]
            result.append(node)
            for succ in node.successors:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        return result

    def sources(self) -> list[DagNode]:
        """Nodes with no predecessors."""
        return [n for n in self.nodes.values() if not n.predecessors]

    def sinks(self) -> list[DagNode]:
        """Nodes with no successors."""
        return [n for n in self.nodes.values() if not n.successors]

    def validate(self) -> None:
        """Check the graph is acyclic and has sources + sinks."""
        self.topological_order()  # raises on cycle
        if not self.sources():
            raise ValueError("Graph has no source nodes (no predecessors)")
        if not self.sinks():
            raise ValueError("Graph has no sink nodes (no successors)")
        # Verify edge consistency
        for name, node in self.nodes.items():
            for succ in node.successors:
                if succ not in self.nodes:
                    raise ValueError(f"Successor '{succ}' of '{name}' not in graph")
                if name not in self.nodes[succ].predecessors:
                    raise ValueError(f"Inconsistent edge: '{name}' -> '{succ}'")
            for pred in node.predecessors:
                if pred not in self.nodes:
                    raise ValueError(f"Predecessor '{pred}' of '{name}' not in graph")
                if name not in self.nodes[pred].successors:
                    raise ValueError(f"Inconsistent edge: '{pred}' -> '{name}'")


# Default (a, b) per layer type — same as in parametrization.py
_DEFAULT_AB = {
    "embedding": (-0.5, 0.5),
    "hidden":    (0.0, 0.5),
    "readout":   (0.5, 0.5),
}


class _DagBuilder:
    """Registers hooks on ParametrizedModules to build a PM-to-PM DAG.

    Works with the _pm_tags provenance tracking on _TracingTensor:
    - Pre-hook: reads _pm_tags from input tensors -> those are predecessors.
    - Post-hook: sets output _pm_tags = frozenset({this_pm_name}).

    When tensors from different PM sources are combined between PMs (via
    add or mul), a synthetic merge node is created so that tags stay as a
    single element.  This keeps predecessor counts O(1) per node, avoiding
    O(depth^2) blowup from residual-stream tag accumulation.
    """

    def __init__(self):
        self.edges: dict[str, set[str]] = {}       # child -> set of parents
        self.merge_types: dict[str, MergeType] = {} # child -> merge type
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._merge_counter: int = 0
        # name -> (predecessors, merge_type)
        self._synthetic_nodes: dict[str, tuple[list[str], MergeType]] = {}

    def create_merge_node(self, input_tag_sets: list[frozenset[str]],
                          merge_type: MergeType) -> str:
        """Create a synthetic merge node and return its name.

        Called from ``_propagate_pm_tags`` when tensors with different PM
        provenance are combined.  The result tensor's tags are reset to
        ``{merge_name}`` so they never accumulate.
        """
        prefix = "_mul" if merge_type == MergeType.SUM else "_add"
        name = f"{prefix}_{self._merge_counter}"
        self._merge_counter += 1
        preds: list[str] = []
        for tag_set in input_tag_sets:
            for tag in sorted(tag_set):
                if tag not in preds:
                    preds.append(tag)
        self._synthetic_nodes[name] = (preds, merge_type)
        return name

    def _make_pre_hook(self, pm_name: str):
        def hook(module, args):
            # Collect _pm_tags from all input tensors
            all_tags: set[str] = set()
            for arg in _iter_tensors(args):
                tags = getattr(arg, '_pm_tags', frozenset())
                if tags:
                    all_tags.update(tags)

            if all_tags:
                self.edges.setdefault(pm_name, set()).update(all_tags)
                if len(all_tags) > 1:
                    self.merge_types.setdefault(pm_name, MergeType.MIN)

        return hook

    def _make_post_hook(self, pm_name: str):
        def hook(module, args, output):
            from maxp_new.trace import _TracingTensor
            new_tags = frozenset({pm_name})
            _set_pm_tags(output, new_tags)
        return hook

    def register(self, model: nn.Module, pm_names: dict[int, str]) -> None:
        """Register pre/post hooks on all ParametrizedModules."""
        from maxp_new.module import ParametrizedModule
        for name, mod in model.named_modules():
            if isinstance(mod, ParametrizedModule) and name in pm_names.values():
                self._hooks.append(
                    mod.register_forward_pre_hook(self._make_pre_hook(name))
                )
                self._hooks.append(
                    mod.register_forward_hook(self._make_post_hook(name))
                )

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def build_graph(self, pms: list[tuple[str, Any]],
                    ab: dict[str, tuple[float, float]] | None = None) -> OpGraph:
        """Build the OpGraph from recorded edges."""
        from maxp_new.module import ParametrizedModule
        ab = ab or _DEFAULT_AB

        nodes: dict[str, DagNode] = {}
        for name, pm in pms:
            lt = pm.layer_type
            a, b = ab.get(lt, (0.0, 0.5))
            nodes[name] = DagNode(
                name=name,
                a=a,
                b=b,
                layer_type=lt,
                has_weight=pm.weight is not None,
                width_dim=pm.width_dim,
            )

        # Create synthetic merge nodes (activation-only, a=0, b=0)
        for name, (preds, merge_type) in self._synthetic_nodes.items():
            nodes[name] = DagNode(
                name=name,
                a=0.0,
                b=0.0,
                layer_type="merge",
                has_weight=False,
                width_dim=0,
                merge_type=merge_type,
            )

        # Wire up synthetic node edges
        for name, (preds, _) in self._synthetic_nodes.items():
            for pred in preds:
                if pred not in nodes:
                    continue
                if pred not in nodes[name].predecessors:
                    nodes[name].predecessors.append(pred)
                if name not in nodes[pred].successors:
                    nodes[pred].successors.append(name)

        # Wire up PM edges (from pre-hooks)
        for child, parents in self.edges.items():
            if child not in nodes:
                continue
            for parent in parents:
                if parent not in nodes:
                    continue
                if parent not in nodes[child].predecessors:
                    nodes[child].predecessors.append(parent)
                if child not in nodes[parent].successors:
                    nodes[parent].successors.append(child)

            # Set merge type
            if child in self.merge_types:
                nodes[child].merge_type = self.merge_types[child]

        return OpGraph(nodes)


def _iter_tensors(obj) -> list:
    """Recursively extract tensors from nested args/kwargs."""
    results = []
    if isinstance(obj, torch.Tensor):
        results.append(obj)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            results.extend(_iter_tensors(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            results.extend(_iter_tensors(v))
    return results


def _set_pm_tags(obj, tags: frozenset[str]) -> None:
    """Set _pm_tags on all tensors in a nested output."""
    from maxp_new.trace import _TracingTensor
    if isinstance(obj, _TracingTensor):
        obj._pm_tags = tags
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _set_pm_tags(item, tags)


def trace_pm_dag(
    model: nn.Module,
    sample_input: torch.Tensor,
    ab: dict[str, tuple[float, float]] | None = None,
) -> OpGraph:
    """Trace PM-to-PM data flow and build a DAG.

    Args:
        model: Model containing ParametrizedModule instances.
        sample_input: Example input tensor for tracing.
        ab: Optional (a, b) overrides per layer_type.

    Returns:
        OpGraph with one node per ParametrizedModule.
    """
    from maxp_new.trace import _TracingTensor
    from maxp_new.module import ParametrizedModule

    # Discover PMs
    pms: list[tuple[str, ParametrizedModule]] = [
        (name, mod) for name, mod in model.named_modules()
        if isinstance(mod, ParametrizedModule)
    ]
    pm_names = {id(mod): name for name, mod in pms}

    builder = _DagBuilder()

    # Set up the dag builder on _TracingTensor
    old_builder = _TracingTensor._dag_builder
    _TracingTensor._dag_builder = builder

    try:
        builder.register(model, pm_names)

        # Wrap embedding weights so F.embedding triggers __torch_function__
        wrapped_embeddings: list[tuple[nn.Embedding, nn.Parameter]] = []
        for mod in model.modules():
            if isinstance(mod, nn.Embedding):
                original = mod.weight
                wrapped = _TracingTensor._wrap(original.data)
                mod.weight = nn.Parameter(wrapped, requires_grad=original.requires_grad)
                wrapped_embeddings.append((mod, original))

        with torch.no_grad():
            x = _TracingTensor._wrap(sample_input)
            model(x)

        # Restore embedding weights
        for mod, original in wrapped_embeddings:
            mod.weight = original

    finally:
        builder.remove_hooks()
        _TracingTensor._dag_builder = old_builder

    graph = builder.build_graph(pms, ab)
    graph.validate()
    return graph
