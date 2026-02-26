"""Trace forward pass matmuls to classify layers by width-scaling behavior."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TracedOp:
    """A single matmul-like operation recorded during tracing."""
    index: int
    op: str
    input_shapes: list[tuple[int, ...]]
    output_shape: tuple[int, ...]
    param_name: str | None = None
    module_path: str | None = None   # e.g. "blocks.0.attn"
    source_loc: str | None = None    # e.g. "transformer.py:44"


class _TracingTensor(torch.Tensor):
    """Tensor subclass that intercepts matmul ops via __torch_function__."""

    _tracer: "_MatmulTracer | None" = None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # Call the real function with unwrapped tensors
        raw_args = tuple(a._real if isinstance(a, _TracingTensor) else a for a in args)
        raw_kwargs = {k: v._real if isinstance(v, _TracingTensor) else v for k, v in kwargs.items()}
        out = func(*raw_args, **raw_kwargs)

        # Record if this is a matmul-like op
        tracer = cls._tracer
        if tracer is not None and func in _TRACED_OPS:
            tensor_inputs = [a for a in args if isinstance(a, (torch.Tensor, _TracingTensor))]
            raw_inputs = [a._real if isinstance(a, _TracingTensor) else a for a in tensor_inputs]
            raw_out = out._real if isinstance(out, _TracingTensor) else out
            tracer._record(_TRACED_OPS[func], raw_inputs, raw_out)

        # Wrap output (including tensors inside tuples/lists)
        return cls._wrap_output(out)

    @classmethod
    def _wrap(cls, t: torch.Tensor) -> "_TracingTensor":
        r = t.as_subclass(cls)
        r._real = t
        return r

    @classmethod
    def _wrap_output(cls, out):
        if isinstance(out, torch.Tensor) and not isinstance(out, _TracingTensor):
            return cls._wrap(out)
        if isinstance(out, (tuple, list)):
            wrapped = [cls._wrap_output(x) for x in out]
            return type(out)(wrapped)
        return out

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


# Which torch functions to trace and what to call them
_TRACED_OPS: dict[Any, str] = {
    torch.matmul: "matmul",
    torch.Tensor.matmul: "matmul",
    torch.mm: "mm",
    torch.bmm: "bmm",
    torch.addmm: "addmm",
    F.linear: "linear",
    F.embedding: "embedding",
}


class _MatmulTracer:
    """Context manager that records all matmul-like ops during a forward pass."""

    def __init__(self, model: nn.Module, record_activations: bool = False):
        self.model = model
        self.ops: list[TracedOp] = []
        self.record_activations = record_activations
        self.activation_stats: list[float] = []
        self._counter = 0
        self._param_ids: dict[int, str] = {}
        self._module_stack: list[str] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._module_names: dict[int, str] = {}
        self._path_to_module: dict[str, nn.Module] = {}
        self._wrapped_embeddings: list[tuple[nn.Embedding, torch.Tensor]] = []

    def __enter__(self):
        for name, p in self.model.named_parameters():
            self._param_ids[p.data_ptr()] = name

        # Wrap nn.Embedding weights as _TracingTensor so F.embedding
        # triggers __torch_function__ (the index input is plain ints,
        # so without this the embedding lookup is invisible to tracing).
        for mod in self.model.modules():
            if isinstance(mod, nn.Embedding):
                original = mod.weight
                wrapped = _TracingTensor._wrap(original.data)
                mod.weight = nn.Parameter(wrapped, requires_grad=original.requires_grad)
                self._wrapped_embeddings.append((mod, original))

        # Register hooks to track module context
        for name, mod in self.model.named_modules():
            self._module_names[id(mod)] = name
            self._path_to_module[name] = mod
            self._hooks.append(mod.register_forward_pre_hook(self._pre_hook))
            self._hooks.append(mod.register_forward_hook(self._post_hook))

        _TracingTensor._tracer = self
        return self

    def __exit__(self, *args):
        _TracingTensor._tracer = None
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        # Restore original embedding weights
        for mod, original in self._wrapped_embeddings:
            mod.weight = original
        self._wrapped_embeddings.clear()

    def _pre_hook(self, module, input):
        self._module_stack.append(self._module_names[id(module)])

    def _post_hook(self, module, input, output):
        if self._module_stack:
            self._module_stack.pop()

    def _record(self, op_name: str, inputs: list[torch.Tensor], output: torch.Tensor):
        param_name = None
        for t in inputs:
            ptr = t.data_ptr()
            if ptr in self._param_ids:
                param_name = self._param_ids[ptr]
                break

        # Find innermost non-root module
        module_path = None
        for path in reversed(self._module_stack):
            if path:  # skip root ""
                module_path = path
                break

        # Capture source location from call stack
        source_loc = _find_user_frame()

        self.ops.append(TracedOp(
            index=self._counter,
            op=op_name,
            input_shapes=[tuple(t.shape) for t in inputs],
            output_shape=tuple(output.shape),
            param_name=param_name,
            module_path=module_path,
            source_loc=source_loc,
        ))
        if self.record_activations:
            val = output.detach().abs().mean().item()
            # Apply scale from enclosing module (e.g. ParametrizedModule.scale)
            for path in reversed(self._module_stack):
                if path and path in self._path_to_module:
                    scale = getattr(self._path_to_module[path], 'scale', None)
                    if isinstance(scale, (int, float)) and scale != 1.0:
                        val *= abs(scale)
                        break
            self.activation_stats.append(val)
        self._counter += 1


# Directories to skip when looking for the user-code frame
_TRACE_INTERNALS = {os.path.dirname(__file__), os.path.dirname(torch.__file__)}


def _find_user_frame() -> str | None:
    """Walk the call stack to find the first frame outside torch/trace internals."""
    for info in inspect.stack():
        fpath = os.path.abspath(info.filename)
        if any(fpath.startswith(d) for d in _TRACE_INTERNALS):
            continue
        if "<" in info.filename:  # skip <string>, <frozen ...>, etc.
            continue
        basename = os.path.basename(info.filename)
        return f"{basename}:{info.lineno}"
    return None


def trace_forward(model: nn.Module, x: torch.Tensor) -> list[TracedOp]:
    """Trace all matmul ops in a single forward pass."""
    with torch.no_grad():
        with _MatmulTracer(model) as tracer:
            x_wrapped = _TracingTensor._wrap(x)
            model(x_wrapped)
    return tracer.ops


@dataclass
class ClassifiedOp:
    """A traced op with its width-scaling classification."""
    index: int
    op: str                              # "matmul", "linear", etc. or "elementwise"
    layer_type: str                      # "embedding", "hidden", "readout"
    parametrized: bool                   # has learnable weight?
    param_name: str | None               # name if parametrized
    module_path: str | None              # e.g. "blocks.0.attn"
    source_loc: str | None               # e.g. "transformer.py:44"
    input_shapes_small: list[tuple[int, ...]]
    input_shapes_large: list[tuple[int, ...]]
    output_shape_small: tuple[int, ...]
    output_shape_large: tuple[int, ...]


def classify(
    small_model: nn.Module,
    large_model: nn.Module,
    small_input: torch.Tensor,
    large_input: torch.Tensor,
) -> list[ClassifiedOp]:
    """Trace two model configs and classify every op.

    Matmul ops are classified by whether their contraction dim scales:
      - No scaling contraction dim → embedding
      - Contraction and output both scale → hidden
      - Contraction scales but output is fixed → readout

    Non-matmul learnable params (e.g. LayerNorm) → embedding.

    Each op is tagged as parametrized (has learnable weight) or not.
    """
    ops_s = trace_forward(small_model, small_input)
    ops_l = trace_forward(large_model, large_input)

    if len(ops_s) != len(ops_l):
        raise ValueError(f"Op count mismatch: {len(ops_s)} vs {len(ops_l)}. "
                         "Models must have the same architecture.")

    # Classify matmul ops
    result: list[ClassifiedOp] = []
    matmul_param_names: set[str] = set()

    for s, l in zip(ops_s, ops_l):
        layer_type = _classify_op(s, l)
        parametrized = s.param_name is not None
        if parametrized:
            matmul_param_names.add(s.param_name)

        result.append(ClassifiedOp(
            index=s.index,
            op=s.op,
            layer_type=layer_type,
            parametrized=parametrized,
            param_name=s.param_name,
            module_path=s.module_path,
            source_loc=s.source_loc,
            input_shapes_small=s.input_shapes,
            input_shapes_large=l.input_shapes,
            output_shape_small=s.output_shape,
            output_shape_large=l.output_shape,
        ))

    # Find non-matmul learnable params → embedding
    idx = len(result)
    for name, _ in small_model.named_parameters():
        # Strip ".weight" / ".bias" to get module name
        mod_name = name.rsplit(".", 1)[0] if "." in name else name
        if name not in matmul_param_names and mod_name not in {c.param_name for c in result}:
            # Check this specific param name hasn't been seen
            if name not in matmul_param_names:
                result.append(ClassifiedOp(
                    index=idx,
                    op="elementwise",
                    layer_type="embedding",
                    parametrized=True,
                    param_name=name,
                    module_path=mod_name,
                    source_loc=None,
                    input_shapes_small=[], input_shapes_large=[],
                    output_shape_small=(), output_shape_large=(),
                ))
                idx += 1

    return result


def _classify_op(small_op: TracedOp, large_op: TracedOp) -> str:
    """Classify an op by comparing dims across widths."""
    # Embedding lookup: no contraction over width-scaled dim → always embedding
    if small_op.op == "embedding":
        return "embedding"

    # For matmul/linear: contraction dim is last dim of first input
    s_shapes = small_op.input_shapes
    l_shapes = large_op.input_shapes

    contraction_small = s_shapes[0][-1]
    contraction_large = l_shapes[0][-1]
    contraction_scales = contraction_small != contraction_large

    out_small = small_op.output_shape[-1]
    out_large = large_op.output_shape[-1]
    output_scales = out_small != out_large

    if not contraction_scales:
        return "embedding"
    elif output_scales:
        return "hidden"
    else:
        return "readout"


def measure_activations(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Run a forward pass and return abs(output).mean() for each traced op."""
    with torch.no_grad():
        with _MatmulTracer(model, record_activations=True) as tracer:
            x_wrapped = _TracingTensor._wrap(x)
            model(x_wrapped)
    return tracer.activation_stats


def trace_pair(
    small_model: nn.Module,
    large_model: nn.Module,
    small_input: torch.Tensor,
    large_input: torch.Tensor,
) -> tuple[list[TracedOp], list[TracedOp]]:
    """Trace both models and return paired op lists."""
    return trace_forward(small_model, small_input), trace_forward(large_model, large_input)
