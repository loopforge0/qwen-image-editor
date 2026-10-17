"""Stream a model through the GPU one unit at a time, from pinned host memory.

Each unit's tensors live in pinned RAM. Before a unit runs, its tensors are copied to the GPU on a side
stream; the next unit's copy starts as soon as the current one begins computing, so transfer overlaps
compute. After a unit runs, its parameters point back at the pinned copies (no copy back, the weights
are read-only), and the GPU memory is released.

Used for the text encoder on cards that cannot hold it next to the transformer: the encoder runs once
per prompt, while the transformer runs every step, so the transformer is the one that stays resident.
"""

import torch
from torch import nn


_ALIGN = 256


def _pinned_arena(nbytes: int) -> torch.Tensor:
    """Page-locked host buffer of exactly ``nbytes``.

    ``Tensor.pin_memory()`` goes through PyTorch's caching host allocator, which rounds each allocation up to
    a power of two (a 100 MB weight takes 128 MB). Windows caps page-locked memory at about half of RAM, so
    that waste decides whether a model fits. Registering one plain buffer with CUDA pins it exactly.
    """
    buf = torch.empty(nbytes, dtype=torch.uint8)
    err = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), nbytes, 0)
    if int(err) != 0:
        raise torch.OutOfMemoryError(f"cudaHostRegister of {nbytes / 2**30:.2f} GiB failed ({err})")
    return buf


class _Unit:
    def __init__(self, module: nn.Module):
        self.module = module
        self.slots = []  # (owner module, attribute name, pinned host tensor, is_param)
        tensors = []
        for owner in module.modules():
            for name, p in owner.named_parameters(recurse=False):
                tensors.append((owner, name, p.data, True))
            for name, b in owner.named_buffers(recurse=False):
                if b is not None:
                    tensors.append((owner, name, b, False))
        sizes = [-(-t.numel() * t.element_size() // _ALIGN) * _ALIGN for _, _, t, _ in tensors]
        self.arena = _pinned_arena(max(sum(sizes), _ALIGN))
        offset = 0
        # Each pinned view replaces the original right away, so peak host memory stays ~1x the weights.
        for (owner, name, t, is_param), size in zip(tensors, sizes):
            nbytes = t.numel() * t.element_size()
            host = self.arena[offset : offset + nbytes].view(t.dtype).view(t.shape)
            host.copy_(t)
            offset += size
            if is_param:
                owner._parameters[name].data = host
            else:
                setattr(owner, name, host)
            self.slots.append((owner, name, host, is_param))
        self.nbytes = offset
        self.ready = None
        self.gpu = None

    def prefetch(self, device, stream):
        if self.gpu is not None:
            return
        # Allocate on the compute stream, copy on the side stream. The caching allocator keeps one pool per
        # stream, so allocating on the side stream would strand freed weight memory where activations can't
        # reuse it (seen as several GiB reserved but unallocated, which on Windows spills into system RAM).
        self.gpu = [torch.empty_like(host, device=device) for _, _, host, _ in self.slots]
        stream.wait_stream(torch.cuda.current_stream(device))  # the memory may have just been freed by compute
        with torch.cuda.stream(stream):
            for g, (_, _, host, _) in zip(self.gpu, self.slots):
                g.copy_(host, non_blocking=True)
            self.ready = torch.cuda.Event()
            self.ready.record(stream)

    def attach(self):
        torch.cuda.current_stream().wait_event(self.ready)
        for (owner, name, _, is_param), t in zip(self.slots, self.gpu):
            if is_param:
                getattr(owner, name).data = t
            else:
                setattr(owner, name, t)

    def release(self):
        if self.ready is not None:  # a prefetch dropped unused: its copy must land before the memory is reused
            torch.cuda.current_stream().wait_event(self.ready)
        for owner, name, host, is_param in self.slots:
            if is_param:
                getattr(owner, name).data = host
            else:
                setattr(owner, name, host)
        self.gpu = None
        self.ready = None

    def close(self):
        """Give the pinned RAM back. It was registered with CUDA directly, so freeing the tensor isn't enough."""
        if self.arena is not None:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaHostUnregister(self.arena.data_ptr())
            self.slots, self.gpu, self.arena = [], None, None


def close_units(units):
    for unit in units:
        unit.close()


class StreamedModule:
    """Attach streaming hooks to ``units`` (run in this order); everything else in ``root`` goes to ``device``."""

    def __init__(self, root: nn.Module, units: list[nn.Module], device="cuda", report=lambda nbytes: None):
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(self.device)
        self.units = []
        for m in units:
            self.units.append(_Unit(m))
            report(self.units[-1].nbytes)
        streamed = {id(p) for u in units for p in list(u.parameters()) + list(u.buffers())}
        for p in list(root.parameters()) + list(root.buffers()):
            if id(p) not in streamed:
                p.data = p.data.to(self.device)
        for i, unit in enumerate(self.units):
            nxt = self.units[i + 1] if i + 1 < len(self.units) else None
            unit.module.register_forward_pre_hook(lambda m, a, u=unit, n=nxt: self._before(u, n))
            unit.module.register_forward_hook(lambda m, a, o, u=unit: self._after(u))

    def _before(self, unit, nxt):
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)  # don't overwrite memory the compute stream still reads
        unit.prefetch(self.device, self.stream)
        unit.attach()
        if nxt is not None:
            nxt.prefetch(self.device, self.stream)

    def _after(self, unit):
        # Keep the GPU copies alive until the compute that reads them is done.
        for t in unit.gpu:
            t.record_stream(torch.cuda.current_stream(self.device))
        unit.release()


class PartiallyEvictable:
    """A resident model that can hand some of its VRAM to another stage and take it back on its next use.

    Only ``units`` (e.g. transformer blocks) can be evicted, last first, and only as many as asked for, so
    the pinned host copy (Windows caps page-locked memory at roughly half of RAM) stays as small as the
    eviction needs. Eviction is free (weights are read-only; they just point back at the pinned copy); the
    reload is one host-to-device copy at the start of the model's next forward.

    The same units can also be *streamed* during the model's own forward (``set_streamed``): each one is
    copied in just before it runs, overlapped with the previous unit's compute, and dropped right after. That
    makes room for a long sequence (an edit with reference images) at almost no speed cost, as long as a unit
    computes for longer than it takes to copy.
    """

    def __init__(self, root: nn.Module, units: list[nn.Module], max_evict_bytes: int, device="cuda"):
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(self.device)
        root.to(self.device)
        self.units, total = [], 0
        for module in reversed(units):
            if total >= max_evict_bytes:
                break
            unit = _Unit(module)  # pins a host copy; the module now points at it
            unit.prefetch(self.device, self.stream)
            unit.attach()
            total += unit.nbytes
            self.units.append(unit)
        self.unit_bytes = max((u.nbytes for u in self.units), default=0)
        self.streamed = []  # in execution order
        order = self.units[::-1]  # execution order (self.units is last-first)
        for i, unit in enumerate(order):
            nxt = order[i + 1] if i + 1 < len(order) else None
            unit.module.register_forward_pre_hook(lambda m, a, u=unit, n=nxt: self._before(u, n))
            unit.module.register_forward_hook(lambda m, a, o, u=unit: self._after(u))
        root.register_forward_pre_hook(lambda m, a: self.restore())

    def evict(self, nbytes: int) -> int:
        """Release units until at least ``nbytes`` are freed (or none are left). Returns bytes freed."""
        freed = 0
        for unit in self.units:
            if freed >= nbytes:
                break
            if unit.gpu is not None:
                self._drop(unit)
                freed += unit.nbytes
        return freed

    def set_streamed(self, count: int):
        """Stream the last ``count`` units through the GPU from now on; the rest stay resident."""
        count = max(0, min(count, len(self.units)))
        self.streamed = self.units[:count][::-1]
        for unit in self.streamed:
            if unit.gpu is not None:
                self._drop(unit)

    def restore(self):
        evicted = [u for u in self.units if u.gpu is None and u not in self.streamed]
        if evicted:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            for unit in evicted:
                unit.prefetch(self.device, self.stream)
            for unit in evicted:
                unit.attach()

    def _drop(self, unit):
        for t in unit.gpu:
            t.record_stream(torch.cuda.current_stream(self.device))
        unit.release()

    def _before(self, unit, nxt):
        if unit not in self.streamed:
            return
        self.stream.wait_stream(torch.cuda.current_stream(self.device))  # don't overwrite memory still being read
        unit.prefetch(self.device, self.stream)
        unit.attach()
        if nxt is not None and nxt in self.streamed:
            nxt.prefetch(self.device, self.stream)

    def _after(self, unit):
        if unit in self.streamed:
            self._drop(unit)


def stream_text_encoder(text_encoder: nn.Module, device="cuda", report=lambda nbytes: None) -> StreamedModule:
    """Qwen3-VL: vision tower and embeddings as single units, then one unit per decoder layer."""
    lm = text_encoder.model.language_model
    units = [text_encoder.model.visual, lm.embed_tokens, *lm.layers]
    return StreamedModule(text_encoder, units, device, report)
