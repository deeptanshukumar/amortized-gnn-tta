"""Whole-target prediction timing with explicit first-event/cache boundaries.

Method-required preprocessing is inside the timer. Disk loading, benchmark
construction, offline training, label scoring and report writing are outside.
Models are resident; target graphs start in canonical CPU memory. No graph-
dependent persistent cache is warmed outside an independent measured event.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import statistics
import time
import math

import torch
from torch_geometric.data import Data

from ..evaluation import validate_input, validate_logits
from ._validation import finite, positive_int, checked_backward, source_state
from .generator import extract_target_parameters
from .iterative import WeightAdapter, PEFTConfig, unlabeled_objective


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass
class PipelineResult:
    logits: torch.Tensor
    delta: torch.Tensor | None
    stages_seconds: dict


def amortization_break_even(*, offline_generated, offline_iterative, online_generated, online_iterative):
    """Approximate event count for constant per-event costs, not matched utility.

    Common source-bank costs cancel. Accuracy/label access must be reported
    alongside this cost-only calculation. No online savings means no positive
    amortization argument, regardless of generator-only microbenchmarks.
    """
    values = (offline_generated, offline_iterative, online_generated, online_iterative)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Costs must be finite and nonnegative")
    savings = online_iterative - online_generated
    if savings <= 0:
        return None
    return max(1, math.ceil(max(0., offline_generated-offline_iterative) / savings))


class StageTimer:
    def __init__(self, device, enabled):
        self.device, self.enabled, self.times = device, enabled, {}

    @contextmanager
    def stage(self, name):
        if self.enabled:
            synchronize(self.device)
            start = time.perf_counter()
        yield
        if self.enabled:
            synchronize(self.device)
            self.times[name] = time.perf_counter() - start


def _inputs(graph, device):
    # A new wrapper prevents .to from changing the canonical Data container;
    # read-only CPU tensors can be shared without an unnecessary clone.
    return Data(x=graph.x, edge_index=graph.edge_index, num_nodes=graph.num_nodes).to(device)


def generated_predictions(model, graph, descriptor, generator, *, scope="head", device="cpu", profile=False):
    timer = StageTimer(device, profile)
    if profile:
        synchronize(device)
    adaptation_start = time.perf_counter()
    with source_state(model), source_state(generator), torch.no_grad():
        with timer.stage("transfer_and_parameters"):
            inputs = _inputs(graph, device)
            weights, schema = extract_target_parameters(model, scope)
        with timer.stage("conditioning"):
            conditioning, hidden = descriptor.describe_with_hidden(model, inputs)
        with timer.stage("generation"):
            delta = finite(generator(conditioning, weights), "generated delta")
        # Applying the functional replacements and predicting are one operation;
        # split their costs explicitly without materializing a duplicate model.
        from .generator import _resolved_updates
        with timer.stage("apply_update"):
            updates = _resolved_updates(model, delta, schema)
            parameters = dict(model.named_parameters())
            parameters.update({name: parameters[name] + value for name, value in updates.items()})
        if profile:
            synchronize(device)
            timer.times["adaptation_ready"] = time.perf_counter() - adaptation_start
        with timer.stage("prediction"):
            from ..models import GCN, GAT, GraphSAGE
            if scope == "head" and hidden is not None and isinstance(model, (GCN, GAT, GraphSAGE)):
                head_parameters = {name.removeprefix("conv2."): value for name, value in parameters.items() if name.startswith("conv2.")}
                logits = torch.func.functional_call(model.conv2, head_parameters, (hidden, inputs.edge_index), strict=True)
            else:
                logits = torch.func.functional_call(model, parameters, (inputs.x, inputs.edge_index))
            validate_logits(logits, inputs.num_nodes)
    return PipelineResult(logits, delta, timer.times)


@torch.enable_grad()
def iterative_predictions(model, graph, *, steps=30, lr=.01, device="cpu", profile=False,
                          peft=None, objective="information_maximization", initialization_seed=801):
    """Existing head/LoRA optimizer without export/report bookkeeping.

    Uses the same WeightAdapter and objective as fit_unlabeled. Separate parity
    tests compare final predictions with that public fitting interface. Only
    head adaptation caches its frozen encoder; LoRA recomputes its changing
    encoder. Full supervised fine-tuning remains a separate diagnostic.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("Adaptation needs autograd; do not use inference_mode")
    positive_int(steps, "steps")
    peft = peft if peft is not None else PEFTConfig("head")
    if peft.method not in {"head", "lora"}:
        raise ValueError("Unlabeled timing supports head/LoRA; full is supervised only")
    if objective not in {"entropy", "information_maximization"}:
        raise ValueError("Unknown unlabeled objective")
    if isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    timer = StageTimer(device, profile)
    if profile:
        synchronize(device)
    adaptation_start = time.perf_counter()
    with timer.stage("transfer_and_setup"):
        validate_input(graph)
        inputs = _inputs(graph, device)
        adapter = WeightAdapter(model, peft, seed=initialization_seed).to(device).eval()
        adapter.prepare_episode(inputs.x, inputs.edge_index)
        parameters = [p for p in adapter.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0.)
    with timer.stage("optimization"):
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = adapter(inputs.x, inputs.edge_index)
            validate_logits(logits, inputs.num_nodes, model.config.out_channels)
            loss = unlabeled_objective(logits, objective)
            checked_backward(loss, parameters)
            optimizer.step()
    if profile:
        synchronize(device)
        timer.times["adaptation_ready"] = time.perf_counter() - adaptation_start
    with timer.stage("prediction"), torch.no_grad():
        logits = adapter(inputs.x, inputs.edge_index)
        validate_logits(logits, inputs.num_nodes, model.config.out_channels)
        delta = torch.cat([value.flatten() for value in adapter.updates().values()])
    return PipelineResult(logits, delta, timer.times)


def frozen_predictions(model, graph, *, device="cpu"):
    with source_state(model), torch.no_grad():
        inputs = _inputs(graph, device)
        logits = model(inputs.x, inputs.edge_index)
        validate_logits(logits, inputs.num_nodes)
    return PipelineResult(logits, None, {})


def measure_methods(methods, *, device="cpu", warmups=3, repeats=15, seed=919):
    """Interleave complete independent events; retain every raw wall time.

    First-event latency is recorded before warmups. It is not a fresh-process
    cold-start benchmark. Each callable must reset its own adaptation state.
    """
    positive_int(repeats, "repeats")
    if type(warmups) is not int or warmups < 0:
        raise ValueError("warmups must be a nonnegative integer")
    import random
    rng = random.Random(seed)
    output = {name: {"samples_seconds": []} for name in methods}
    names = list(methods)
    def timed(name):
        synchronize(device)
        start = time.perf_counter()
        result = methods[name]()
        synchronize(device)
        elapsed = time.perf_counter() - start
        # Validate outside timing too: an invalid fast result is never a result.
        finite(result.logits if isinstance(result, PipelineResult) else result, "timed output")
        return elapsed
    order = names.copy()
    rng.shuffle(order)
    for name in order:
        output[name]["first_event_seconds"] = timed(name)
    for _ in range(warmups):
        rng.shuffle(order)
        for name in order:
            timed(name)
    for _ in range(repeats):
        rng.shuffle(order)
        for name in order:
            output[name]["samples_seconds"].append(timed(name))
    for name, row in output.items():
        values = row["samples_seconds"]
        row.update(median_seconds=statistics.median(values), min_seconds=min(values), max_seconds=max(values),
                   stdev_seconds=statistics.stdev(values) if len(values) > 1 else 0.)
        if len(values) >= 4:
            quartiles = statistics.quantiles(values, n=4, method="inclusive")
            row["iqr_seconds"] = quartiles[2] - quartiles[0]
    return output


def memory_observation(function, *, device="cpu"):
    """Separate profiling pass; no profiler overhead in the main timing.

    CPU RSS is sampled (5ms), so peaks are lower bounds, not exact allocator
    peaks. psutil is optional; absence is explicitly reported. CUDA reports
    allocator peaks including resident tensors. Backend claims require execution.
    """
    import threading
    try:
        import psutil
    except ImportError:
        psutil = None
    rss = []
    stop = threading.Event()
    def sample():
        process = psutil.Process()
        while not stop.is_set():
            rss.append(process.memory_info().rss)
            stop.wait(.005)
    thread = threading.Thread(target=sample, daemon=True) if psutil else None
    synchronize(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if thread:
        thread.start()
    try:
        result = function()
        synchronize(device)
    finally:
        stop.set()
        if thread:
            thread.join()
    row = {"cpu_rss_sampling_interval_seconds": .005, "cpu_rss_peak_is_lower_bound": True,
           "cpu_rss_peak_bytes": max(rss) if rss else None,
           "cpu_rss_start_bytes": rss[0] if rss else None,
           "cpu_rss_status": "sampled whole-process resident memory; includes setup/resident state" if rss else "unavailable: optional psutil not installed"}
    if torch.device(device).type == "cuda":
        row.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return row
