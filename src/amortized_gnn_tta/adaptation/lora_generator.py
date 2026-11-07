"""Generated LoRA factors around a reproducible identity initialization.

Independently implemented using the existing WeightAdapter's LoRA convention:
W' = W + (alpha/r) B (A0 + dA). Generators predict [dA, B], never a dense
unconstrained update. Dense output is an explicit compatibility boundary for the
shared functional forward/evaluator; non-target parameters get exact zero deltas.
"""

from dataclasses import asdict
import math

import torch
from torch import nn

from ._validation import finite
from .iterative import PEFTConfig, WeightAdapter
from .generator import extract_target_parameters


class LoRAUpdateSpace(nn.Module):
    """Named factor schema and fixed seeded A0 buffers, portable across devices.

    Factor coordinates are not unique: BQ and Q^-1 A can describe the same
    dense update. A shared initialization makes teacher coordinates traceable;
    it does not solve this symmetry or make factor MSE a functional distance.
    """

    def __init__(self, schema, targets, peft_config, initialization_seed=801):
        super().__init__()
        self.peft = PEFTConfig(**peft_config)
        if self.peft.method != "lora":
            raise ValueError("Generated LoRA requires a LoRA configuration")
        if type(initialization_seed) is not int or not 0 <= initialization_seed < 2**32:
            raise ValueError("Invalid LoRA initialization seed")
        self.initialization_seed = initialization_seed
        self.schema = [(name, tuple(shape)) for name, shape in schema]
        self.targets = list(targets)
        if len(dict(self.schema)) != len(self.schema) or len(set(targets)) != len(targets) or not targets:
            raise ValueError("Duplicate/empty LoRA schema")
        shapes = dict(self.schema)
        self.factor_schema, self.scales = [], {}
        rng = torch.Generator(device="cpu").manual_seed(initialization_seed)
        for i, name in enumerate(targets):
            if name not in shapes or len(shapes[name]) != 2:
                raise ValueError("Unknown/non-matrix LoRA target")
            rows, columns = shapes[name]
            rank = self.peft.backbone_rank if name.startswith("conv1.") else self.peft.head_rank
            alpha = self.peft.backbone_alpha if name.startswith("conv1.") else self.peft.head_alpha
            if rank >= min(rows, columns):
                raise ValueError("LoRA rank must be below both matrix dimensions")
            reference = torch.empty(rank, columns)
            nn.init.kaiming_uniform_(reference, a=math.sqrt(5), generator=rng)
            self.register_buffer(f"a0_{i}", reference)
            self.scales[name] = alpha / rank
            self.factor_schema.extend([(f"a.p{i}", (rank, columns)), (f"b.p{i}", (rows, rank))])
        self.factor_dim = sum(math.prod(shape) for _, shape in self.factor_schema)
        self.dense_dim = sum(math.prod(shape) for _, shape in self.schema)

    @classmethod
    def from_model(cls, model, peft=None, initialization_seed=801):
        peft = peft or PEFTConfig("lora")
        adapter = WeightAdapter(model, peft, seed=initialization_seed)
        _, schema = extract_target_parameters(model, "all")
        result = cls(schema, adapter.target_names, asdict(peft), initialization_seed)
        # Reject silent drift between this representation and the iterative API.
        for i in range(len(result.targets)):
            torch.testing.assert_close(getattr(result, f"a0_{i}"), adapter.a[f"p{i}"], rtol=0, atol=0)
        return result

    def configuration(self):
        return dict(schema=self.schema, targets=self.targets, peft_config=asdict(self.peft),
                    initialization_seed=self.initialization_seed)

    def factors(self, vector):
        if vector.ndim != 1 or vector.numel() != self.factor_dim:
            raise ValueError("LoRA factor vector dimension mismatch")
        finite(vector, "LoRA factors")
        result, offset = {}, 0
        for i, name in enumerate(self.targets):
            a_shape, b_shape = self.factor_schema[2*i][1], self.factor_schema[2*i+1][1]
            a_count, b_count = math.prod(a_shape), math.prod(b_shape)
            reference = getattr(self, f"a0_{i}")
            if reference.device != vector.device or reference.dtype != vector.dtype:
                raise ValueError("LoRA reference dtype/device mismatch")
            result[name] = (reference + vector[offset:offset+a_count].view(a_shape),
                            vector[offset+a_count:offset+a_count+b_count].view(b_shape))
            offset += a_count+b_count
        return result

    def dense_delta(self, vector):
        factors = self.factors(vector)
        updates = {name: (b @ a)*self.scales[name] for name, (a, b) in factors.items()}
        return finite(torch.cat([(updates[name] if name in updates else vector.new_zeros(shape)).flatten()
                                  for name, shape in self.schema]), "LoRA dense delta")

    def encode_adapter(self, adapter):
        if adapter.config != self.peft or list(adapter.target_names) != self.targets:
            raise ValueError("Teacher LoRA configuration disagrees")
        pieces = []
        for i in range(len(self.targets)):
            a, b = adapter.a[f"p{i}"], adapter.b[f"p{i}"]
            reference = getattr(self, f"a0_{i}").to(a)
            pieces.extend([(a-reference).flatten(), b.flatten()])
        vector = torch.cat(pieces).detach()
        # Check that offset encoding preserves the actual teacher weight product.
        expected = self.dense_delta(vector.to(next(self.buffers()).device))
        updates = adapter.updates()
        dense = torch.cat([(updates[name] if name in updates else vector.new_zeros(shape)).flatten()
                           for name, shape in self.schema])
        torch.testing.assert_close(expected.to(dense), dense, rtol=1e-5, atol=1e-6)
        return vector


class LoRAGenerator(nn.Module):
    """Factor generator plus explicit low-rank-to-dense functional boundary.

    core is the existing MLP or DDIM generator with output size factor_dim.
    Raw-weight conditioning is deliberately disabled in this first matched
    comparison; both update types receive the same descriptor information.
    """
    representation = "lora-offset-factors-v1"

    def __init__(self, core, update_space):
        super().__init__()
        if core.head_dim != update_space.factor_dim or core.include_head_weights:
            raise ValueError("LoRA core must generate factors without a raw-weight input")
        self.core, self.update_space = core, update_space
        self.descriptor_dim = core.descriptor_dim
        self.head_dim = update_space.dense_dim  # shared evaluator output contract
        self.latent_dim = update_space.factor_dim
        self.include_head_weights = False

    def forward_factors(self, descriptor):
        return self.core(descriptor)

    def forward(self, descriptor, current_weights=None):
        return self.update_space.dense_delta(self.forward_factors(descriptor))

    def set_target_normalizer(self, **kwargs):
        return self.core.set_target_normalizer(**kwargs)

    def diffusion_loss(self, descriptor, weights, target_factors):
        return self.core.diffusion_loss(descriptor, None, target_factors)
