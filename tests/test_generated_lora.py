"""Research contracts: identity, learnable zero start, factor/dense replay and labels."""
from dataclasses import replace

import pytest
import torch

from amortized_gnn_tta.adaptation import (
    HeadAdapterGenerator, DiffusionHeadGenerator, PEFTConfig, WeightAdapter,
    extract_target_parameters, adapted_forward, train_adaptation_generator,
    train_diffusion_generator, fit_oracle_deltas,
)
from amortized_gnn_tta.adaptation.lora_generator import LoRAUpdateSpace, LoRAGenerator
from amortized_gnn_tta.adaptation.descriptors import GraphShiftDescriptor
from amortized_gnn_tta.adaptation.checkpoints import save_generator_checkpoint, load_generator_checkpoint
from amortized_gnn_tta.adaptation.benchmark import generated_predictions
from amortized_gnn_tta.models import build_model, make_model_spec
from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset


def setup(architecture, kind="hypernetwork"):
    model = build_model(make_model_spec(architecture, in_channels=8, hidden_channels=8, out_channels=2)).eval()
    peft = PEFTConfig("lora", backbone_rank=2)
    space = LoRAUpdateSpace.from_model(model, peft, 801)
    descriptor = GraphShiftDescriptor(hidden_channels=8)
    cls = HeadAdapterGenerator if kind == "hypernetwork" else DiffusionHeadGenerator
    options = {} if kind == "hypernetwork" else dict(diffusion_steps=10, sampling_steps=3)
    generator = LoRAGenerator(cls(descriptor.output_dim, space.factor_dim, hidden_dim=12,
        include_head_weights=False, **options), space)
    config = CSBMConfig(num_nodes=40, num_features=8, p_in=.2, p_out=.04)
    bundle = load_dataset([CSBMEnvironment("train", "source_train", config),
        CSBMEnvironment("validation", "validation", replace(config, edge_seed=59, feature_seed=60))])
    manifest = {"environments": [{"id": "train", "role": "meta_train"}, {"id": "validation", "role": "validation"}]}
    return model, generator, descriptor, bundle, manifest


@pytest.mark.parametrize("architecture", ["gcn", "gat", "graphsage"])
def test_identity_gradient_and_manual_factor_reconstruction(architecture):
    model, generator, descriptor, bundle, _ = setup(architecture)
    space = generator.update_space
    graph = bundle.unlabeled_input("train")
    before = {k: v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    LoRAUpdateSpace.from_model(model, space.peft)
    assert torch.equal(rng, torch.get_rng_state())
    vector = torch.zeros(space.factor_dim, requires_grad=True)
    delta = space.dense_delta(vector)
    assert delta.eq(0).all()
    _, schema = extract_target_parameters(model, "all")
    logits = adapted_forward(model, graph, delta, schema)
    torch.testing.assert_close(logits, model(graph.x, graph.edge_index), rtol=0, atol=0)
    logits.square().mean().backward()
    assert vector.grad.abs().sum() > 0 and torch.isfinite(vector.grad).all()
    desc = descriptor(model, graph)
    optimizer = torch.optim.Adam(generator.parameters(), lr=.001)
    loss = adapted_forward(model, graph, generator(desc), schema).square().mean()
    loss.backward(); optimizer.step()
    assert generator(desc).abs().sum() > 0
    vector = torch.randn(space.factor_dim)*.01
    adapter = WeightAdapter(model, space.peft)
    offset = 0
    for i,name in enumerate(space.targets):
        a, b = adapter.a[f"p{i}"], adapter.b[f"p{i}"]
        with torch.no_grad():
            a.add_(vector[offset:offset+a.numel()].view_as(a)); offset += a.numel()
            b.copy_(vector[offset:offset+b.numel()].view_as(b)); offset += b.numel()
    encoded = space.encode_adapter(adapter)
    torch.testing.assert_close(encoded, vector)
    actual = space.dense_delta(encoded)
    updates = dict(zip([n for n,_ in schema], actual.split([torch.tensor(s).prod().item() for _,s in schema])))
    for name,(a,b) in space.factors(encoded).items():
        torch.testing.assert_close(updates[name].reshape_as(dict(model.named_parameters())[name]), (b@a)*space.scales[name])
        assert torch.linalg.matrix_rank((b@a).double(), atol=1e-6) <= a.shape[0]
    for name in set(updates)-set(space.targets):
        assert updates[name].eq(0).all()
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())
    for bad in (vector[:-1], vector.unsqueeze(0), vector*float("nan")):
        with pytest.raises(ValueError): space.dense_delta(bad)


@pytest.mark.parametrize("architecture", ["gcn", "gat", "graphsage"])
@pytest.mark.parametrize("kind", ["hypernetwork", "diffusion"])
def test_training_checkpoint_and_replay(tmp_path, architecture, kind):
    model, generator, descriptor, bundle, manifest = setup(architecture, kind)
    initial = {k: v.clone() for k,v in model.state_dict().items()}
    original_modes = [m.training for m in model.modules()]
    common = dict(models={"m":model}, manifest=manifest, load_env_fn=lambda m,e:bundle,
        descriptor=descriptor, generator=generator, epochs=2, val_models={"v":model},
        verbose=False, target_scope="all")
    fit = (train_adaptation_generator(**common) if kind == "hypernetwork" else
           train_diffusion_generator(**common, oracle_steps=2))
    generator.load_state_dict(fit.best_state_dict)
    graph = bundle.unlabeled_input("validation")
    output = generated_predictions(model, graph, descriptor, generator, scope="all")
    path = tmp_path/"generator.pt"
    save_generator_checkpoint(path, generator, descriptor, generator.update_space.schema,
        target_scope="all", provenance={"test":True})
    loaded, d, payload = load_generator_checkpoint(path, model=model)
    assert payload["update_representation"] == "lora-offset-factors-v1"
    replay = generated_predictions(model, graph, d, loaded, scope="all")
    torch.testing.assert_close(replay.delta, output.delta, rtol=0, atol=0)
    torch.testing.assert_close(replay.logits, output.logits, rtol=0, atol=0)
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in initial.items())
    assert [m.training for m in model.modules()] == original_modes
    assert all(p.grad is None for p in model.parameters())
    assert torch.isfinite(output.logits).all()


def test_teacher_role_and_factor_target_contract():
    model, generator, descriptor, bundle, manifest = setup("gcn")
    records = fit_oracle_deltas({"m":model}, manifest, lambda m,e:bundle, descriptor,
        steps=2, verbose=False, target_scope="all", update_space=generator.update_space)
    assert len(records) == 1 and records[0].environment_id == "train"
    assert records[0].target_delta.numel() == generator.update_space.factor_dim
    assert records[0].target_delta.abs().sum() > 0
    with pytest.raises(ValueError):
        fit_oracle_deltas({"m":model}, {"environments":[manifest["environments"][1]]},
            lambda m,e:bundle, descriptor, steps=2, verbose=False, target_scope="all", update_space=generator.update_space)


def test_checkpoint_rejects_rehashed_reference_tampering(tmp_path):
    from amortized_gnn_tta.adaptation_artifacts import state_sha256
    model, generator, descriptor, _, _ = setup("gcn")
    path = tmp_path/"g.pt"
    save_generator_checkpoint(path, generator, descriptor, generator.update_space.schema,
        target_scope="all", provenance={})
    payload = torch.load(path, weights_only=True)
    payload["state_dict"]["update_space.a0_0"].add_(1)
    payload["state_sha256"] = state_sha256(payload["state_dict"])
    torch.save(payload, path)
    with pytest.raises(ValueError, match="seeded reference"):
        load_generator_checkpoint(path, model=model)
