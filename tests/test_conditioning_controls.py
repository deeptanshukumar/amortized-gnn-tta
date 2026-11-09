"""The donor changes conditioning only; labels and source tensors stay excluded."""
import pytest
import torch
from torch_geometric.data import Data
from amortized_gnn_tta.models import make_model_spec, build_model
from amortized_gnn_tta.adaptation.experiments import make_descriptor
from amortized_gnn_tta.adaptation.generator import HeadAdapterGenerator
from amortized_gnn_tta.adaptation.lora_generator import LoRAUpdateSpace, LoRAGenerator
from amortized_gnn_tta.adaptation.iterative import PEFTConfig
from amortized_gnn_tta.adaptation.benchmark import generated_predictions
from amortized_gnn_tta.adaptation.conditioning_controls import donor_conditioning_predictions
from amortized_gnn_tta.adaptation_artifacts import state_sha256


@pytest.mark.parametrize('architecture',['gcn','gat','graphsage'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual CUDA device required'))])
def test_donor_conditioning_parity_label_exclusion_and_source_integrity(architecture, device):
    torch.manual_seed(71)
    target=Data(x=torch.randn(60,8),edge_index=torch.stack((torch.arange(59),torch.arange(1,60))),num_nodes=60)
    donor=target.clone(); donor.x=donor.x+2
    model=build_model(make_model_spec(architecture,8,4,hidden_channels=8)).to(device).eval()
    descriptor,_=make_descriptor(model,'D',scope='all')
    space=LoRAUpdateSpace.from_model(model,PEFTConfig('lora',backbone_rank=2,head_rank=2))
    core=HeadAdapterGenerator(descriptor.output_dim,space.factor_dim,include_head_weights=False)
    with torch.no_grad(): core.net[-1].weight.normal_(0,.005)
    generator=LoRAGenerator(core,space).to(device).eval()
    initial=state_sha256(model.state_dict()); original=target.x.clone()
    normal=generated_predictions(model,target,descriptor,generator,scope='all',device=device)
    same=donor_conditioning_predictions(model,target,target,descriptor,generator,device=device)
    changed=donor_conditioning_predictions(model,target,donor,descriptor,generator,device=device)
    torch.testing.assert_close(normal.logits,same.logits,rtol=1e-5,atol=1e-6)
    assert not torch.equal(normal.factor_vector,changed.factor_vector)
    donor.y=torch.full((60,),-1000,dtype=torch.long)
    hidden_labels=donor_conditioning_predictions(model,target,donor,descriptor,generator,device=device)
    torch.testing.assert_close(hidden_labels.logits,changed.logits,rtol=0,atol=0)
    assert torch.equal(target.x,original) and state_sha256(model.state_dict())==initial


def test_larger_descriptor_chunks_preserve_all_conditioning_statistics():
    torch.manual_seed(81)
    graph=Data(x=torch.randn(2200,8),edge_index=torch.randint(2200,(2,9100)),num_nodes=2200)
    model=build_model(make_model_spec('gcn',8,4,hidden_channels=8)).eval()
    small,_=make_descriptor(model,'D',scope='all',chunk_size=1024)
    large,_=make_descriptor(model,'D',scope='all',chunk_size=16384)
    torch.testing.assert_close(small(model,graph),large(model,graph),rtol=1e-4,atol=1e-5)
