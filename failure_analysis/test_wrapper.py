"""Small CPU fixtures only. No Arxiv training and no Modal submission."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data
from worker import Patience, attach_scale, fit_scale, run, load_for_analysis
from amortized_gnn_tta.models import GCN, GCNConfig
from amortized_gnn_tta.adaptation.generator import HeadAdapterGenerator
from amortized_gnn_tta.adaptation.evaluator import AdaptationRecord
from amortized_gnn_tta.data.api import DatasetBundle, GraphRecord
from amortized_gnn_tta.adaptation._validation import source_state
import run_arxiv_experiments as original_runner


class TinyStudy:
    def __init__(self, args):
        self.device = 'cpu'
        self.plan = dict(bank_kind='formal-real-v1', device='cuda',
                         generator_seeds=[202801], generator_lr=.001,
                         delta_l2=.001, lora_rank=1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(77)
            self.model = GCN(GCNConfig(in_channels=2, hidden_channels=4, out_channels=2))
        self.initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.graph = Data(x=torch.tensor([[1.,0.], [0.,1.], [1.,1.], [.1,.5]]),
                          edge_index=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]]), num_nodes=4)
        self.manifest = dict(environments=[dict(id=f'{role}-{i}', role=role,
                             family='control', severity=0.)
                             for role in ['meta_train', 'validation'] for i in range(7)])
        self.validation = self.manifest['environments'][7:]

    def sources(self, architecture, role):
        return {f'm{i}': self.model for i in range(18 if role == 'meta_train' else 6)}

    def loader(self, manifest, eid):
        role = 'meta_train' if eid.startswith('meta_train') else 'validation'
        splits = dict(train=torch.tensor([0,1]), val=torch.tensor([2]), test=torch.tensor([3]))
        if role == 'validation':
            splits = dict(eval=torch.tensor([2,3]))
        return DatasetBundle({eid: GraphRecord(self.graph, torch.tensor([0,1,0,1]), splits, role)}, ('a','b'))

    def evaluate(self, *args):
        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(v, self.initial_state[k], rtol=0, atol=0)
        return []  # Original selected-export evaluator is unchanged and tested upstream.


class Tests(unittest.TestCase):
    def test_patience_recovers_below_identity(self):
        p = Patience(2)
        self.assertFalse(p.update(.9))
        self.assertFalse(p.update(.8))
        self.assertFalse(p.update(.8))
        self.assertFalse(p.update(.7))
        self.assertFalse(p.update(.7))
        self.assertTrue(p.update(.7))

    def test_scaling_only_changes_input(self):
        torch.manual_seed(1)
        g = HeadAdapterGenerator(2, 3, include_head_weights=False)
        x = torch.tensor([1000., 2.])
        torch.testing.assert_close(g(x), torch.zeros(3), rtol=0, atol=0)
        with torch.no_grad():
            g.net[-1].weight.fill_(.01)
        expected = g(x / torch.tensor([1000., 2.]))
        handle = attach_scale(g, torch.tensor([1000., 2.]))
        torch.testing.assert_close(g(x), expected)
        self.assertTrue(torch.isfinite(g(x)).all())
        handle.remove()

    def test_training_only_rms_and_mode_restoration(self):
        study = TinyStudy(None)
        class Descriptor:
            def __call__(self, model, graph):
                return torch.tensor([1000., 0., -2.])
        study.model.train()
        state = fit_scale(study, Descriptor(), study.sources('gcn', 'meta_train'))
        self.assertEqual(state['training_pairs'], 126)
        torch.testing.assert_close(state['scale'], torch.tensor([1000., 1e-6, 2.]))
        self.assertTrue(study.model.training)

    def test_original_training_loop_callback_artifacts_and_early_stop(self):
        # Fixed validation ties force epoch zero selection and stop at epoch 11.
        def fixed_evaluate(model, mid, bundle, eid, metadata, *args, **kwargs):
            return AdaptationRecord(mid, eid, 'validation', 'control', 0., False,
                                    .5, .5, 0., 0.)
        repo = Path(original_runner.__file__).resolve().parents[1]
        from amortized_gnn_tta.adaptation import meta_trainer
        import amortized_gnn_tta.adaptation.benchmark as benchmark
        import amortized_gnn_tta.training as training
        original_fork = torch.random.fork_rng
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(original_runner, 'Study', TinyStudy), \
             patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'get_device_name', return_value='NVIDIA A10 fixture'), \
             patch.object(torch.cuda, 'current_device', return_value=0), \
             patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
             patch.object(torch.random, 'fork_rng', side_effect=lambda **kw: original_fork(devices=[])), \
             patch.object(benchmark, 'execution_hardware', return_value={'fixture':True}), \
             patch.object(training, 'software_versions', return_value={'fixture':True}), \
             patch('os.link', side_effect=PermissionError('fixture: volume rejects hard links')), \
             patch.object(meta_trainer, 'evaluate_adaptation_pair', fixed_evaluate):
            torch.set_num_threads(1)
            for arm, epochs in [('control',5), ('epochs',11), ('scaling',5)]:
                result = run(repo, Path(temporary), Path(temporary), Path(temporary),
                             arm, configurations='A-head')
                folder = Path(temporary) / arm / 'A-head'
                self.assertEqual(result[0]['epochs_completed'], epochs)
                self.assertEqual(result[0]['selected_epoch'], 0)
                self.assertEqual(result[0]['optimizer_steps'], 126*epochs)
                with (folder/'validation_cases.csv').open() as stream:
                    cases = list(csv.DictReader(stream))
                self.assertEqual(len(cases), 42*(epochs+1))
                self.assertEqual(max(int(r['epoch']) for r in cases), epochs)
                initial = torch.load(folder/'epoch_000.pt', weights_only=True)['state_dict']
                trained = torch.load(folder/'epoch_001.pt', weights_only=True)['state_dict']
                self.assertTrue(any(not torch.equal(initial[k],trained[k]) for k in initial))
                model = TinyStudy(None).model
                g,d,payload = load_for_analysis(folder, model)
                self.assertEqual(payload['provenance']['selected_epoch'],0)
                with source_state(model):
                    graph = TinyStudy(None).graph
                    torch.testing.assert_close(g(d(model,graph)), torch.zeros(g.head_dim), rtol=0, atol=0)
                self.assertIs(meta_trainer.evaluate_adaptation_pair, fixed_evaluate)
                # Exercise the LoRA wrapper/core hook and serialization path too.
                lora_result = run(repo, Path(temporary), Path(temporary), Path(temporary)/'lora',
                                  arm, configurations='A-lora')
                lora_folder = Path(temporary)/'lora'/arm/'A-lora'
                lg,ld,lp = load_for_analysis(lora_folder, model)
                with source_state(model):
                    torch.testing.assert_close(lg(ld(model,graph)),torch.zeros(lg.head_dim),rtol=0,atol=0)
                self.assertEqual(lora_result[0]['epochs_completed'],epochs)
            # Longer budget changes no updates in its shared prefix with control.
            for epoch in range(6):
                a=torch.load(Path(temporary)/'control/A-head'/f'epoch_{epoch:03d}.pt',weights_only=True)
                b=torch.load(Path(temporary)/'epochs/A-head'/f'epoch_{epoch:03d}.pt',weights_only=True)
                for k in a['state_dict']:
                    torch.testing.assert_close(a['state_dict'][k],b['state_dict'][k],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
