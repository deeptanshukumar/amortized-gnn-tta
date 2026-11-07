# Adaptation references and attribution

Our project-specific PyG integration, supervision boundaries, experiment runner,
artifacts and tests are independently implemented. The following MIT-licensed
implementations ground the adaptation mechanisms; no reference package is a
runtime dependency. Separate unchanged clones live outside this repository.

| Reference | Inspected revision/files | What is adapted |
| --- | --- | --- |
| [Microsoft LoRA](https://github.com/microsoft/LoRA) | c4593f060e6a368d7bb5af5273b8e42810cdef90; loralib/layers.py | Frozen base plus alpha/r-scaled BA; Kaiming-uniform A and zero B initialization. Our functional effective-weight integration differs from its nn.Linear subclass and merge/unmerge mechanism. |
| [Tent](https://github.com/DequanWang/tent) | e9e926a668d85244c66a6d5c006efbd2b82e83e8; tent.py | Prediction-entropy objective and episodic reset principle. We update explicit PEFT sites, not normalization affine parameters; this is not a full Tent reproduction. |
| [SHOT](https://github.com/tim-learn/SHOT) | f7d555a0d53b525b885e5ef2a887a267a5be3c36; object/image_target.py, object/loss.py | Conditional entropy minus marginal entropy. We use stable log-softmax/log-sum-exp arithmetic, without SHOT clustering, pseudo-labels, schedules or image architecture. |
| [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric/blob/ea2c5f62d54c8b414dd3ea7139814a7c1ef83182/examples/gcn.py) | ea2c5f62d54c8b414dd3ea7139814a7c1ef83182; examples/gcn.py, LICENSE | Standard supervised Adam/backward/step with masked cross-entropy grounds the full-weight diagnostic. We start from pretrained weights, disable dropout, omit weight decay, and select minimum development-validation loss to match the existing adaptation protocol. We do not copy the example's dataset transforms, optimizer parameter groups or test-based reporting loop. |

Copyright notices supplied with these references:

- Copyright (c) Microsoft Corporation.
- Copyright (c) 2021 Dequan Wang and Evan Shelhamer.
- Copyright (c) 2020. (SHOT's upstream notice names no additional holder.)
- Copyright (c) 2023 PyG Team <team@pyg.org>.

The following permission/warranty text applies to the referenced MIT material:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Generator equations and timing references

The cosine schedule and x0/DDIM equations in the contributor's prototype were independently checked against [OpenAI improved-diffusion](https://github.com/openai/improved-diffusion/tree/1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5) and the [DDIM author implementation](https://github.com/ermongroup/ddim/tree/e8df66cf02e457719930748b5c26e37e3126dfdd). The repair retains the tested default equations and restricts unsupported short linear schedules. Reference checks are equation validation, not learned-generator performance evidence.

The new reliability guards, exact-schema integration, checkpoint recipes, bounded reductions and benchmark runner are project implementation work. Measurement design is grounded in [Matcha Appendix C.8](https://arxiv.org/abs/2410.06976), [GNN-Diff Appendix I.3](https://arxiv.org/abs/2410.05697), [HyperFlow Figure 4](https://arxiv.org/html/2504.15323), [computational-time-aware TTA](https://arxiv.org/html/2304.04795) and [PyTorch benchmarking documentation](https://docs.pytorch.org/docs/2.13/benchmark_utils.html). These works use different tasks and timing scopes; no upstream timer is copied or represented as an exact equivalent of our protocol. GNN-Diff uses validation labels for candidate selection; that selection rule is not imported into unlabeled deployment. All reference repositories remain external and unmodified.
