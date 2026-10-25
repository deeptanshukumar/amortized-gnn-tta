# AFML learning handbook: from graph data to our research experiment

**Current focus: CSBM only.** CSBM means *Contextual Stochastic Block Model*: a recipe for generating random graphs together with informative node features. Other datasets appear here so you can understand the research documents. We are not implementing them now.

This handbook is for learning, explaining the project to a research committee, and preparing for interviews. Start with the examples; return to the formulas after the ideas make sense. It covers our current questions and likely follow-ups, and will grow when the work exposes new ones.

**Status, 2026-10-25:** GCN/CSBM Phase 1 is complete, with 104 CPU tests passed. The full-budget thirty-model bank is trained, artifact-audited, split 18/6/6, and compared in weight/prediction/representation space. Source-only evaluation is recorded. Adaptation and the generator remain future work. We retain the explicit Matcha-code sampling convention. Sections 24–27 explain the milestones; numerical teaching examples are illustrations unless identified as executed checks.

The assistant previously recorded “real benchmark first” as your preference. That was a misinterpretation and has been corrected. Your instruction is to use CSBM now.

## Reading map

| If you want to understand… | Read |
| --- | --- |
| What the project is actually trying to do | Sections 1–2 |
| What a graph dataset contains | Sections 3–5 |
| What each dataset name in our documents means | Section 6 |
| What CSBM generates and why Matcha was investigated | Sections 7–8 |
| The degree discrepancy, with the full reasoning | Section 9 |
| Target labels, data leakage, splits, and graph visibility | Sections 10–12 |
| Independent graphs versus changing one existing graph | Section 13 |
| The proposed loader and how data reaches a GCN | Sections 14–15 |
| Seeds, tests, and fair research claims | Sections 16–18 |
| Important project choices and how to defend them | Sections 19–20 |
| Questions to paste into another ChatGPT conversation | Section 21 |
| Meanings of unfamiliar words and primary references | Sections 22–23 |
| Our working source GCN: every training/evaluation step, results, and saved files | Section 24 |
| Five source-label subsets: what changes, what stays fixed, and why | Section 25 |
| Our thirty-model bank: model identities, files, failures, and remaining limits | Section 26 |
| Phase 1 closure: grouped holdouts, CKA, bank audit, and what the actual results mean | Section 27 |

You do not need to memorize everything at once. After each section, explain the main idea without looking. An explanation in your own words is more useful than repeating a prepared paragraph.

## 1. What problem are we solving?

Imagine a model that learned to classify people in one social network. It uses each person's attributes and information from connected people. In a new network, people may have different attributes or different connection patterns. The old model may then make more mistakes.

That difference between the training setting and the new setting is called **distribution shift**. A *distribution* describes which kinds of observations are likely and how often they occur. For graphs, it can describe features, connections, labels, and their relationships.

Our question is:

> Can we train a second model beforehand that looks at a new unlabeled graph and the pretrained GNN, then predicts a small useful change to the GNN's weights?

A **weight** is a learned number inside a neural network. A **pretrained model** is a model whose weights have already been learned on earlier data. A **generator** here is a second neural network that produces weight changes rather than class labels.

### What “amortized” means

An *amortized* method pays a large preparation cost once, hoping to reduce repeated costs later.

| Approach | Before deployment | When a new target graph arrives |
| --- | --- | --- |
| Iterative adaptation | Train the original model | Repeatedly calculate an objective and optimize some model parameters |
| Our proposed amortized adaptation | Train the original model and an adaptation generator | Run the generator to predict an update, then use the updated GNN |

**Deployment** means using the trained system on new data. **Optimization** means adjusting parameters to improve an objective. An **objective** is a numerical quantity we try to minimize or maximize.

This is our research hypothesis, not a proven benefit. A generator can be slow or inaccurate too. We must measure accuracy, adaptation time, prediction time, and memory use.

### What we are doing right now

We are building the foundation that will supply correctly structured graphs. We are not yet training that generator.

A broken data pipeline could produce an apparently successful adaptation method simply by exposing target labels or changing the benchmark. Getting the loader right protects the later scientific claim.

## 2. Where GCN and the loader fit into the whole project

**GNN** means *Graph Neural Network*: a neural network that uses connections between objects. **GCN** means *Graph Convolutional Network*: one particular way to combine a node's information with its neighbors' information.

You own GCN implementation and experiments. Teammates handle GAT and GraphSAGE. The data representation should be shared.

| Stage | What it means | Current status |
| --- | --- | --- |
| Data pipeline | Generate/load graphs and keep labels, splits, and environments organized | Implemented and CPU-tested; CSBM only |
| Source GCN | Train a GCN using permitted source labels | Implemented and CPU-tested; Section 24 |
| Weight bank | Save multiple source-trained GCN parameter sets | Full-budget thirty-model bank audited, grouped, and compared; Sections 26–27 |
| Small adaptations | Change only the classifier or a small structured part of the weights | Later |
| Adaptation generator | Learn to predict those changes | Later |
| Evaluation | Score a frozen model on source-test and target labels | Basic baseline scoring implemented; research-wide holdouts remain to be designed |

The plan suggests 5 labeled-source splits × 6 model seeds = 30 models, then 18/6/6 models for generator training/validation/testing. This bank preparation and model partition are now implemented and verified for GCN/CSBM. The generator has not been trained. A subset manifest alone still does not contain trained models.

**Do not confuse two generators:**

| Name | Produces | Example input |
| --- | --- | --- |
| CSBM data generator | Graph edges, features, and hidden ground-truth classes | Number of nodes, connection probabilities, feature parameters, random seeds |
| Adaptation generator | A change to GNN parameters | Permitted target-graph/model information |

The first creates a learning problem. The second is the eventual method being researched.

## 3. Graph vocabulary, starting from zero

A **graph** is a collection of objects and connections.

| Word | Plain meaning | Example |
| --- | --- | --- |
| Node, also called vertex | One object | A paper or a person |
| Edge | A connection between two nodes | A citation or friendship |
| Neighbor | A node directly connected to another node | A person's friend |
| Feature, also called attribute | A number describing a node | A word indicator or a synthetic measurement |
| Feature vector | The list of feature numbers for one node | [0.8, 0.2] |
| Label | The answer we want to predict | Research topic or class 0/1 |
| Class | One possible label category | “Machine learning” |
| Degree | Number of incident edges; for our simple undirected graphs, number of neighbors | Three friends gives degree 3 |
| Directed graph | Connections have an orientation | Paper A cites B; B need not cite A |
| Undirected graph | Connections work both ways | A friendship |
| Self-loop | A connection from a node to itself | Used by many GCN implementations to retain own-node information |
| Isolated node | A node with no edges to other nodes | A person with no recorded friends |
| Homogeneous graph | A graph using one node type and one edge type in this setting | All nodes are papers and all edges are citations |

An isolated node still exists and can have features and a label. We must not accidentally drop it.

### Three different prediction tasks

| Task | What receives an answer? | Example |
| --- | --- | --- |
| Node classification | Each selected node | Predict each paper's topic |
| Graph classification | A whole graph | Predict a molecule's property |
| Link prediction | A possible pair of nodes | Predict whether two people will connect |

Our current task is **node classification**. Generating multiple CSBM graphs does not turn it into graph classification: we still predict individual nodes' classes.

## 4. A complete tiny graph, written as data

Here is a hand-made teaching example, not a sample reported by the implemented generator.

~~~text
0 —— 1 —— 2       3

Node 3 is isolated.
~~~

Each node has two features and one class label:

| Node ID | Feature 1 | Feature 2 | Label |
| --- | --- | --- | --- |
| 0 | 1.0 | 0.0 | 0 |
| 1 | 0.8 | 0.2 | 0 |
| 2 | 0.0 | 1.0 | 1 |
| 3 | 0.1 | 0.9 | 1 |

A **node ID** is a row address, not automatically a meaningful model feature.

### Feature matrix X

A **matrix** is a rectangular table of numbers. Here rows are nodes and columns are features:

~~~text
X = [[1.0, 0.0],
     [0.8, 0.2],
     [0.0, 1.0],
     [0.1, 0.9]]

shape = [4, 2]
~~~

**Shape** describes the size of each dimension. More generally, X has shape [N, F], where N is the node count and F is the feature count per node.

### Edges as a list of endpoint pairs

For undirected message passing, we commonly store both directions:

~~~text
edge_index = [[0, 1, 1, 2],
              [1, 0, 2, 1]]

Columns:       0→1  1→0  1→2  2→1
shape = [2, 4]
~~~

For standard PyG message passing, the first row identifies sending nodes and the second receiving nodes.

There are **2 undirected edges** but **4 stored directed edge entries**. Always state which count you report.

Node 3 does not appear in the edge list. The record must still say num_nodes = 4. Inferring node count from the largest edge endpoint would incorrectly give 3 in this example.

### Labels are a separate vector

~~~text
labels = [0, 0, 1, 1]
shape = [4]
~~~

A **vector** is a one-dimensional list of numbers. A **tensor** is the general programming term for arrays of numbers: vectors and matrices are examples.

A GCN's prediction function receives features and edges. The training code separately receives the permitted labels to calculate a loss.

### Why not a large adjacency matrix?

An **adjacency matrix** A is an N-by-N table: entry A[i,j] records a connection. For this undirected example:

~~~text
A = [[0, 1, 0, 0],
     [1, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 0]]
~~~

It is useful for explaining mathematics. But a graph can have many nodes and relatively few edges.

**Sparse** means storing mostly the existing connections instead of all the absent ones. An edge list takes space proportional to the edge count. A dense adjacency matrix takes space proportional to N².

With 100,000 nodes, a dense matrix contains 10 billion entries—about 40 GB for 32-bit numbers, before other computation. We will keep normal graph connectivity sparse.

Source for PyG's representation and explicit node-count behavior: [official Data documentation](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.data.Data.html).

## 5. What is a dataset, a split, and an environment?

These words describe different things.

| Object | Meaning in our project | Example |
| --- | --- | --- |
| Graph | One set of nodes, edges, and features | One generated CSBM graph |
| Dataset | An organized collection of graph data and associated information | Several CSBM graphs with their labels and settings |
| Split | A selection of examples assigned a purpose | Source nodes allowed for training |
| Environment | A setting with a particular data-generating distribution | High same-class connectivity with one feature distribution |
| Realization | One actual random sample from a distribution | A particular graph drawn with a particular seed |
| Protocol | The complete rules for training, observing data, adapting, and evaluating | Which labels and graph context are allowed in each phase |
| Benchmark | A specified task/data/protocol used for comparison | A named dataset version and its prescribed evaluation |

Two different random graphs can come from the same environment. Conversely, two graphs with the same size can come from very different environments.

**Source** means the original training setting. **Target** means the setting to which we want to transfer/adapt. Neither word alone tells us which labels are permitted; the protocol must do that.

For CSBM, a useful mental model is:

~~~text
Generation settings + seed
          ↓
One graph realization
          ↓
Features and edges ─────────→ GCN inputs
Labels ────────────────────→ permitted training or separate evaluation
Split/environment records ─→ rules selecting which examples may be used
~~~

There is no universal “official CSBM train/test split.” We must name and document our own development convention.

## 6. What each dataset name in the documents means

Only **CSBM is in the current implementation scope**. This section provides background, not a request to add other datasets.

### Dataset atlas

| Name | What nodes and edges represent | What features and labels mean | Why researchers use it |
| --- | --- | --- | --- |
| CSBM | Artificial nodes; edges sampled using hidden class memberships | Artificial numerical features correlated with the hidden classes; labels are those classes | Control the difficulty and type of graph change |
| Cora, commonly the small citation benchmark | Papers connected by citations | Word indicators; paper-topic labels | A small standard node-classification example |
| GOOD-Cora | Full Cora citation data, with GOOD's own domain splits | Paper features and topic labels; domains include degree or word-related variation | Study out-of-distribution performance with a specified split |
| Twitch-E / Twitch-Explicit | Users connected in language-specific social networks | A shared set of 3,170 binary features in the inspected EERM loader; binary explicit/mature-content target | Transfer between separate language graphs |
| PyG Twitch | Six language-specific user graphs in that distribution | 128-dimensional feature embeddings; binary mature-content labels | Convenient graph data, but a different feature representation from Twitch-E |
| GOOD-Twitch | Language graphs assembled into a benchmark graph with separate components | PyG Twitch features plus GOOD's language-domain splits | Evaluate a defined language shift |
| Standard ogbn-arxiv | Computer-science papers with directed citation edges | 128 numerical features derived from text; 40 topic classes | A larger benchmark with chronological node splits |
| GOOD-Arxiv | Arxiv data organized using GOOD's domains and splits | Related underlying data but a different benchmark protocol | Study time/degree-related OOD settings |
| EERM/Matcha Arxiv protocol | Year-bounded cumulative citation graphs, with evaluation masks | Arxiv features/labels; Matcha also constructs edge-deleted targets | Test a paper-specific temporal/structural transfer setting |
| Syn-Cora / Syn-Products | Synthetic graph constructions using real-data-derived node features | Attributes derived from citation/product data and corresponding classes | Control graph connectivity while retaining real-data-derived attributes |

**Binary feature** means a 0/1 indicator. **Embedding** means a numerical representation produced by another process, often learned. **Chronological** means ordered by time. **OOD** means *out of distribution*: outside the training setting's distribution.

Standard ogbn-arxiv uses labels through 2017 for training, 2018 for validation, and 2019 onward for testing. Those are node-label splits on a graph, not automatically three separate disjoint graphs. GOOD and EERM/Matcha definitions must be checked independently.

### Why similarly named datasets are not interchangeable

“Twitch” does not specify the feature construction, graph preprocessing, language mapping, or evaluation protocol. Likewise, GOOD-Cora is not just small Cora with another filename.

A common interface promises that the rest of our code can consume a consistent record. It does **not** promise that all datasets represent the same problem.

**Metric** means a rule for scoring predictions. **Accuracy** is the fraction of correct class predictions. **ROC-AUC** measures how well binary scores rank positives above negatives across thresholds; for the usual binary interpretation it is the probability that a random positive is ranked above a random negative, with ties handled appropriately. EERM Twitch and GOOD-Twitch use ROC-AUC; Matcha's inspected Twitch experiment uses accuracy. These scores cannot be substituted silently.

Sources: [SNAP Twitch description](https://snap.stanford.edu/data/twitch-social-networks.html), [EERM protocol code](https://github.com/qitianwu/GraphOOD-EERM/blob/ffdc4a11161976fac7dd71e2aa1dcd72db6e44e9/multigraph/main.py), [PyG Twitch source](https://github.com/pyg-team/pytorch_geometric/blob/ea2c5f62d54c8b414dd3ea7139814a7c1ef83182/torch_geometric/datasets/twitch.py), [GOOD dataset descriptions](https://good.readthedocs.io/en/latest/_autosummary/GOOD.data.good_datasets.html), and [OGB protocol](https://ogb.stanford.edu/docs/nodeprop/). For Syn-Cora/Syn-Products, see the [Matcha paper, Appendix D.1](https://arxiv.org/abs/2410.06976) and [its dataset notes](https://github.com/baowenxuan/Matcha#prepare-the-datasets).

## 7. CSBM: how do we manufacture the data?

Break the name into pieces:

- **Stochastic:** involves randomness.
- **Block:** nodes belong to groups, also called communities.
- **Model:** a mathematical recipe.
- **Contextual:** nodes also have features connected to their groups.

A basic stochastic block model produces connections based on group membership. Adding informative node features gives the contextual version.

### Step 1: choose the hidden classes

For a simple two-class balanced setup, half the nodes belong to class 0 and half to class 1. **Balanced** means the classes have equal counts here.

The graph-generation program knows these classes. The prediction model must not be given held-out class labels as inputs.

### Step 2: draw edges

Choose:

- p_in: probability of an edge between two different nodes in the same class.
- p_out: probability of an edge between nodes in different classes.

For every unordered pair, make a random yes/no decision. A yes/no random variable is called **Bernoulli**. With probability 0.04, an edge appears about 4% of the time over many draws, not exactly four times in every hundred pairs.

| Connection setting | What it encourages |
| --- | --- |
| p_in > p_out | Same-class connections |
| p_in < p_out | Different-class connections |
| p_in = p_out | Edges do not depend on class membership in this balanced basic model |

**Homophily** means a tendency to connect to similar nodes; here “similar” specifically means the same class label. **Heterophily** means a tendency to connect across different classes. The reference point depends on class proportions; 50% is a useful approximate boundary for the balanced two-class setup, not a universal threshold.

### Step 3: draw features

Give each class a feature-center vector and add random noise around it.

~~~text
node feature vector = its class's center + random noise
~~~

A **class center** is the mean feature vector for that class. **Noise** is the random variation around it.

For illustration, class 0 could have center [1,0] and class 1 center [0,1]. Samples might lie near those centers rather than match them exactly.

A **Gaussian distribution**, also called a normal distribution, is a common bell-shaped model for numerical variation. In several dimensions it describes a cloud of feature vectors. **Variance** describes the spread in one direction; **covariance** also describes how directions vary together.

The exact feature-center and noise formulas are part of the generator definition. Our initial implementation uses the released Matcha-code formula, with explicit class-center coefficients and noise_scale.

### What differs between original-style and Matcha features?

The first formula is our selected initial distribution; the second explains an alternative:

~~~text
Matcha code:
    x_i = a_class * u + z_i / sqrt(F)
    u = [1/sqrt(F), ..., 1/sqrt(F)]

Original paper's model:
    x_i = sqrt(mu/N) * s_i * v + z_i / sqrt(F)
    v is drawn from a centered Gaussian with covariance I/F
~~~

Here x_i is node i's feature vector; F is its feature count; z_i has independent Gaussian coordinates with mean 0 and variance 1; a_class is the selected class's center coefficient; s_i is a hidden class sign, +1 or −1; mu controls the original model's feature signal; and v is a shared hidden feature direction. A **coordinate** is one position in the feature vector.

The symbol u in the Matcha code is fixed, while v in the original model is randomly drawn. Symbols such as mu do not automatically have the same meaning across papers or code. A **signal** is useful predictive information; **signal-to-noise ratio** compares useful information with random variation, under a specified definition.

For transfer experiments we must decide whether source and target share their feature direction. Drawing a new direction for every graph can rotate the relationship between features and classes. That is a different task from changing connectivity while keeping feature semantics fixed. We should not make this decision accidentally through seed handling.

### Is generating features from labels cheating?

Not by itself. In a synthetic experiment, hidden classes are part of the simulated world's construction. They cause observable patterns, just as a real paper's topic influences its words.

The distinction is between **creating the simulated world offline** and **giving its hidden answers to the prediction/adaptation algorithm**. The model may observe the generated features and edges; target labels stay with the evaluator.

Node IDs must not become a shortcut to the answer. For example, if generation stores all class-0 nodes first, we must not feed “row number” as a predictive feature or use an unexamined first-rows-only split.

### What can we control?

| Quantity | Intuition | Why it matters |
| --- | --- | --- |
| N, node count | How many objects exist | Runtime, graph statistics |
| F, feature count | How many measurements describe each object | Input size |
| Class balance | How common each answer is | Baseline difficulty and edge homophily |
| Within/between connection probabilities | Who tends to connect | Structural signal |
| Feature-center separation | How different class feature clouds are | Feature signal |
| Noise scale | How blurry the feature signal is | Prediction difficulty |
| Random seeds | Which particular samples are drawn | Reproducibility |

A graph can be easy from features alone, easy from edges alone, or require both. We should eventually compare against an **MLP**, a neural network that uses node features without graph message passing, to check whether connectivity matters.

Source: [original Contextual Stochastic Block Models paper](https://papers.nips.cc/paper/2018/file/08fc80de8121419136e443a70489c123-Paper.pdf). This section describes the family; it does not claim all implementations use the same formulas.

## 8. Why did we start investigating Matcha?

The chain was:

~~~text
Your proposal and resource index name Matcha
                 ↓
Matcha studies structure shifts and test-time adaptation
                 ↓
Its official repository includes a CSBM generator
                 ↓
Its README credits GPRGNN for that generator's lineage
                 ↓
We compare the paper, generator code, GPRGNN, and original CSBM definitions
~~~

That is why Matcha was a sensible investigation starting point. It was not selected because CSBM belongs to Matcha or because our project must reproduce it.

### Four references, four different roles

| Reference | What it helps us understand | What it does not establish |
| --- | --- | --- |
| Original CSBM paper | Mathematical graph-plus-feature model | A universal node-label split or every later variant |
| GPRGNN repository | An established graph-learning implementation with a CSBM generator | That Matcha retained all its formulas unchanged |
| Matcha paper/repository | Concrete controlled shifts and a relevant adaptation setting | That its parameters and protocols are automatically right for us |
| Original GCN paper / PyG GCNConv | The standard GCN computation we plan to implement with a maintained library | That Matcha's particular adapter transfers directly to our architecture |

**GPRGNN** means *Generalized PageRank Graph Neural Network*. For now, think of it as another architecture with adjustable contributions from different neighborhood distances.

Matcha adjusts **hop-aggregation parameters**: numbers controlling how information from different graph distances is combined. A **hop** is one edge traversal. Its main experiments use GPRGNN.

### An important difference from standard GCN

Matcha's simplified analysis uses a form like:

~~~text
Z = X + gamma × D^(-1) A X
~~~

That means “own features plus gamma times average neighbor features.” Its gamma controls the mixture.

Standard Kipf–Welling GCN/PyG GCNConv uses symmetric degree normalization with self-connections, explained in section 15. These formulas are related graph-aggregation ideas, but are not identical operators.

Therefore, Matcha's theoretical results do not automatically prove that a low-rank update to a standard GCN will recover accuracy. That remains a research question.

### What “compatible with Matcha” could mean

| Possible meaning | What would need to match? |
| --- | --- |
| Distribution-compatible | Edge probabilities, class construction, feature distribution |
| Protocol-compatible | Source/target settings, supervision, evaluation rules |
| Exact artifact reproduction | Actual graph realizations, random-number behavior, cached files |
| End-to-end result reproduction | Data, model, adapter, optimization, selection, metrics, and repeated runs |

Matching one level does not imply the others. Our approved 60/20/20 source development split already differs from Matcha's inspected full-source-label CSBM training routine.

Our present job is to choose a documented CSBM definition and build a correct loader. We are not committing to Matcha's adaptation algorithm.

Sources: [Matcha README](https://github.com/baowenxuan/Matcha), [Matcha CSBM code](https://github.com/baowenxuan/Matcha/blob/f0e7104a84d118a8dabb7ad210915c43722b94af/src/dataset/CSBM.py), [GPRGNN CSBM code](https://github.com/jianhao2016/GPRGNN/blob/4e0a7ee5435058b70eaec3c23c55fb96dc37f2d5/src/cSBM_dataset.py), and [GCN paper](https://arxiv.org/abs/1609.02907).

## 9. Issue 1: why does Matcha's degree parameter produce roughly half that degree?

### First understand expected degree

A random graph's observed degree varies. **Expected degree** is its mathematical average over repeated draws from the same recipe. It is not a promise about each node in one graph.

If a node has 10 possible neighbors and each edge appears with probability 0.2, its expected degree is 10 × 0.2 = 2. We add probabilities over possible neighbors.

### Count the possible neighbors correctly

Assume:

- N is even.
- Two classes each have N/2 nodes.
- No self-loops are generated.

For any one node:

| Other-node group | Potential neighbors |
| --- | --- |
| Same class | N/2 − 1, because the node cannot be its own other-node neighbor |
| Different class | N/2 |

Therefore, for any chosen probabilities:

~~~text
Expected degree = (N/2 − 1) × p_in + (N/2) × p_out
~~~

This finite-size expression is exact under these assumptions.

### Substitute the inspected Matcha code

The code sets:

~~~text
p_in  = d × h / N
p_out = d × (1 − h) / N
~~~

Here d and h are input parameters. Substituting:

~~~text
Expected degree
= (N/2 − 1) × d*h/N + (N/2) × d*(1−h)/N
= d*h/2 − d*h/N + d*(1−h)/2
= d/2 − d*h/N
~~~

For N=1000, d=10, h=0.8:

~~~text
p_in = 0.008
p_out = 0.002
Expected degree = 499 × 0.008 + 500 × 0.002 = 4.992
~~~

This is a calculation from the code, not a measured result. It is approximately 5 rather than 10.

### Why the paper interpretation differs

The Matcha paper describes approximately:

~~~text
d = N × (p_in + p_out) / 2
h = p_in / (p_in + p_out)
~~~

Ignoring the small no-self-loop correction, these imply:

~~~text
p_in  = 2*d*h/N
p_out = 2*d*(1−h)/N
~~~

The inspected code lacks that factor of 2. The discrepancy is between a parameter's described meaning and its sampling implementation. It does not prove that every released artifact or reported paper result is wrong; we have not audited those artifacts or reproduced results.

Also, h is a probability-ratio parameter here. A sampled graph's actual fraction of same-class edges fluctuates; finite class sizes and different homophily definitions matter.

### Would storing each undirected edge twice explain the difference?

No. For undirected edges {0,1} and {1,2}, degrees are [1,2,1]. Their mean is 4/3.

Storing (0,1), (1,0), (1,2), and (2,1) gives 4 edge entries. Dividing 4 by 3 gives the same mean degree. Multiplying that stored count by 2 again would be double-counting.

### Options we can choose, with different scientific meanings

| Option | What we do | Benefit | Cost or limitation |
| --- | --- | --- | --- |
| Preserve Matcha code semantics | Keep its probabilities; use a name such as degree_scale and report the derived expected degree | Faithful to the inspected sampling convention | The input number is not the average degree |
| Use a paper-style degree parameter | Include the factor 2 and state the finite-size correction | Closer to the paper's approximate degree interpretation | Produces a different distribution from the inspected code |
| Specify p_in and p_out directly | Treat edge probabilities as the primitive inputs and report implied degree/homophily | Very explicit and easy to audit | Less convenient if experiments are described only by degree and homophily |
| Use original-style CSBM parameters | Use its degree/separation and feature definitions | Clear connection to the original mathematical model | Not automatically a reproduction of Matcha's variant |

**Primitive input** means a quantity defined directly, from which other quantities are calculated.

For example, to set an exact finite-size expected degree k while defining h as the ratio of expected same-class neighbors to expected total neighbors, one could use p_in = k*h/(N/2−1) and p_out = k*(1−h)/(N/2). This is another explicitly defined convention, requires N>2, and requires valid probabilities. It is not automatically Matcha's h definition or our approved choice.

Every probability must lie between 0 and 1. Parameters that violate that condition must be rejected, not silently clipped: clipping changes the requested model.

**Implemented choice (2026-10-25):** under the owner's delegated autonomy, selected the released Matcha-code distribution. CSBMConfig stores p_in/p_out explicitly; from_matcha uses degree_scale. Expected degree is calculated separately. This resolves ambiguity without silently altering the reference; other conventions remain valid, separately named alternatives.

Evidence: [Matcha generator at the inspected revision](https://github.com/baowenxuan/Matcha/blob/f0e7104a84d118a8dabb7ad210915c43722b94af/src/dataset/CSBM.py) and [Matcha paper, section 3.2](https://arxiv.org/abs/2410.06976). The expectations above are our derivations.

## 10. Issue 2: labels can enter before training or adaptation

A **target label** is the true answer for a node in the new evaluation setting. **Ground truth** means that true answer. **Leakage** means information enters a learning or selection step where the protocol says it is unavailable.

### What the inspected Matcha corruption does

For real target graphs, its utility looks at endpoint labels, marks edges as same-class or different-class, then deletes selected edges to reach a desired edge-homophily level.

A **corruption** here is a controlled change applied to data to test robustness. It does not mean a damaged file.

For an illustrative target:

~~~text
Labels:       node 0 → class 0; node 1 → class 0; node 2 → class 1
Edges:        0—1 and 1—2
Same-class:   0—1
Different:    1—2
~~~

Deleting the same-class edge requires knowing which edge it is. True labels were used even if they never appear in the model's forward function.

### Three uses of labels that must stay distinct

| Situation | Is label access part of the stated setting? | How to report it |
| --- | --- | --- |
| Offline synthetic/benchmark construction | It can be, if explicitly defined and frozen before method evaluation | Label-aware data construction |
| Offline generator training on permitted training environments | Yes, according to our plan | Supervised meta-training |
| Adaptation to a held-out target advertised as unlabeled | No | Using true target labels here would violate the setting |

Offline construction is not automatically an invalid benchmark. But claiming that such construction could be performed on a genuinely unlabeled deployment graph would be incorrect.

A **frozen artifact** is saved data that is not repeatedly redesigned after observing which version makes our method look best.

### What does this mean for CSBM?

CSBM necessarily knows hidden classes when generating edges/features. That is legitimate simulation construction. Once generated, target labels must remain outside deployment inputs. The adaptation system cannot call the simulator for the true labels or use a true-label homophily statistic to choose an update.

**Pseudo-labels** are the model's own guesses. Agreement between pseudo-labels is observable without true target labels, but can be wrong. It must not be described as measured ground-truth homophily.

A useful test is to permute target labels while holding permitted inputs fixed and confirm deployment inputs do not change. **Permute** means rearrange. Passing this test is helpful, but does not prove every possible leakage path is absent.

### What is a supervised oracle?

An **oracle** is a reference method given extra information the actual method does not have. A target-label-trained adapter can indicate what is recoverable with supervision. It is not a fair unlabeled adaptation baseline. Nor is it guaranteed to be a literal accuracy upper bound: optimization can fail and labels can be noisy.

Source: [Matcha edge-deletion implementation](https://github.com/baowenxuan/Matcha/blob/f0e7104a84d118a8dabb7ad210915c43722b94af/src/dataset/utils.py). This issue is background for understanding label boundaries; no real-graph corruption is in our current scope.

## 11. Splits: which labels may do what?

Training, validation, and testing are separate jobs.

| Role | What labels are used for | What they must not do |
| --- | --- | --- |
| Source training | Learn initial GCN weights | Include held-out target answers |
| Source validation | Choose settings/checkpoints under the approved protocol | Become an unrecorded training pool |
| Source test | Evaluate source-task performance | Drive repeated model selection |
| Meta-training environments | Train the adaptation generator offline | Include final held-out target realizations/labels |
| Generator validation | Choose generator settings without training gradients on those labels | Be reused as final unbiased test evidence |
| Held-out target test | Score the final adapted model | Train, condition, or select that adaptation |
| Deployment input | Features/edges and permitted model information | Carry true target labels or statistics derived from them |

**Meta-training** here means training the model that learns to adapt other models. It is another learning stage, not permission to use every available label.

**Checkpoint** means a saved model state. **Model selection** means choosing settings or checkpoints based on performance. Label leakage can occur through selection even when gradients never use those labels.

### Approved CSBM development split

For a balanced 100-node source graph:

| Split | Total nodes | Class-0 nodes | Class-1 nodes |
| --- | --- | --- | --- |
| Train | 60 | 30 | 30 |
| Validation | 20 | 10 | 10 |
| Test | 20 | 10 | 10 |

**Stratified** splitting means deliberately preserving class proportions approximately, or exactly when counts allow. For small class counts we must define rounding and reject impossible requests.

The approved initial development ratio is configurable 60/20/20. This is not an official CSBM benchmark split and not the final label budget for every experiment.

### Index list versus mask

For four nodes:

~~~text
train_indices = [0, 2]
train_mask    = [True, False, True, False]
~~~

Both select the same nodes. A **mask** is a list of yes/no values selecting rows. Indices directly list row numbers. We must check bounds, duplicates, overlap, and required coverage.

### Three different 60/20/20-looking ideas

Do not confuse:

1. Source-node label splits.
2. Splits of graph environments.
3. Splits of pretrained models.

The plan's 18/6/6 model bank partition is not a node mask. Unseen models do not guarantee unseen target graphs, and unseen graphs do not guarantee unseen pretrained models.

## 12. Issue 3: a mask does not determine visible graph structure

**Supervision** means learning from true labels. **Observation** means information the model can see. They are different.

Consider:

~~~text
Training node A —— Test node B
~~~

A GCN computes A's representation partly from B's features if that edge and B's features are in the visible graph.

Calculating loss only on A does not remove B's message.

| Protocol | What can be visible? | Important condition |
| --- | --- | --- |
| Transductive | A fixed graph, potentially including unlabeled test nodes/features | Test labels remain withheld; the benchmark must permit this graph context |
| Inductive | New nodes or graphs evaluated outside the training graph context | The exact allowed training/target context must be stated |
| Chronological | Data observed according to a time rule | Future nodes/edges cannot be assumed available if the protocol forbids them |

**Transductive** and **inductive** describe learning settings. They do not by themselves define every adaptation permission. For example, an inductive graph-transfer experiment can still expose the entire unlabeled target graph when adaptation occurs.

### What is an induced subgraph?

Select some nodes, then retain only edges whose two endpoints are selected. That gives the **induced subgraph** on those nodes.

If we keep only A, edge A—B disappears. Degrees change, available features change, and GCN normalization changes. This is not just a different way to store the same loss mask.

### Why official splits are not the whole protocol

A benchmark's node indices specify label membership. Separate documentation or reference pipelines specify graph visibility. Standard OGB, GOOD, and cumulative EERM temporal graphs cannot be interchanged just because all have train/validation/test terminology.

### What applies now?

We approved independent source/target CSBM development graphs. The target graph will not be part of source-graph message passing. Within the source graph, we still need to explicitly specify whether all source features/edges remain visible while only source-training labels supervise learning.

**Selected initial policy:** retain the full source graph with training-label masking for the initial source GCN, and expose the whole separate unlabeled target graph only at target evaluation/adaptation. This initial visibility policy was selected under the owner's implementation autonomy and is recorded in Decision D1. Both the loader smoke test and the implemented source-training pipeline follow it.

### Why the five labeled-source splits need care

If a dataset already supplies official splits, creating five new random splits across all nodes can move official test labels into training.

For CSBM we define our own protocol, but should keep source validation/test labels reserved. Varying the labeled-training subset inside a fixed permitted pool is different from reshuffling the entire pool five times. Before building the weight bank, choose which form of generalization we want to test.

Source context: [OGB node protocols](https://ogb.stanford.edu/docs/nodeprop/) and [inspected EERM cumulative graph construction](https://github.com/qitianwu/GraphOOD-EERM/blob/ffdc4a11161976fac7dd71e2aa1dcd72db6e44e9/temp_arxiv/dataset.py).

## 13. Independent CSBM draws versus paired interventions

This distinction concerns “what exactly stayed unchanged?”

### Independent draws

Generate a source graph and a target graph separately. They can share parameter values for one component while differing in their realized values because of randomness.

Example: use identical edge probabilities but different seeds. The two graphs do not have identical edge lists.

This can test transfer between graph distributions. It does not prove the effect of changing only one realized component.

### Paired intervention

An **intervention** deliberately changes one part of an existing example. **Paired** means the source and modified target have linked identities.

| Intended paired change | Keep exactly fixed | Change |
| --- | --- | --- |
| Attribute-only | Node identities, labels, edge list | Features |
| Structure-only | Node identities, labels, feature values | Edges |
| Combined | State the shared identities/labels explicitly | Features and edges |

A **realization** is the concrete draw, not the recipe. Keeping a recipe fixed is weaker than keeping the realized data fixed.

### Why this matters for conclusions

If we regenerate features while changing graph structure, a performance difference may reflect both structural effects and random feature difficulty. Repeated independent experiments can study average distributional effects, but that is a different design from a matched comparison on one base graph.

**Confounding** means two changing factors are mixed so their effects are hard to separate. Paired experiments can reduce some confounding, but do not automatically solve every leakage or selection issue.

### Paired data also create a leakage risk

If one base graph is reused across generator training and target testing through different corruptions, the target may be more familiar than claimed. Keep related base-graph families together when the intended claim requires new graph identities.

This relationship is **lineage**: a record of which examples were derived from the same parent graph/features.

Our approved independent development graphs are sufficient for initial loader tests. They do not settle the later paired-versus-independent shift experiment. A different seed is also not proof that every meaningful property is new.

## 14. What exactly will the unified loader return?

A **loader** retrieves stored data or generates synthetic data, converts it into an agreed representation, and checks it. **Unified** means later code uses one common contract instead of separate training logic for every dataset.

An **interface** is that contract: what a function takes and returns. **API** is the programmer-facing form of an interface.

The approved direction is:

~~~text
DatasetBundle
├── graph records, addressed by graph ID
│   ├── PyG Data: features, edges, explicit node count
│   ├── labels stored separately
│   ├── named node split indices
│   └── graph metadata
├── dataset/task metadata
└── explicit experiment selections and permitted roles
~~~

A **bundle** is a small container for these related records. **Metadata** is information describing the data, rather than the main feature/edge arrays.

### Proposed fields and their meaning

| Field | Shape/type | Meaning |
| --- | --- | --- |
| graph.x | float32 [N,F] | Node features; 32-bit approximate numerical values |
| graph.edge_index | int64 [2,E] | Stored edge endpoints; 64-bit whole-number IDs |
| graph.num_nodes | Integer | Total nodes, including isolated ones |
| labels | int64 [N] initially | One class index per node, stored outside model input |
| splits | Name → index vector | Which nodes belong to each supervision/evaluation split |
| graph_id | String | Which graph realization this is |
| environment settings | Recorded configuration | Which generating distribution the graph belongs to |
| task metadata | Small descriptive record | Feature count, class schema, metric |
| provenance | Recorded origin/history | Parameters, seeds, versions, transformations, lineage |

A **class schema** describes the known label categories and their meaning. It must come from the task definition or permitted source information, not from peeking at target-label frequencies.

**Provenance** means where something came from and how it was produced.

### Why separate labels from PyG Data?

PyG allows Data.y. That is a convenient library feature, not a scientific requirement. We choose a separate label store to reduce accidental exposure to adaptation code.

The deployment-facing graph should be assembled using an **allowlist**: explicitly include permitted fields. Merely removing a field named y can leave a second label alias or a label-derived statistic behind.

This is an engineering guardrail, not a security boundary against malicious code with full memory access. Good tests and disciplined function arguments are still necessary.

### Sketch of how it will be used

The implemented API is shown below; environments is a sequence of CSBMEnvironment records naming each graph, role, and CSBMConfig. See README.md for a complete construction:

~~~python
bundle = load_dataset(environments)
source_indices = bundle.select(graph_id="source_0", split="train")
source_labels = bundle.training_labels("source_0")
target_graph = bundle.unlabeled_input(graph_id="target_0")
~~~

The adaptation function should receive target_graph and permitted model information, not the full label-bearing bundle. The evaluator receives held-out labels separately.

**Compatibility across architectures:** GCN, GAT, and GraphSAGE can consume the same node-feature/edge representation. Their internal aggregation rules differ; shared input format does not imply identical layers or normalization.

## 15. How is the data fed into a GCN?

### The basic data flow

~~~text
CSBM configuration + RNG
           ↓
Hidden classes → generated features and edges
           ↓
Validation of graph record and split selections
           ↓
Features + edges → GCN → one vector of class scores per node
                              ↓
           permitted labels + selected scores → training loss

Held-out target:
Features + edges → pretrained/adapted GCN → predictions
Evaluation labels stay separate until predictions are scored.
~~~

A **forward pass** is running the model to compute outputs. A **loss** is a number describing prediction error for learning. A **gradient** describes how changing a parameter would change that loss. **Backpropagation** calculates those gradients through the computation.

### One GCN layer in plain English

For each node:

1. Gather information from its neighbors and, usually, itself.
2. Scale those contributions according to graph degrees.
3. Apply a shared learned linear transformation.
4. Optionally apply a nonlinear function.

**Shared** means the same layer weights are used at every node. The layer does not learn a separate weight matrix for each node ID.

A **linear transformation** mixes and rescales feature coordinates using a matrix. **Nonlinear** functions let multiple layers express more than one combined linear rule. **ReLU**, a common choice, replaces negative values with zero.

### The usual formula, with every symbol unpacked

~~~text
H_next = activation( D_tilde^(-1/2) A_tilde D_tilde^(-1/2) H W )
A_tilde = A + I
~~~

| Symbol | Meaning |
| --- | --- |
| A | Adjacency matrix describing edges |
| I | Identity matrix: ones on the diagonal, zeros elsewhere |
| A_tilde | Adjacency including self-connections |
| D_tilde | Diagonal matrix of degrees after those self-connections |
| H | Current node representations; X at the first layer |
| W | Learned matrix transforming feature dimensions |
| activation | A function such as ReLU |
| D_tilde^(-1/2) | Inverse square roots of degrees on the diagonal |

**Diagonal matrix** means only diagonal entries may be nonzero. The inverse-square-root factors scale a message from j to i by 1/sqrt(degree_i × degree_j), using degrees that include the self-loops. This is called **symmetric normalization**. It is not simply “take an arithmetic mean of neighbors.”

This mathematical matrix form does not require us to build a dense adjacency matrix in code. PyG can perform the operation through sparse connectivity.

### Tiny numerical aggregation example

Use nodes 0—1—2 from section 4. After including self-loops, their degrees are 2,3,2. For node 0, before applying W and ignoring bias:

~~~text
aggregated features
= X[0]/2 + X[1]/sqrt(6)
≈ [0.8266, 0.0816]
~~~

Node 3, isolated from other nodes, can still use its own features through its self-loop.

A **bias** is an additional learned offset. We omitted it in the hand calculation to show aggregation clearly.

### Shapes through a two-layer model

For N nodes, F input features, H hidden coordinates, and C classes:

| Step | Shape |
| --- | --- |
| Input features | [N,F] |
| First GCN output | [N,H] |
| Hidden representation after activation | [N,H] |
| Final class scores | [N,C] |
| Predicted class IDs | [N] |

A **hidden representation**, also called an embedding here, is the model's intermediate numerical description of each node. A **logit** is a raw class score before conversion to probabilities. **Softmax** converts a vector of logits into nonnegative numbers that sum to one. **Argmax** selects the position of the largest score.

For single-label classification using PyTorch CrossEntropyLoss, pass raw logits to the loss, not already-softmaxed probabilities. A two-class task can use two logits with cross-entropy or a properly designed one-logit binary formulation; those are different output conventions.

A hypothetical node with logits [2,1] has class-0 probability about 0.731. If the true class is 0, its negative-log-probability loss is about 0.313. These are mathematical illustration values, not experiment results.

### What a training step would do

~~~text
1. Compute predictions from features and edges.
2. Select only permitted training rows.
3. Compare those scores with permitted labels.
4. Calculate gradients.
5. Let the optimizer update weights.
~~~

An **optimizer** is the update rule, such as Adam. An **epoch** is one pass through the chosen training data; in a small full-graph setup it commonly includes one full-graph training update.

**Full-batch** means processing the whole chosen graph context for a step. A **mini-batch** processes a smaller collection at a time. **Neighbor sampling** selects limited neighborhoods for scalable graph computation. Small initial CSBM experiments can start with full-graph processing; larger methods need separate memory/runtime checks.

### Why preprocessing stays out of the common loader

Default normalized GCNConv already handles its self-loop/normalization logic. Reapplying GCN normalization to already-normalized data can change the operator. Adding or replacing self-loops in multiple places can also change graph semantics or edge weights.

Other preprocessing words you may meet: **symmetrization** adds/combines reverse connections to represent an undirected relation; **coalescing** groups repeated edge entries using a defined combination rule; **feature scaling** changes feature magnitudes. Each can change model behavior, so the loader must state what it does. Raw node features are not automatically required to sum to one.

GAT learns attention-based neighbor weights. GraphSAGE combines own-node and neighborhood information through its chosen aggregation rule. They should not automatically inherit GCN-normalized edges from the common loader.

**Caching** means retaining a computed result for reuse. GCNConv(cached=True) can reuse normalization from an earlier graph. Across source/target environments, that can be wrong. Prefer cached=False unless graph changes and cache clearing are deliberately handled.

Sources: [original GCN paper](https://arxiv.org/abs/1609.02907), [official PyG GCNConv](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.GCNConv.html), and [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). Real forward/backward checks now pass for individual GCN, GAT, and GraphSAGE layers through this interface. They establish compatibility, not trained-model performance.

## 16. Random seeds and reproducibility

A **seed** initializes a pseudorandom-number generator, or **RNG**. A pseudorandom generator produces a repeatable sequence from its initial state; it is not a source of independently unpredictable physical randomness.

| Random choice | What it should control | What should stay fixed when varying it |
| --- | --- | --- |
| Edge-generation stream | Which permitted edges appear | Feature draw if studying only connectivity |
| Feature-generation stream | Numerical feature draws | Edges if studying only features |
| Split seed | Which source labels are available in each split | The graph and features |
| Shift seed | The random realization of a defined perturbation | The base example and unrelated choices |
| Model seed | Initialization and intended training randomness | Dataset and supervision split |

One global seed can make the entire script repeat, but mixes causes: adding random draws in generation can change later splits or model initialization. Separate streams make controlled comparisons easier.

**Deterministic** means giving the same result under specified conditions. **Reproducible** means we can recreate a claimed procedure/result with enough recorded information and stated tolerances.

A seed alone is insufficient. Save parameters, split indices, graph identity, versions, preprocessing, model configuration, and checkpoints where appropriate. Different software versions, hardware, or parallel operations can still change results.

**Cache identity** means the information used to decide whether saved data can be reused. It must include the generator version, relevant parameters, seeds, and transformations. “Same node count” is not enough.

## 17. How will we know the loader is correct?

No single successful load proves correctness. We need several kinds of evidence.

| Check | What could go wrong without it? | Example |
| --- | --- | --- |
| Shapes and data types | Arrays accepted by one function but misinterpreted by another | edge_index accidentally shaped [E,2] |
| Node/label alignment | A node receives another node's answer | Reordering X without reordering labels |
| Edge bounds | An edge refers to a nonexistent node | Node ID N in a graph numbered 0 through N−1 |
| Explicit node count | Isolated nodes disappear | Section 4's node 3 |
| Edge convention | Direction or degree silently changes | Confusing unique undirected edges with stored entries |
| Split integrity | Training and evaluation labels overlap | A train index also in test |
| Seed isolation | A split comparison changes the graph too | Split seed also affects feature generation |
| Statistical checks | The generator implements a different distribution | Observed degrees systematically inconsistent with the stated formula |
| Target-input boundary | Hidden labels reach adaptation | Label histogram included as metadata |
| Small realistic execution | Components disagree despite unit checks | A generated graph cannot enter the agreed data interface |

A **unit test** checks a small component. A **smoke test** checks that a small realistic path runs. A **statistical check** compares random samples with their predicted behavior.

Statistical tests need repeated samples and sensible tolerances. **Tolerance** is the allowed numerical difference. One random graph should not fail merely because its degree is not exactly equal to the expectation.

**Edge cases** are unusual valid or invalid inputs: zero edges, tiny classes, invalid probabilities, nonfinite feature values, or impossible split sizes. **Nonfinite** means not an ordinary finite number, for example NaN or infinity.

For empty connectivity, keep edge_index shaped [2,0], not an ambiguous empty one-dimensional array.

The data milestone checks actual layer output shapes, finite loss, and gradients. The source-GCN milestone now also checks learning, checkpoint reloads, best-epoch restoration, and end-to-end training/evaluation. Changing source-test and target labels leaves the seeded training history and selected weights unchanged. See Section 24.

### What the first working milestone taught us

The obstacle was ambiguous parameter meaning, not a reason to stop building. We resolved it by storing actual probabilities, naming the released-code conversion accurately, reporting expected degree, and testing sampled behavior. We did not silently insert a factor of two.

The 5,000-node example observed source/target mean degrees 2.4676/2.5768 with degree_scale=5, around expectations near 2.5. These are single-run graph diagnostics, not accuracies. Isolated nodes were retained; they are expected in sparse random graphs.

The layer check computes predictions and source gradients but does not optimize a trained model. Target labels stay outside the model inputs; offline inspection is a separate operation.

We also caught a practical issue in review: an unanchored data/ Git ignore pattern can hide src/amortized_gnn_tta/data/. Using /data/ ignores only the runtime directory at the repository root. Reviewing untracked source matters.

## 18. How will the later research avoid unfair conclusions?

### Same information, fair comparison

A **baseline** is a simpler or established comparison method. If our generator saw labeled shifted environments during training but a baseline did not, an improvement may come from extra supervision rather than generated updates.

Keep a comparison trained with the same permitted diverse environments but without target-specific generated adaptation. Also compare generated and optimized updates using the same permitted parameterization where feasible.

An **ablation** removes or changes one component to study its contribution. Example: compare conditioning on target features alone with features plus the pretrained model's embeddings.

### What small weight updates mean

**Parameter-efficient** means adapting relatively few numbers compared with the whole model.

| Approach | Intuition | Important distinction |
| --- | --- | --- |
| Head-only adaptation | Adjust the final classifier | Cannot change all earlier message representations |
| Low-rank update | Express a large weight change using two smaller matrices | A restricted family of updates, not every possible full update |
| Representation modification | Change intermediate embeddings | Not automatically a change to model weights |

For a mathematical weight matrix W of size m-by-n, a rank-at-most-r update can be written:

~~~text
W_new = W + U V
U has shape [m,r]
V has shape [r,n]
~~~

The new factors contain r(m+n) numbers rather than mn. **Rank** describes the number of independent directions a matrix can express. This saves generated parameters only when r(m+n) is suitably smaller than mn.

Matrix orientation in an actual library can differ from textbook notation; inspect real parameter shapes before applying an update.

A **hypernetwork** is a neural network that generates parameters for another network. **Conditioning** means making its output depend on specified information. **Diffusion** and **flow matching** are families of generative methods that learn transformations from noise to desired samples. They are later options, not prerequisites for our loader.

### What would count as a new target?

| Generalization claim | What must be new? |
| --- | --- |
| New realization | The actual target draw |
| New severity | The amount of a shift |
| New family | The kind of change |
| New pretrained model | The underlying model instance |
| New base graph under paired shifts | The parent graph/features, not only its corruption seed |

**Generalization** means succeeding beyond the specific examples used for learning. These are different claims and need separate tests. Seeds and a random model split do not automatically establish all of them.

**Latency** is elapsed time for an operation. Measure adaptation latency separately from final prediction cost, and include conditioning/model-summary computation. Zero target-time gradient steps do not mean zero computation.

## 19. Important decisions: what is settled and what is not?

This table summarizes the important project records in plain English. It does not replace the detailed decision log.

| Topic | Status | Reason or remaining question |
| --- | --- | --- |
| Dataset now | CSBM only | Explicit owner instruction |
| Other datasets | Background only | Not current implementation work; the roadmap's later natural-shift goal is not erased |
| Task | Node classification | Predict labels for nodes |
| Your architecture | GCN | Teammates handle GAT/GraphSAGE |
| Common graph/label interface | Implemented and tested | Separate labels, named splits, allowlisted cloned inputs |
| Initial source development split | Configurable stratified 60/20/20 approved | A project convention, not an official benchmark rule |
| Independent development source/target graphs | Approved | Initial loader exercise; not a final paired-shift protocol |
| CSBM probability and feature formulas | Released Matcha-code distribution selected | Explicit p_in/p_out, fixed-direction Gaussian features |
| Matcha compatibility | Sampling distribution only | Different RNGs/splits, not exact artifacts or paper-result reproduction |
| Source graph visibility | Initial policy selected | Full source graph, training-label loss; separate target graph |
| Five source-label subsets | Implemented inside fixed training pool | Explicit budget/seeds; validation/test unchanged; Section 25 |
| Phase 1 budget/model grouping | Fixed in phase configuration | 50% of native training pool; whole label subsets grouped 3/1/1; Section 27 |
| Final-paper protocol | Still open | Development graph/scores do not establish a sealed final benchmark |
| Dependencies/environment | CPU stack installed/tested in .venv | Reuses existing torch/NumPy; clean install and other backends unverified |
| Source training and scoring | Implemented and CPU-tested | Select minimum source-validation loss; score held-out labels afterward |
| Sequential weight bank | Implemented and CPU-tested | Thirty subset/seed jobs retained; 18/6/6 assignment and diversity audit executed |
| Adaptation/generators | Not implemented | Separate milestones and protocols |
| Commits/pushes | Foundation committed as 30af4f9; Phase 1 completion commit explicitly authorized | Later commits need approval; push not authorized |

**Next:** discuss exact parameter-efficient adaptation methods and shift/environment holdouts before Phase 2. Phase 1's budget and model grouping are fixed; later paper-level protocols remain open. Dataset choice and the approved interface do not need reopening.

Project records: [DECISIONS.md](../DECISIONS.md), [PROJECT_STATE.md](../PROJECT_STATE.md), and [AGENTS.md](../../AGENTS.md).

## 20. Committee and interview questions, with concise answers

Use these as explanations you understand. Separate executed data/training checks from planned adaptation experiments.

| Question | Answer you should be able to explain |
| --- | --- |
| What is your research question? | Whether an offline-trained model can predict small useful GNN weight updates for unlabeled shifted target graphs, reducing deployment optimization. |
| What are you implementing now? | A CSBM data pipeline plus a small source-GCN training/evaluation baseline; adaptation is later. |
| Why CSBM? | It lets us control graph connections and features, test assumptions, and generate repeated environments. It does not alone prove performance on natural shifts. |
| Is CSBM one fixed dataset? | No. It is a family of generation models; the exact equations and protocol must be stated. |
| Why investigate Matcha? | It is in our research materials and studies the shifts of interest with inspectable CSBM code. It provides evidence and comparisons, not authority over our design. |
| Are you reproducing Matcha? | We match the released generator's distribution through an independent implementation, not its seeded artifacts or full model/training/adaptation protocol. |
| What exactly enters the GCN? | Features and graph connectivity. This first runner supports unweighted graphs; optional edge fields in the common contract require explicit future support. Labels stay separate. |
| Can a model predict an isolated node? | It can use that node's features through own-node processing; it has no other-node neighborhood information. |
| Are edges or weights being adapted? | Our eventual research focus is GNN parameter updates. Dataset shifts can change edges, but graph editing and weight adaptation are different operations. |
| Why not place all labels in Data.y? | PyG allows it, but separating labels reduces accidental access by adaptation code. |
| Is observing test features always leakage? | No. Some transductive protocols permit it. Access must match the specified graph context; withheld labels remain protected. |
| Does a training mask remove test neighbors? | No. It usually selects the loss rows. Removing nodes requires a graph operation and changes message passing. |
| Why are there two edge counts? | An undirected connection is often stored as two directed entries for message passing. |
| Why is source splitting not enough? | We must separately hold out target environments and pretrained model instances. |
| Does the degree discrepancy prove the paper is invalid? | No. It establishes a mismatch in the inspected sampling convention, with a derived expected degree. Published artifact/results reproduction is a separate investigation. |
| Can offline construction use labels? | Yes if explicitly part of the benchmark/simulator definition; the deployment algorithm must still obey its label-free setting. |
| Does unlabeled adaptation mean no labels ever? | No. Source training and permitted offline meta-training use labels; held-out target adaptation does not. |
| Why not use true target homophily? | It requires true labels. A pseudo-label statistic is available but has different meaning and can be wrong. |
| What would invalidate the claim of an attribute-only paired change? | Changing the edge realization too, while claiming it was fixed. |
| Does a new random seed guarantee a new environment? | No. It can give a new realization from the same environment. |
| Why use an MLP comparison? | It checks how much can be predicted from features without graph aggregation. |
| What accuracy have you achieved? | Two single CPU source-only runs gave source-test/target accuracy 71.0%/53.4% at 1,000×64 and 91.0%/57.1% at 5,000×2,000. Sizes and epoch caps differ; these verify execution, not adaptation or a controlled comparison. See Section 24. |
| What is the novelty? | A proposed systematic study of amortized GNN weight adaptation; a novelty claim needs a broader literature review and experiments. |
| What will you report? | Accuracy or the native task metric, uncertainty across repeated runs, adaptation/prediction cost, memory, protocol, and limitations. |
| What does the implementation verify today? | CPU tests cover graph/split/seed correctness, sampling statistics, label boundaries, layer compatibility, source training/selection, and checkpoints. Other backends and future adaptation remain unverified. |

## 21. Copy-ready questions for another ChatGPT conversation

Each prompt is self-contained. Paste one at a time. Ask the other assistant to challenge assumptions, distinguish definitions from implementation details, and cite primary sources. A confident answer is not proof; compare its derivation and sources with ours.

### Prompt A — teach the whole project from zero

~~~text
I am a college student working on “Amortized Weight Adaptation for Graph Neural Networks under Distribution Shift.” I own GCN experiments. GCN/CSBM Phase 1 is complete with 104 CPU tests: loader, source training/evaluation, a thirty-model bank, artifact validation, a whole-label-subset 18/6/6 model partition, and weight/prediction/representation diagnostics. No adaptation/generator results exist. The graph and paired seeds are shared; this is development engineering evidence, not a final-paper benchmark.

Teach me what a graph, node, edge, feature, label, GCN, source environment, target environment, and distribution shift mean. Explain how an offline-trained adaptation generator differs from a CSBM data generator. Then explain how one generated graph reaches a GCN and how labels are used separately for loss/evaluation. Use a tiny numerical graph and define unfamiliar terms. Ask me one reasoning question at a time. Do not invent experimental results or imply adaptation has been implemented.
~~~

### Prompt B — audit the Matcha degree discrepancy independently

~~~text
Please independently audit this CSBM sampling convention. There are N even nodes, two classes of size N/2, no self-loops, independent undirected Bernoulli edges, p_in=d*h/N and p_out=d*(1-h)/N.

1. Derive the finite-size expected degree from neighbor counts.
2. Evaluate it for N=1000, d=10, h=0.8.
3. Explain whether storing both directed entries of each undirected edge changes that result.
4. Compare it with the approximate paper definition d=N*(p_in+p_out)/2.
5. Distinguish code-compatible sampling, paper-style degree semantics, and an exact finite-size expected-degree parameterization.
6. Discuss the meaning of h and why sampled homophily need not equal it.

Reference code: https://github.com/baowenxuan/Matcha/blob/f0e7104a84d118a8dabb7ad210915c43722b94af/src/dataset/CSBM.py
Paper: https://arxiv.org/abs/2410.06976

Do not assume either our derivation or the paper/code is correct without checking. Separate verified facts, mathematical deductions, and unverified claims about reported results. Explain in beginner-friendly English.
~~~

### Prompt C — choose a CSBM definition without silently changing methodology

~~~text
We are designing a GCN distribution-shift project and have selected CSBM only for now. We selected the released-code convention for the initial milestone and want an independent critique of that choice. One reference, Matcha, uses p_in=d*h/N and p_out=d*(1-h)/N and a fixed feature direction. The original CSBM paper uses c_in=d+lambda*sqrt(d), c_out=d-lambda*sqrt(d), edge probabilities c_in/N and c_out/N, and a latent Gaussian feature direction with its own scaling.

Compare preserving Matcha sampling, using paper-style expected-degree semantics, specifying probabilities directly, and using original-style CSBM. Explain the trade-offs for reproducibility, interpretability, feature signal/noise, and comparison to published results. Define every symbol and explain probability-validity constraints. Matching one generator must not be presented as reproducing an entire paper. Recommend a minimal specification, clearly labeled as a recommendation, with primary sources.
~~~

### Prompt D — explain label-aware construction versus deployment leakage

~~~text
Our research method is supposed to adapt a GCN to an unlabeled held-out target graph. A reference benchmark constructs a target by using true node labels to identify and delete same-class edges. CSBM likewise uses hidden classes when generating graph edges and features.

Explain when this label use is legitimate offline data construction and when it violates an unlabeled deployment claim. Distinguish benchmark construction, supervised meta-training, validation/model selection, unlabeled adaptation, and supervised target-label oracles. Use concrete examples. Propose tests and API boundaries that reduce accidental leakage, and explain why removing only a field called y is insufficient. Do not say all label-aware synthetic construction is automatically invalid.
~~~

### Prompt E — masks, graph visibility, and induced subgraphs

~~~text
I am learning GCN node classification. A training node A is connected to a test node B. The model processes the full graph, but the loss uses only A's label.

Can B's features influence A's prediction and parameter gradients? Explain with a small GCN aggregation example. Distinguish a label mask from an induced subgraph, transductive from inductive protocols, and future-information restrictions in chronological evaluation. Explain why official split indices alone do not specify every graph-visibility rule. We plan separate source/target CSBM graphs, so discuss source internal splits separately from target-graph access. Use simple English and do not equate all test-feature observation with label leakage.
~~~

### Prompt F — independent samples versus paired shift experiments

~~~text
Our CSBM project plans attribute-only, structure-only, and combined shifts. For initial loader development we approved independent source/target graphs, but the later scientific shift protocol is undecided.

Compare independent graph draws with paired interventions on a shared base graph. State exactly what must be identical for an attribute-only paired intervention and a structure-only paired intervention. Explain “same distribution” versus “same realization,” confounding, variance reduction, and possible leakage from reusing base graphs across meta-training and testing. Propose graph/feature/shift seed roles and lineage records. Do not imply independent draws automatically satisfy a claim that the realized edge list was fixed.
~~~

### Prompt G — audit the proposed PyG data interface

~~~text
We implemented a minimal node-classification dataset interface: a mapping from graph IDs to CPU PyG Data objects containing x:float32[N,F], edge_index:int64[2,E], and explicit num_nodes; labels:int64[N] are stored separately; named split indices, task metadata, graph-generation settings, and seeds are retained. An unlabeled_input method builds an allowlisted target graph without labels or label-derived diagnostics.

Critique this design for GCN now and GAT/GraphSAGE later. Explain how source training and target evaluation would access it, and how one-graph versus multiple-graph datasets fit. Discuss isolated nodes, empty edges, duplicate edges, directionality, class schema, graph mutation, and accidental label aliases. Keep the proposed implementation small; do not add speculative frameworks. Explain what this API can and cannot enforce.
~~~

### Prompt H — teach standard GCN and compare Matcha's analysis

~~~text
Teach standard Kipf–Welling GCN aggregation from zero, using H_next=activation(D_tilde^(-1/2)*(A+I)*D_tilde^(-1/2)*H*W). Define all symbols, explain self-loops and symmetric degree normalization, and trace shapes through two layers for N nodes, F features, H hidden coordinates, C classes.

Compare this with the simplified expression X+gamma*D^(-1)*A*X used in Matcha's analysis. Explain why related aggregation ideas do not make the operators identical or make its theoretical guarantees automatically apply to low-rank updates of PyG GCNConv. Discuss cached=True when graphs change. Cite the original GCN paper and official PyG documentation.
~~~

### Prompt I — split design and the future weight bank

~~~text
Our GCN project uses generated CSBM graphs. We approved configurable stratified 60/20/20 source-node splits for development. The roadmap later suggests 5 labeled-source splits × 6 model initialization seeds = 30 pretrained models, divided into 18/6/6 models for adaptation-generator training/validation/testing.

Explain which decisions remain unresolved. Separate node splits, graph-environment splits, and model-instance splits. Compare varying labeled subsets within a fixed training pool with reshuffling all source labels. Explain how model-bank grouping changes claims about new initializations versus new supervised tasks. Define a leakage-free protocol without silently promoting source validation/test labels into generator training. Treat all final protocol choices as recommendations pending discussion.
~~~

### Prompt J — randomness and statistical validation

~~~text
We independently implemented the released Matcha-code CSBM distribution, with explicit probabilities, degree_scale conversion, and tagged RNG streams. One hundred four CPU tests cover the pipeline through audited/grouped bank preparation. The full-budget thirty-model source bank completed, with no adaptation. Explain how to separate edge, feature, split, shift, subset-selection, partition, and model RNG streams. Distinguish reproducibility within a tested software environment from bitwise reproducibility across hardware/versions. Explain why reusing six model seeds across five subsets is a paired design, not thirty independent repetitions.

Design meaningful tests for a generator/loader: shapes, label alignment, empty edges, isolated nodes, probability validity, exact split membership, deterministic seed behavior, target-label isolation, and repeated-sample degree/feature statistics. Explain how to choose statistical tolerances without requiring one random graph to equal its expectation. Distinguish stated executed checks from additional proposed tests; do not invent results.
~~~

### Prompt K — explain the other datasets without expanding our scope

~~~text
Our implementation scope is CSBM only, but I need to understand dataset names in our research proposal: Twitch-E through EERM, PyG Twitch, GOOD-Twitch, small Cora, GOOD-Cora, standard ogbn-arxiv, GOOD-Arxiv, EERM/Matcha temporal Arxiv, Syn-Cora, and Syn-Products.

For each, explain what nodes/edges/features/labels represent and what protocol distinguishes it. Explain why 3,170 binary Twitch-E features are not interchangeable with PyG's 128-dimensional Twitch features, why GOOD-Cora is not simply small Planetoid Cora, and why official OGB chronological label splits differ from cumulative temporal graph protocols. Verify with original papers or official source; separate unknown artifact details from confirmed definitions. Do not recommend implementing additional datasets now.
~~~

### Prompt L — prepare a fair research defense

~~~text
I am preparing to explain an early-stage AFML project: an offline-trained generator is proposed to produce small GCN parameter updates for unlabeled shifted target graphs. The CSBM loader and a source-only GCN baseline are implemented and CPU-tested. Two single-run baseline experiments verify execution; no adaptation-performance results exist.

Explain what claims we can honestly make now, which baselines control for extra supervised meta-training, how to compare generated versus iterative updates using equal parameterization, why target-label oracles are separate, and how to measure total deployment cost. Distinguish head-only adaptation, low-rank weight updates, representation changes, hypernetworks, diffusion, and flow matching. Ask challenging committee questions, give understandable answers, and identify claims that still require a literature review or experiments.
~~~

## 22. Quick glossary

Terms are also explained at first use. This table is for quick revision.

| Term | Plain-English meaning |
| --- | --- |
| Adaptation | Changing a model or its behavior for a new setting |
| Adjacency matrix | A table recording which nodes are connected |
| Aggregation | Combining information from a group, such as neighbors |
| Allowlist | Explicit list of fields or actions permitted |
| API / interface | The agreed way code receives and returns information |
| Architecture | The model's arrangement of layers and operations |
| Attribute / feature | A measured or generated input number |
| Augmentation | A deliberately produced data variation |
| Baseline | A comparison method used to judge a proposed method |
| Batch | The collection of data processed together |
| Bernoulli | A random yes/no outcome with a specified probability |
| Bias | A learned additive offset; elsewhere the word can also mean a systematic distortion |
| Cache | Saved data/computation reused later |
| Checkpoint | A saved model state |
| Class / label | A possible answer category / the true category for an example |
| Conditioning | Making a generator's output depend on supplied information |
| Configuration | Explicit settings for a run or component |
| Confounding | Mixing changes so their separate effects are hard to identify |
| Covariate | An input variable; covariate shift concerns changes to input distributions under specified assumptions |
| Covariance | How numerical directions vary together |
| Data contract | The shapes, fields, and meanings other code can rely on |
| Deployment | Running the learned system on new inputs |
| Deterministic | Same output under specified repeated conditions |
| Device | CPU or accelerator on which computation runs |
| Distribution | A description of which observations are likely |
| Embedding | A numerical representation of an object |
| Entropy | A measure of spread/uncertainty in a probability vector; uncertain predictions often have higher entropy |
| Environment | A specified setting of the data-generating distribution |
| Epoch | One pass through the chosen training data |
| Evaluation | Scoring predictions under fixed rules |
| Expected value | Average value over repeated draws from a probability model |
| Finite-size correction | A term that matters because the number of nodes is finite |
| Forward pass | Computing model outputs from inputs |
| Generalization | Performance beyond the examples used for learning |
| Gradient | Sensitivity of a quantity to changes in a parameter |
| Ground truth | The true answer used for supervision or evaluation |
| Held-out | Reserved from particular learning/selection operations |
| Homophily / heterophily | Preference for same-class / different-class connections in our usage |
| Hyperparameter | A setting chosen outside ordinary parameter learning, such as hidden size |
| Hypernetwork | A neural network producing parameters for another network |
| IID | Independent and identically distributed; examples neither depend on one another nor follow different marginal distributions under that assumption |
| Inductive | Evaluating new examples beyond the training graph context, with visibility rules specified |
| Intervention | A deliberate change to part of a system/example |
| Latent | Hidden rather than directly supplied as an observed input |
| Latency | Elapsed time to perform an operation |
| Leakage | Use of information forbidden at a particular experimental stage |
| Lineage | Record of shared origins/derivations between data examples |
| Logit | Raw model score before a probability conversion |
| Loss | A numerical learning penalty |
| Low-rank | Expressible through a limited number of independent matrix directions |
| Manifest | An explicit record listing artifacts, selections, or their identities |
| Mask | Boolean row-selection vector |
| Meta-training | Training the system that learns how to adapt models |
| Metric | Rule used to score predictions |
| MLP | Multilayer perceptron; a neural network without graph aggregation in our feature-only comparison |
| Model selection | Choosing model settings/checkpoints using permitted evidence |
| Normalization | Rescaling under a specified rule; feature and adjacency normalization are different operations |
| Objective | Numerical quantity an optimization procedure seeks to improve |
| OOD | Out of distribution, relative to the stated training distribution |
| Optimizer | Rule that updates parameters to improve an objective |
| Oracle | A comparison with access to extra information |
| Parameter | A number in a model or generator; specify whether learned or configured |
| PEFT | Parameter-efficient fine-tuning/adaptation: changing a relatively small parameter set |
| Permutation | A rearrangement of entries |
| Preprocessing | Transformations applied before the main model operation |
| Protocol | Rules defining allowed information, learning, selection, and evaluation |
| Provenance | Origin and production history |
| Pseudo-label | A model-generated guess used as a label-like quantity |
| Realization | One concrete sample from a random recipe |
| ReLU | Function replacing negative inputs with zero |
| Reproducibility | Ability to recreate the stated procedure/results under documented conditions |
| RNG / seed | Pseudorandom-number generator / its initialization input |
| Schema | Definition of fields/categories and their meanings |
| Severity | Amount or strength of a shift |
| Softmax | Conversion from multiple scores to numbers summing to one |
| Sparse | Stored/computed mainly through existing entries, such as edges |
| Stratified | Split while retaining relevant group proportions |
| Supervision | Learning information supplied by true labels |
| Tensor | Numerical array of one or more dimensions |
| Tolerance | Allowed difference in a numerical check |
| Transductive | Learning/evaluating on a fixed graph context with some labels withheld |
| TTA | Test-time adaptation: changing the system using permitted test/deployment inputs |
| Unlabeled | True labels unavailable to the specified learner at that stage |
| Validation | Data-based selection/checking separate from final testing |
| Variance | Spread of a random quantity around its mean |
| Weight bank | Saved collection of pretrained model parameters |

Graph observations are not automatically IID: connected nodes influence one another's model representations and their data can be statistically dependent. An edge-list interface does not remove that dependence.

## 23. Sources, evidence, and how we will keep learning

Our local roadmap is edit.pdf; the one-pager and resource index identify research directions and references. Their proposals are not experimental evidence.

Primary references used in this handbook:

| Source | Why it matters |
| --- | --- |
| [Original CSBM paper](https://papers.nips.cc/paper/2018/file/08fc80de8121419136e443a70489c123-Paper.pdf) | Model definition and original parameterization |
| [Matcha paper](https://arxiv.org/abs/2410.06976) | Structure-shift motivation, simplified analysis, dataset protocol |
| [Matcha official repository](https://github.com/baowenxuan/Matcha) | Inspectable generator and label-aware construction |
| [GPRGNN official repository](https://github.com/jianhao2016/GPRGNN) | CSBM implementation lineage and another architecture |
| [Original GCN paper](https://arxiv.org/abs/1609.02907) | Standard graph-convolution definition |
| [Original GCN implementation](https://github.com/tkipf/gcn) | Historical TensorFlow reference; not our runtime dependency |
| [PyG Data documentation](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.data.Data.html) | Graph storage and node-count semantics |
| [PyG GCNConv documentation](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.GCNConv.html) | Layer normalization, self-loops, shapes, caching |
| [PyTorch cross-entropy documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | Correct class-score/label contract |
| [EERM official repository](https://github.com/qitianwu/GraphOOD-EERM) | Twitch and temporal dataset protocols |
| [GOOD official repository](https://github.com/divelab/GOOD) | Named graph OOD splits and metrics |
| [OGB node benchmark documentation](https://ogb.stanford.edu/docs/nodeprop/) | Standard arxiv task, split, and evaluator definitions |

Pinned revisions and license findings are in [DECISIONS.md](../DECISIONS.md). A public repository is not automatically licensed for copying. Dataset licenses and code licenses can differ. We have not copied reference implementations or installed their environments.

### Important limits to our present evidence

- The degree discrepancy is derived from inspected code and paper definitions; it is not a reproduced experimental finding.
- Hand-worked numerical examples are illustrations. Section 24 explicitly identifies the executed source-only results; they are not adaptation evidence.
- The graph-label interface is executable and CPU-tested. That is pipeline evidence, not proof that the proposed adaptation method improves accuracy.
- Observed repository state is not evidence that a scientific method works.
- A second assistant's answer is another explanation to audit, not a replacement for sources, derivations, and tests.

### A short learning routine

1. Read one section.
2. Explain it aloud in your own words.
3. Draw or calculate its tiny example.
4. Try the related copy-ready question if you want a second explanation.
5. Compare the reasoning, including disagreements.
6. Record only important corrections and real lessons from implementation.

**Start with these four questions:**

1. What are X, edge_index, and labels, and which of them enters a GCN's prediction function?
2. Why does Matcha's inspected d produce roughly d/2 expected neighbors?
3. Can test-node features influence training without violating a permitted transductive protocol?
4. What exactly must stay fixed in an attribute-only paired intervention?

Understanding check: not yet discussed. Reading this handbook is progress; we will check reasoning through examples rather than assume that reading means mastery.

## 24. Our first working source GCN: from graph to saved predictions

### What have we built, and why this step?

We can now generate CSBM graphs, train one GCN on source training labels, select its best saved weights using source validation labels, and score those fixed weights on source-test and target nodes. This is a **source-only baseline**: a model learned from the source, then used directly on the target without any target-specific update.

We need this baseline before adaptation. Otherwise, we would have nothing reliable to compare an adaptation method against. A successful training run also tests whether data, model, supervision, metrics, and saved files agree with each other. Section 26 builds a model bank on this trainer with a smaller explicit label budget.

This is our documented small baseline, not a reproduction of Matcha's architecture or the original GCN paper's full experiment protocol. No adaptation generator has been trained.

### What exactly is fed into the model?

The small example creates a source graph and a separate target graph, each with 1,000 nodes and 64 features per node. The GCN receives only `x` and `edge_index`. It does not receive node IDs as features, true labels, split membership, or generation parameters.

| Step | Shape in the small example | Plain meaning |
| --- | --- | --- |
| Source feature matrix `x` | [1000,64] | One row of 64 measurements per node |
| Source `edge_index` | [2,E] | Endpoints of stored directed edge entries |
| First GCN layer | [1000,32] | Combine neighboring features into 32 learned numbers per node |
| ReLU | [1000,32] | Replace negative numbers with zero |
| Hidden dropout, during training | [1000,32] | Randomly zero some hidden values and rescale retained values |
| Second GCN layer | [1000,2] | Produce two class scores per node |
| Select source training rows | [600,2] | Only these rows contribute supervised training loss |
| Compare with training labels | [600] | One correct integer class, 0 or 1, per selected node |

`E` is the number of stored edge entries, not a fixed constant. Our undirected edges have both directions stored. Each GCN layer adds its own self-loops and degree normalization; the loader does not duplicate that work.

The two-layer calculation is:

~~~text
hidden = ReLU(GCNConv_1(x, edge_index))
hidden = dropout(hidden, probability=0.5, only_during_training=True)
logits = GCNConv_2(hidden, edge_index)
~~~

The default hidden width is 32. **Width** means how many learned coordinates each hidden node representation has. There is no separate linear classification head: the second graph-convolution layer produces the final class scores. Later “head-only adaptation” must specify exactly which parameters it changes.

We set `cached=False` in both layers. A **cache** stores a previous calculation for reuse. Reusing source connectivity normalization on a different target graph would be wrong, so we recompute it. This is one reason not to copy every option from a single-graph benchmark example.

### What is a logit, and why is there no final softmax in the model?

A **logit** is a raw class score. Larger means the model prefers that class; it is not itself a probability. For example, scores `[2,1]` prefer class 0. Softmax converts these to probabilities of approximately `[0.731,0.269]`.

**Cross-entropy**, abbreviated CE, penalizes assigning low probability to the correct class. For one node it is `−log(probability assigned to its true class)`. With the example above, a true class-0 label gives about 0.313 loss; a true class-1 label gives about 1.313. These are hand-worked illustrations, not our experimental losses.

PyTorch's cross-entropy function expects raw logits and performs the probability-related calculation internally in a numerically stable way. We therefore do not apply softmax before calling it. Accuracy simply chooses the largest score and counts correct answers.

### What happens in one training epoch?

An **epoch** is one pass through the training data. This is full-graph training: one forward pass handles the source graph, followed by one optimizer step per epoch. We do not create separate minibatches of nodes.

1. Put the model in training mode, which enables dropout.
2. Clear gradients left from the previous step. A **gradient** tells us how changing each weight would change the loss.
3. Compute scores for all source nodes, using the full source graph.
4. Compute cross-entropy using only source training rows and their labels.
5. Backpropagate: calculate gradients through the model operations.
6. Use Adam to update the weights. **Adam** is an optimizer that adjusts updates using running summaries of gradients. The learning rate, 0.01 by default, controls update size. **Weight decay**, 0.0005 here, adds a penalty toward smaller weights; it applies to all trainable parameters in this baseline.
7. Switch to evaluation mode, disable gradient recording, and measure source validation loss/accuracy with dropout off.
8. Save a copy of the weights if source validation loss is strictly lower than before. Stop after the configured number of consecutive epochs without improvement, or at the epoch cap.

The training loss in our history is measured before the update with dropout on. Validation loss is measured after the update with dropout off, on different labeled nodes. They are useful curves but are not computed under identical conditions.

**Does every source node affect training?** Its features and edges can affect message passing under our declared full-source-graph protocol. Only training labels enter the training loss. Validation labels select the checkpoint; source-test labels do neither. “Not used for gradients” and “not used for selection” are different protections.

### Why choose validation loss rather than the last epoch or target accuracy?

Training loss can keep falling while performance on other examples worsens. This is **overfitting**: learning details that do not transfer well. Source validation labels provide permitted feedback for choosing when to stop and which weights to retain.

**Patience** is the number of consecutive epochs without a strictly lower validation loss that we allow. A tie keeps the earlier checkpoint and counts as no improvement. For illustration:

| Epoch | Validation loss | Action with patience=2 |
| --- | --- | --- |
| 1 | 0.70 | Save weights |
| 2 | 0.60 | Save new best weights |
| 3 | 0.63 | First non-improving epoch |
| 4 | 0.65 | Second; stop and restore epoch 2 |

This table is an illustration. An automated test deliberately makes validation worsen and verifies that our code restores the earlier best weights. Saving `state_dict()` alone would leave references to live tensors; later training could change the supposed saved best values. Our snapshot clones the tensors, so it is an actual copy.

Minimum validation loss and maximum validation accuracy need not select the same epoch. Accuracy ignores confidence when the predicted class is unchanged; CE also measures how much probability the model assigns to the correct class. We chose minimum source-validation CE as a simple explicit rule before interpreting the target scores. We do not claim it is the only possible rule.

Choosing an epoch, seed, architecture, or hyperparameter because it gives the highest held-out target accuracy would use target labels for model selection. That would invalidate a claim that those targets were untouched final tests.

### How does the software keep the boundaries clear?

| Component | Receives | Important boundary |
| --- | --- | --- |
| `source_training_data` | A bundle and the source graph name | Extracts source graph, train labels/indices, validation labels/indices; no source-test/target labels |
| `fit_source` | That source view, a model factory, training config | Cannot access a target record through its arguments; selects source validation loss |
| `predict` | A model and label-free graph input | Returns logits; accepts no labels, uses evaluation mode and no gradients |
| `score_logits` | Saved predictions, selected indices, selected true labels | Explicit offline scoring; does not update the model |

A **factory** here is simply a function that constructs a fresh model. We set the model seed before calling it, so initialization is repeatable in the tested CPU environment. The shared trainer does not need to know how a GCN, GAT, or GraphSAGE is constructed; compatible models must accept `(x, edge_index)` and return raw `[N,C]` scores. The initial runner supports unweighted graphs and rejects unsupported input fields rather than silently ignoring them.

We tested a direct leakage question: if we change source-test and target labels, does training change? In the executed test, the seeded history, selected epoch, and weights stayed identical. This is useful evidence for this pipeline, not a proof against every possible future form of leakage.

`model.eval()` changes behaviors such as dropout; it does not disable gradient recording by itself. We also use an inference/no-gradient context. Evaluation leaves weights unchanged, and our inference helper restores the caller's previous training-mode flag.

### What did we actually run?

Both CPU runs used hidden width 32, dropout 0.5, Adam learning rate 0.01/weight decay 0.0005, and model seed 31. Source generation uses Matcha-style degree_scale=5, h=0.8, means −0.03/+0.03. Target generation uses degree_scale=5, h=0.2, means −0.01/+0.05. Noise scale is 1; features retain the loader's scaling by `1/sqrt(F)`. Source edge/feature/split seeds are 11/12/13; target seeds are 21/22/23.

These are independent graph draws with changed connection tendencies and feature means. They are not a paired intervention that preserves an original graph. Model seeding controls training randomness separately from the loader's tagged random streams.

| Executed run | Nodes × features per graph | Epoch cap / patience | Epochs run / chosen | Source-test accuracy | Target accuracy |
| --- | --- | --- | --- | --- | --- |
| Small review run | 1,000 × 64 | 100 / 20 | 46 / 26 | 71.0% | 53.4% |
| Paper-sized execution check | 5,000 × 2,000 | 20 / 10 | 20 / 20 | 91.0% | 57.1% |

**What can I say to the committee?** “Our CSBM source-GCN pipeline trains, selects using source validation only, saves/restores its checkpoint, and evaluates the frozen model. In two single seeded CPU checks, target performance was lower than source-test performance. We have not implemented adaptation or established a repeated-run research result.”

**What can I not conclude?** The larger run does not establish that increasing graph size caused better accuracy: feature dimension, random realizations, and training caps also differ. Twenty epochs do not establish convergence. One model seed gives no estimate of variation across seeds. “Paper-sized” refers to dimensions; it does not mean paper-accuracy reproduction.

We have now seen these development targets' scores. If those observations influence later design/tuning, use fresh held-out target realizations for final research claims. Keeping labels out of gradient code is necessary, but human tuning from repeatedly viewed test scores can also leak evaluation information.

### What is saved, and how do I run it myself?

From the repository root, after setup:

~~~bash
.venv/bin/python examples/train_source.py --epochs 100 --patience 20 --device cpu --output-dir runs/my-source-gcn
~~~

Choose a new output directory each time; the script refuses to overwrite an existing run. For Windows-style environment activation instructions if needed, use the equivalent environment Python command; the Python code uses portable paths.

| File | Meaning |
| --- | --- |
| `best_model.pt` | Best weights as CPU tensors, model/training configuration, selected epoch, source provenance/indices, history, software versions |
| `metrics.json` | Human-readable run record with configurations, seeds, splits, data identity, history, source/target scores, device, and reload verification |

A **checkpoint** is saved model state. A **state dictionary** maps parameter names to tensor values. We save this dictionary and plain metadata instead of a pickled whole model object; loading constructs the architecture explicitly and maps tensors to CPU first. The example checks that predictions agree after reloading. See the [README reload example](../../README.md#train-and-evaluate-a-source-gcn).

This is an inference checkpoint, not a complete training-resumption checkpoint. It does not store Adam's internal state or all random-generator states needed to resume the interrupted optimization trajectory. Checkpoint files and run outputs are ignored by Git. No new dependency was needed for this milestone.

Only CPU execution has been verified. `auto` chooses available CUDA, otherwise CPU. MPS is explicit opt-in; availability alone does not prove that all required graph operations work. An unsupported operation should fail visibly. We do not promise identical results across hardware or library versions.

### Quick revision and second opinion

| Question | Answer to practice in your own words |
| --- | --- |
| Why are there predictions for 1,000 nodes but only 600 training labels? | Message passing uses the whole source graph; the supervised loss selects only the training rows. |
| Why not softmax before cross-entropy? | The PyTorch function expects raw scores and handles that calculation internally. |
| Why can training and validation loss differ even in one epoch? | Different labeled rows, dropout behavior, and pre-update versus post-update weights. |
| Is a low target score evidence that adaptation will work? | No. It motivates investigation; a proposed update could also fail or harm performance. |
| Why save the best epoch rather than just stop training? | The stopping epoch can come after the best validation epoch. |
| Have we learned this material just because these notes exist? | No. Explain the flow and solve a small example without looking first. |

### Prompt M — audit and teach our implemented source baseline

~~~text
I am learning an AFML project on amortized GNN weight adaptation. Currently only CSBM generation and a source-only GCN baseline exist; adaptation is NOT implemented. Teach and audit this baseline in beginner-friendly English, defining technical words and using small worked examples.

Data: balanced binary CSBM; p_in=d*h/N, p_out=d*(1-h)/N, degree_scale d=5 (expected degree approximately 2.5); features (class_mean+Gaussian_noise)/sqrt(F), noise scale 1. Source h=.8, means -.03/+.03, edge/feature/split seeds 11/12/13; independent target h=.2, means -.01/+.05, seeds 21/22/23. Source stratified 60/20/20 splits. Full source graph visible for message passing; only train labels in supervised loss. Target graph is separate and unused during fitting.

Model: uncached PyG GCNConv(F,32), ReLU, hidden dropout .5, GCNConv(32,2), raw logits. Adam lr .01, weight decay .0005 on all trainable parameters; model seed31 before construction. Each epoch trains with source-train cross-entropy, then measures source validation CE in eval/no-gradient mode. Strictly lower validation CE saves cloned weights; ties keep earlier weights. Patience counts consecutive non-improving epochs; restore best before final scoring. No source-test or target labels in trainer inputs. Label-free prediction and explicit-label scoring are separate.

Executed CPU development runs: N1000/F64, cap100/patience20, stopped46/best26, source-test71.0%, target53.4%; N5000/F2000, cap20/patience10, best20, source-test91.0%, target57.1%. Both reloaded CPU checkpoints successfully. Forty-four tests passed, including unchanged seeded training after held-out labels change. These are two single seeded runs, not a controlled size comparison, a Matcha reproduction, or adaptation results.

Explain the tensor flow, why raw logits enter CE, gradient versus selection boundaries, early stopping versus best-checkpoint restoration, inference mode versus eval mode, cache risks across graphs, and inference checkpoint versus exact training resumption. Challenge limitations in the protocol and what the results establish. Explain why fresh final targets are needed if viewed scores inform future tuning. Cite official PyTorch/PyG documentation or original papers when verifying claims. Do not invent results or claim novelty. End with five reasoning questions I should be able to answer aloud.
~~~

## 25. Five source-label subsets: preparing the weight bank carefully

### Why not just train the same model thirty times?

The roadmap proposes two kinds of variation: which source labels a model learns from, and its random initialization/training seed. Five label selections × six model seeds give thirty models. A **weight bank** is the saved collection of those trained models. This section explains the label-preparation step; Section 26 describes the subsequently implemented runner and executed banks.

A **training pool** is the set of node labels already permitted for source learning. A **subset** contains some members of that pool. Our subsets change the supervised training examples; they do not generate new graphs or change the validation/test split.

### One concrete example: where do all 1,000 nodes go?

The existing source graph has 1,000 nodes. Its original split stays 600 training, 200 validation, 200 test. The executed development example uses half of each training class for every subset:

| Group | Original membership | What one subset uses |
| --- | --- | --- |
| Native source training pool | 600 nodes: 300 per class | 300 labels: 150 per class |
| Other nodes in that training pool | Included in those 600 | Remaining 300 labels unused for that model's supervised loss |
| Source validation | The same 200 nodes | Used to select its checkpoint |
| Source test | The same 200 nodes | Reserved for later offline scoring |
| Source graph features/edges | All 1,000 nodes | Full source context still visible for message passing |

The second row is part of the first row, not an extra group of 300 nodes. Nothing was moved into testing. The label budget is **50% of the training pool**, which is **30% of the entire graph**: 0.5 × 0.6 = 0.3.

The fraction is required in our API/command, not a hidden default. We first demonstrated 0.5, then fixed it for the Phase 1 reference configuration. It is not an official CSBM budget or a finalized paper protocol. On a 5,000-node source, it selects 1,500 of its 3,000 training labels.

### What does stratified sampling mean here?

**Stratified** means sampling separately within groups, here the known source-training classes. For each class, we select `floor(fraction × class training-pool size)` nodes. **Floor** means round down: floor(15.9)=15. If that would leave a class with zero selected labels, the code rejects the request rather than silently changing the budget.

Sampling is **without replacement** within a subset: one node cannot appear twice in that subset. Different subsets may overlap. For example, node 17 can provide a label to both model groups A and B.

Why allow overlap? The five selections represent different labeled samples from the same permitted pool, not five mutually exclusive folds. Each needs a useful amount of supervision. In the balanced 1,000-node example, two independent 300-label subsets have expected overlap 150: within each class, 150 × 150 / 300 = 75 shared labels, summed over two classes. That is a mathematical expectation, not a measured overlap report.

We reject identical selections. Different seeds alone do not guarantee different selected IDs, especially in tiny pools. Taking every training label gives only one possible selection, so requesting five distinct full-pool subsets is impossible.

### Which seed does what?

| Seed | Example | Controls |
| --- | --- | --- |
| Source edge/feature seeds | 11 / 12 | Source graph connections and feature values |
| Native source split seed | 13 | The original 600/200/200 membership |
| Subset seeds | 101–105 | Five selections inside those same 600 training nodes |
| Model seed | 31 in earlier baseline | Model initialization and intended training randomness |

Subset sampling uses local NumPy streams and sorted node IDs. It does not advance the model's RNG or regenerate the graph/split. Reordering a list of the same pool nodes does not change our selected sets. Class labels used for sampling come only from the permitted source training pool.

The tests changed source validation/test and target labels and obtained exactly the same subset manifest. Validation labels will still be used later for checkpoint selection; they do not choose these training subsets.

### What is a manifest, and why save node IDs instead of just seeds?

A **manifest** is an explicit record of what belongs to an experiment. Here it stores every subset's node IDs and seed, the requested fraction and actual class counts, source graph identity, training-pool identity, fixed validation IDs, provenance, and software versions.

A **fingerprint/hash** is a compact value computed from content to help detect a mismatch. A changed graph, training label pool, or validation membership cannot silently reuse our old manifest. This is an error-detection measure, not a security guarantee against deliberate tampering.

Saving exact IDs also avoids having to redraw subsets when reloading. Random-library behavior can differ across versions; the stored IDs remain the authoritative membership. We retain seeds and versions to explain how they were produced.

One implementation lesson: Python distinguishes tuples `(a,b)` from lists `[a,b]`, but JSON has only arrays for both. Our first persistence test found that metadata changed representation on reload. We fixed the creation boundary to use JSON-compatible values from the start. We kept the exact-equality test, rather than weakening it to ignore the discrepancy.

### How do these subsets reach the existing trainer?

~~~text
native source bundle
    → source_training_data(bundle, "source")
    → create/load the five-subset manifest
    → source_subset_view(source, manifest, "source_subset_0")
    → fit_source(model_factory, subset_view, training_config)
    → checkpoint records selected IDs and subset/manifest identity
~~~

The source graph is shared read-only by the views. The trainer makes its own execution copy, so we do not store five duplicate graphs. Each view supplies only its selected training labels and the unchanged source validation supervision. No target graph is required to construct it.

Run the preparation example from the repository root:

~~~bash
.venv/bin/python examples/prepare_source_subsets.py --label-fraction 0.5 --output runs/my-source-subsets.json
~~~

Use a new output filename; existing manifests are not overwritten. Both the 1,000-node and 5,000-node examples completed with five distinct subsets and exact JSON reload checks. The full CPU suite passed 70 tests, including actual training/checkpoint integration from a subset. This verifies the preparation machinery, not model diversity or adaptation quality.

### What remains before calling the weight bank complete?

1. Train six fixed model seeds per subset and record all thirty model identities/configurations. This is now implemented and development-tested; see Section 26. A converged final scientific bank still needs an agreed protocol.
2. Decide how the generator's 18/6/6 model partition should work. Phase 1 now groups whole label subsets as 3/1/1 (Section 27). Randomly separating instances could put the same selection into multiple roles. The chosen grouping tests a new selected subset on the same graph; it does not prove new-graph transfer.
3. Measure predictions and hidden representations, as well as weights. This audit is now implemented (Section 27). Hidden units can be reordered without changing model behavior, so raw weight distance alone is insufficient; useful diversity for adaptation still requires experiments.

The earlier baseline used all 600 source training labels; the development subsets use 300. If new bank models score differently, the supervision budget is one possible cause. A fair adaptation comparison must keep the underlying pretrained model and permitted information consistent.

References: [PyG dataset splitting](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/dataset_splitting.html) explains class-aware splitting; we preserve existing holdouts rather than applying a new full-node random split. [NumPy random-stream guidance](https://numpy.org/doc/stable/reference/random/parallel.html) explains explicitly separated streams.

### Prompt N — audit the source-label subset protocol

~~~text
I am building a CSBM GCN weight bank. A fixed 1,000-node source graph has a native stratified 600/200/200 train/validation/test split. We prepare five distinct training-label subsets. In a development example, each independently samples 50% of each class inside the existing training pool, without replacement within a subset. That gives 300 labels per subset, 150 per class. Different subsets may overlap. Unselected training labels stay unused; they do not become test labels. The full source graph remains visible for message passing.

The fraction is required/configurable, with floor rounding per class and rejection of empty classes. Local subset seeds 101–105 are separate from graph, native-split, and model seeds. Exact node memberships, source fingerprints, budgets, provenance and versions are saved in JSON and validated on reload. No source-test or target labels are needed for construction. Phase 1 fixes 50% for its full-budget bank, groups whole subsets as 3/1/1 for 18/6/6 models, and measures diversity. Final-paper protocols remain future work.

Teach and critique this protocol in simple English. Explain the difference between a label subset, graph split, cross-validation fold, graph realization, and model seed. Derive expected overlap of two subsets. Discuss fairness when comparing these models with an earlier baseline trained on all 600 labels. Explain thirty-model construction from five subsets × six seeds, and compare random model-instance 18/6/6 partitions with holding out whole label subsets as 3/1/1. Explain why neither automatically demonstrates new-graph generalization or useful model diversity. Use original papers/official documentation for factual claims and do not invent experiment results. Finish with five questions to check my understanding.
~~~

## 26. The working thirty-model runner: what does a bank actually contain?

### Thirty jobs, built from two factors

A **factor** is a setting we deliberately vary. Here the factors are the label subset and model seed. Their **Cartesian product** means every combination: five subsets, each trained with all six seeds, gives thirty jobs.

| Factor | Current development settings | What stays fixed |
| --- | --- | --- |
| Labeled subset | source_subset_0 through source_subset_4 | Graph, source validation/test membership, requested label budget |
| Model seed | 31 through 36 for every subset | Architecture, optimizer settings, epoch cap/patience |

Each job constructs a fresh GCN and Adam optimizer. It does not continue training the previous model. Each also gets its own minimum-source-validation-loss checkpoint. We retain every planned subset/seed pair rather than discarding models with lower validation accuracy.

**Why reuse the same six seed values across subsets?** This pairs starting points: for example, seed 31 starts with the same GCN initialization when the architecture is unchanged. That makes supervision comparisons easier to interpret. It also means we must not describe the bank as thirty independent initializations. These are thirty trained instances in a paired design. Distinct source subsets can still produce very similar behavior.

The tests run the same jobs in a different seed order and compare their histories and weights. They matched exactly on the tested CPU stack. Per-job reseeding before construction prevents previous jobs, including checkpoint reloads, from changing the next initialization.

### From one saved subset to one saved model

~~~text
saved subset manifest
    → reconstruct and validate the fixed CSBM source
    → select source subset A
    → construct fresh GCN with model seed 31
    → fit using A's labels; select using fixed source validation
    → save selected weights/configuration/history
    → reload on CPU and compare source predictions
    → record artifact hashes and mark this job complete
    → repeat for seed 32, and so on
~~~

The runner accepts only a source view, not the label-bearing full dataset bundle. Source-test/target labels are absent from that interface. Its example creates no target graph. Selected-training and source-validation scores are execution evidence; validation has already influenced checkpoint selection, so it is not an untouched final test.

**Two meanings of validation:** source validation nodes select an epoch while pretraining each GCN. Generator validation models, the later six-model partition, will assess a different learner that predicts adaptations. They are different levels of the experiment; the word “validation” alone does not tell you which one is meant.

### Files, in plain English

~~~text
runs/gcn-bank/
    bank.json
    source_subsets.json
    models/
        source_subset_0_seed_31/
            best_model.pt
            metrics.json
        ... 29 other model directories ...
~~~

| File or field | Meaning |
| --- | --- |
| bank.json | Index of planned model IDs, configurations, progress, file locations/hashes, and per-model summaries |
| source_subsets.json | Exact label membership that produced this bank |
| best_model.pt | Named weight tensors and checkpoint metadata, not a JSON list of weights |
| metrics.json | Training history, selected epoch, selected-training/validation scores, fit time, reload check |
| SHA-256 artifact hash | Content identity used to check that a referenced file has not changed |
| Git revision plus dirty flag | Which commit the checkout started from, and whether uncommitted changes were present |

Model IDs are unique within a bank. To identify a model across experiments, also retain the bank/source identity and its architecture/training configuration. “seed 31” alone is not a complete experimental identity.

We do not flatten all weights into one giant array yet. Named tensors preserve layer meanings and let a future adaptation method specify which parameters it changes. The existing trainer/checkpoint interface stays reusable for GAT or GraphSAGE factories with the same input/output contract.

### What if the fourth model fails?

Completed jobs keep their files. The progress record marks the failed job/bank as failed when possible and leaves unstarted jobs pending. A deliberately injected failure test verified this behavior. A process killed abruptly may leave status running; that must not be interpreted as completion.

The runner refuses existing output directories, so another attempt cannot silently overwrite earlier evidence. It currently has no automatic resume/skip logic. Choose a new directory and retain the partial run for diagnosis. Also remember that inference checkpoints do not contain the optimizer/random states needed to resume an interrupted epoch trajectory exactly.

Sequential training keeps memory use modest: we keep the source graph and current job's models rather than thirty active models on the device. The runner saves each result before moving on.

### What actually completed on our CPU?

| Development bank | Nodes × features | Labels per model | Cap/patience | Completed jobs | Reloads |
| --- | --- | --- | --- | --- | --- |
| Small | 1,000 × 64 | 300 | 100/20 | 30/30 | All passed |
| Paper-sized execution check | 5,000 × 2,000 | 1,500 | 20/10 | 30/30 | All passed |

Small models selected epochs 15–25 and ran 35–45 epochs. Every paper-sized model selected epoch 20, its cap. That tells us the larger run executed, not that further training was unnecessary. Both banks' checkpoint/metric hashes were verified after execution. The complete automated suite passed 80 tests.

The sums of model-fitting times were approximately 6.41 and 45.71 seconds. They exclude data reconstruction, checkpoint saving/reloading and final scoring, so these are not total script runtimes. The shallow GCN, sparse graphs and sequential jobs keep these development checks manageable on CPU. Other backends remain unverified.

The foundation was committed as 30af4f9. Bank runs correctly recorded dirty=true because their new runner was still uncommitted. We cannot claim that checking out the foundation commit alone recreates this later bank implementation. Final research runs should use an agreed frozen code/configuration state.

Run it using an existing subset file:

~~~bash
.venv/bin/python examples/train_weight_bank.py --subsets runs/source-subsets-small.json --epochs 100 --patience 20 --device cpu --output-dir runs/my-gcn-bank
~~~

Those were the runner milestone's limits. Section 27 records the subsequently completed 200/30 bank, phase reference budget, model partition and diversity audit. A directory with thirty successful checkpoints alone still proves only artifact construction; useful adaptation supervision and research generalization require further evidence.

Reference: [PyTorch saving/loading guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) explains state dictionaries, inference mode, and the extra states needed for training resumption. Our orchestration and experimental policies are documented project choices, not an official benchmark protocol.

### Prompt O — critique our implemented model-bank runner

~~~text
I am learning a CSBM GCN research project. We have implemented a sequential source-only bank runner: five saved class-stratified label subsets × six model seeds (31–36 reused across every subset), thirty jobs. Source graph and validation/test membership stay fixed. Each job constructs a fresh two-layer GCN/Adam optimizer, trains on that subset only, selects minimum source-validation CE, retains the selected checkpoint, reloads it on CPU, and compares predictions. All thirty pairs are retained, regardless of validation score. Source-test and target labels are absent from the runner input.

Each job saves a CPU state_dict checkpoint and JSON history/metrics. bank.json records planned IDs, progress/status, configs, software/device, artifact hashes and Git revision/dirty status. Existing output directories are rejected. Errors preserve completed jobs and mark failure when possible; automatic resumption is not implemented. Phase 1 closure passed 104 CPU tests. A full-budget N5000/F2000 bank with 1500 labels/model, cap200/patience30 completed; selected epochs24–98, all CPU reloads passed. Model partitions and diversity diagnostics are implemented; adaptation is not. Earlier short smoke banks are preserved separately.

Explain from zero: Cartesian products, paired seeds versus independent repetitions, source-node validation versus generator-model validation, artifact identity, and partial versus complete banks. Critique what these checks establish. Explain why epoch-cap selection does not establish convergence, how to partition models honestly, why raw weight distance alone can mislead, and what must be measured before training a useful adaptation generator. Use primary references where verifying facts, define unfamiliar terms, and do not invent experimental results. Ask five reasoning questions afterward.
~~~

## 27. Phase 1 is complete: how to explain and defend the bank audit

### What exactly does “Phase 1 complete” mean?

It means our **GCN/CSBM engineering foundation** works through pretraining-bank preparation: data, source training, saved models, validated loading, model holdouts, diversity checks, and source-only evaluation. It covers the first three roadmap steps for your architecture/current dataset. It does not mean that adaptation works, that teammates' architectures are trained, or that the final research study is complete.

A **reference protocol** is the exact procedure/settings that define this phase's example. Our [phase configuration](../../configs/phase1_csbm_gcn.json) fixes the existing sampling convention, 5,000 nodes, 2,000 features, source split 60/20/20, half of the training pool per subset, five subsets, six paired model seeds, and source-validation stopping with a maximum of 200 epochs and patience 30. We did not invent an official CSBM benchmark split.

### Two different splits: nodes versus models

| Split | What is divided? | Current purpose |
| --- | --- | --- |
| Source train/validation/test | Labels for nodes inside one graph | Train each GCN, select its epoch, then score held-out source predictions |
| Model meta-train/validation/test | Saved trained GCN instances | Reserve models for later adaptation-generator learning/selection/testing |

We group the five label selections as three for meta-training, one for validation, one for testing. Each selection has six trained models, so the counts are 18/6/6. A fixed partition seed, 401, produces:

| Model role | Whole source-label subsets | Models |
| --- | --- | ---: |
| Meta-training | 1, 0, 3 | 18 |
| Validation | 4 | 6 |
| Test | 2 | 6 |

No performance score chooses this assignment. The file is saved before final source-test scoring. Keeping a whole labeled selection in one role prevents that exact selection from being shared across model roles.

**What does it not prevent?** Individual training nodes can belong to several label selections, because the subsets overlap. The graph is the same, and seed values 31–36 are reused in each role. The claim is “new label selections/trained instances on this source graph.” It does not establish new graphs, disjoint source labels, or unseen initialization seeds. These stronger claims require different experiments.

### What does the bank loader audit?

An **audit** is a systematic consistency check. We check more than whether a file opens:

- All intended subset/seed pairs exist exactly once and have complete status.
- Files stay inside their bank directory, exist, and match stored hashes.
- Checkpoint tensors are finite, have the model's expected names/shapes/types, and restore correctly.
- Training configuration, source/subset identity, selected training IDs, and validation IDs agree.
- History is finite and the saved epoch is the earliest minimum source-validation loss.
- Restored source-validation loss/accuracy reproduce the stored metrics.
- Model partitions cover every model exactly once and keep each labeled subset together.

A hash detects a content mismatch; it is not an authentication guarantee against someone deliberately rewriting both data and hashes. Tests deliberately updated corrupted files' hashes and still caught inconsistent seeds, supervision, shapes, NaNs, and selected epochs. This demonstrates why checks must agree across artifacts, not rely on a checksum alone.

### Why compare weights, predictions, and representations separately?

A hidden unit is a learned coordinate. Suppose a model's hidden coordinates are `[a,b]`. We could reorder them to `[b,a]` and reorder the next layer's matching weights. Predictions can stay identical even though the flattened parameter vector changes.

An executed test performs this hidden-unit permutation on a GCN: raw weights differ, predictions agree, and representation CKA stays approximately 1. Therefore raw weight distance is not a sufficient measure of functional diversity.

**Functional** here means the behavior of the model—its outputs for supplied inputs. **Representation** means the hidden numerical description of a node, before the final prediction layer.

| Diagnostic | Plain-English question | Main limit |
| --- | --- | --- |
| Weight L2 / cosine | How far apart/aligned are the parameter vectors? | Hidden-coordinate symmetries can make distance misleading |
| Prediction disagreement | On what fraction of probe nodes do the predicted classes differ? | Ignores confidence changes when the class stays the same |
| Probability RMSE | How different are the complete class-probability vectors? | Differences are not automatically improvements |
| Hidden linear CKA | Do the hidden representations have similar centered geometry? | Depends on the chosen nodes and cannot certify useful adaptation diversity |

**L2 norm** is square root of the sum of squared values. The symmetric relative weight distance divides the distance by the average norm of the two vectors. **Cosine similarity** measures their directional alignment.

**RMSE** means root mean squared error/difference: square coordinate differences, average them, then take the square root. Here it compares one model's probabilities with another's; neither is treated as ground truth. It is not classification error or calibration error.

### CKA without hiding the mathematics

**CKA** means *Centered Kernel Alignment*. A **probe** is the fixed set of nodes used to compare model behavior. A **kernel** compares pairs of representations; the linear kernel uses their dot product, which multiplies matching coordinates and sums the products. Start with X and Y, the two models' hidden matrices on the same probe nodes. Subtract each column's mean to obtain Xc and Yc. Then:

~~~text
linear_CKA(X,Y) = || Xc.T @ Yc ||_F²
                  / ( || Xc.T @ Xc ||_F * || Yc.T @ Yc ||_F )
~~~

`.T` means transpose; `@` means matrix multiplication; the **Frobenius norm**, written `||M||_F`, is square root of the sum of all squared matrix entries. An **orthogonal transformation** rotates, reflects, or reorders coordinates while preserving lengths and angles. The score is unchanged by those transformations or by multiplying all values by a nonzero constant. It is not invariant to every arbitrary transformation.

For us, the probe has 3,000 native source-training nodes and each hidden vector has 32 coordinates. The products above are 32×32. We do not create a 3,000×3,000 matrix in the normal diagnostic path. This is feature-space calculation, based on the original paper and official reference notebook. We use the ordinary **biased** finite-sample version, rather than the notebook's optional debiasing correction; “biased” here concerns statistical estimation, not fairness or source sampling.

If centering leaves a representation entirely zero, the denominator is zero and CKA is undefined. We report `None`/JSON `null` and flag it, rather than replacing it with a fake similarity score. The current bank had no such pairs. A small independent test checks the formula using centered Gram matrices; another checks permutation/scaling invariance.

### What actually changed when we used the full training budget?

The earlier 5,000-node bank stopped at its 20-epoch cap. We preserved it, then trained a new bank with a 200-epoch cap and patience 30. All thirty models selected epochs 24–98 and stopped after 54–128 epochs. None selected the cap. This establishes validation-based stopping in this run; it does not prove that optimization converged to an ideal solution.

The sum of fitting times was about 198.73 seconds, excluding reconstruction, checkpoint I/O/reloading and final scoring. The original tiny runs were fast because they performed much less work; this longer bank is the proper full-budget phase check.

### What did the diversity checks and source evaluation show?

There are 30×29/2 = 435 unordered pairs. The source probe uses features/connectivity and native training-node IDs. It does not use their labels. A test flips training/validation labels passed alongside that graph and confirms the diversity report is unchanged.

| Executed observation | Interpretation |
| --- | --- |
| 30 distinct hard prediction patterns; no exact parameter duplicates | The saved models are not collapsed to identical probe outputs/weights |
| Mean disagreement 7.32%, range 0.43–9.83% | Models have measurable prediction variation on this probe |
| Same-subset mean disagreement 1.19%; same-seed/different-subset 8.60% | In this bank, label selection varied behavior more than changing only the seed |
| Mean probability RMSE 0.1412; mean hidden CKA 0.8614 | Confidence and hidden representations also vary; neither is an improvement metric |
| Source-test mean accuracy 88.37%, range 86.4–89.5% | Descriptive source performance of the frozen bank, with no adaptation |

The population standard deviation across model accuracies is 0.71 **percentage points**. A percentage point is an absolute difference between percentages: 88% to 89% is one percentage point. This is spread across paired models, not an uncertainty interval for future graphs. The report also provides per-subset and per-seed summaries rather than treating 30 jobs as independent datasets.

**Are these good numbers?** They are a useful starting point: 88.37% is well above the 50% chance baseline for our balanced two-class source task, and the bank has measurable variation. They do not establish superiority over matched baselines or successful distribution-shift adaptation. The 7.32% disagreement is measured between model pairs on source-training probe nodes, not against true test labels. Its complement, 92.68%, is agreement between two models, not classification accuracy: two models can agree and both be wrong. Higher disagreement alone is not better; inaccurate models can disagree too. Most observed variation here comes from changing the label selection rather than changing only the model seed, which controls initialization and intended training randomness.

Important qualification: this source graph/split was used in earlier development examples, where source-test scores were already viewed. The final audit freezes settings and model partition before its own test scoring, but it does not turn the historically viewed data into a sealed paper benchmark. Future research claims need an agreed final holdout/tuning protocol.

### What can I claim now, and what should I avoid claiming?

> “We completed the GCN/CSBM data-to-bank engineering phase. Thirty source-trained models are validated, grouped 18/6/6 by their label selections, and audited for weight, prediction, and representation differences. We observed measurable diversity and recorded source performance. Adaptation effectiveness and broader generalization remain untested.”

Do not claim that the bank guarantees a useful generator, that high/low CKA alone proves good/bad diversity, or that the results reproduce Matcha. We did not add arbitrary weight noise to manufacture bigger distances. The next test is whether the proposed adaptation mechanism learns useful, generalizable updates under an agreed protocol.

The final review is [documented in project state](../PROJECT_STATE.md#phase-1-closure), with local artifacts under runs/phase1-complete. It includes JSON, CSV, a readable Markdown report, the frozen configuration and partition, and hashed analysis source files. The snapshot identifies the audit implementation; it is not retroactively called the earlier training code. Training's Git dirty flag remains explicit. The owner authorized the Phase 1 completion commit; later commits and any push need separate approval.

Primary reference: [Kornblith et al., Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html). Its official Google Research demonstration notebook was inspected separately; our PyTorch metric was independently implemented. CKA is a useful comparison tool with limits, not a novelty claim for this project.

### Prompt P — audit and defend our completed Phase 1

~~~text
I am a beginner preparing to defend the GCN part of an AFML research project on amortized weight adaptation under graph distribution shift. GCN/CSBM engineering Phase 1 is complete; adaptation and the generator are NOT implemented.

Protocol: one source graph with 5,000 nodes and 2,000 features per node from the released Matcha-code CSBM distribution, fixed source train/validation/test 60/20/20, five distinct class-stratified subsets using 50% of the existing training pool (1,500 labels per model), and six model seeds 31–36 reused across subsets. Thirty two-layer GCNs use 32 hidden channels, dropout 0.5, Adam learning rate 0.01/weight decay 0.0005, source-validation CE selection, maximum 200 epochs/patience 30. All selected epochs 24–98 and stopped after 54–128. Model partitions hold out whole source-label selections as 3/1/1 for 18/6/6 models, seed 401, with no score-based assignment. Individual node labels overlap across selections, and the graph/seed values are shared across roles.

Artifact validation checks completeness, hashes, model/tensor/source/config identities, supervision membership, histories/selection, and restored source-validation scores. A label-free probe uses all 3,000 native training nodes. Metrics: raw/relative weight L2 and cosine, class disagreement, probability RMSE, and biased centered feature-space linear CKA on first-layer post-ReLU/pre-dropout representations. There are 435 pairs, 30 unique hard prediction patterns, no exact parameter duplicates or undefined CKA. Mean disagreement 7.32%, same-subset 1.19%, same-seed/different-subset 8.60%, probability RMSE 0.1412, hidden CKA 0.8614. Source-test accuracy mean 88.37%, population SD 0.71 percentage points, range 86.4–89.5%, on the same 1,000 nodes. These are descriptive paired-model results, not an IID uncertainty estimate. This development graph/split has earlier viewed source-test scores and is not a sealed final benchmark. 104 CPU tests passed; other backends are unverified. Training used an uncommitted working tree, while the later audit saves its own separate source snapshot.

Explain the entire phase from zero with clear tables and worked examples. Critique the holdout claim, show why a hidden-unit permutation can change raw weights while preserving predictions, derive centered linear CKA and its degenerate case, distinguish stopping from proven convergence, and explain source-node versus model-level validation. Separate artifact correctness, observed diversity, adaptation usefulness, and research novelty. Cite primary sources, do not invent results, and ask me one reasoning question at a time. Also identify what Phase 2 must decide before implementing head-only or low-rank adaptation.
~~~

**Learning check:** explain why keeping an entire label selection in one model partition still does not mean that individual source-training node labels are disjoint across partitions. Then explain why 88.37% source accuracy says nothing yet about adaptation accuracy.
