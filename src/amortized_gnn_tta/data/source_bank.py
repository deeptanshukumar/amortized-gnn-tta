"""Formal source-only supervision groups; no target labels enter this API."""

from dataclasses import replace
import hashlib
import json

import torch

from ..training import validate_source_training_data


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(f'{value.dtype}:{tuple(value.shape)}'.encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def formal_groups(view, native_ids, *, split_seeds, initialization_seeds):
    """Reshuffle fit/stop membership, keeping each class's exact original budget.

    All five groups use the same source-owned union. Natural-source diagnostic
    nodes remain excluded. Distinct group splits are overlapping supervision,
    not five independent graphs. Only source labels are present in ``view``.
    """
    validate_source_training_data(view)
    if (len(split_seeds) != 5 or len(set(split_seeds)) != 5
            or len(initialization_seeds) != 5
            or any(len(s) != 6 for s in initialization_seeds)):
        raise ValueError('Formal bank requires five split seeds and six initializations per group')
    all_seeds = [*split_seeds, *(s for group in initialization_seeds for s in group)]
    if any(type(s) is not int or not 0 <= s < 2**32 for s in all_seeds):
        raise ValueError('Seeds must be integers in [0,2**32)')
    if len(set(all_seeds[5:])) != 30:
        raise ValueError('Initialization seeds must be globally distinct')
    if (native_ids.dtype != torch.long or native_ids.shape != (view.graph.num_nodes,)
            or native_ids.unique().numel() != len(native_ids)):
        raise ValueError('Native IDs must uniquely map all source context nodes')
    indices = torch.cat((view.train_indices, view.val_indices))
    labels = torch.cat((view.train_labels, view.val_labels))
    groups = []
    for group_id, seed in enumerate(split_seeds):
        rng = torch.Generator().manual_seed(seed)
        train_positions, stop_positions = [], []
        for category in labels.unique().tolist():
            positions = torch.where(labels == category)[0]
            positions = positions[torch.randperm(len(positions), generator=rng)]
            count = int((view.train_labels == category).sum())
            train_positions.append(positions[:count])
            stop_positions.append(positions[count:])
        def ordered(parts):
            positions = torch.cat(parts)
            return positions[indices[positions].argsort()]
        train, stop = ordered(train_positions), ordered(stop_positions)
        role = 'meta_train' if group_id < 3 else 'validation' if group_id == 3 else 'test'
        manifest = dict(group=group_id, role=role, split_seed=seed,
                        initialization_seeds=list(initialization_seeds[group_id]))
        for name, positions in [('train', train), ('stop', stop)]:
            manifest[name + '_indices'] = indices[positions].tolist()
            manifest[name + '_native_ids'] = native_ids[indices[positions]].tolist()
            manifest[name + '_labels_sha256'] = tensor_sha256(labels[positions])
        manifest['supervision_sha256'] = json_sha256(manifest)
        grouped = replace(view, train_indices=indices[train], train_labels=labels[train],
                          val_indices=indices[stop], val_labels=labels[stop],
                          provenance={**view.provenance, 'formal_group': manifest})
        validate_source_training_data(grouped)
        groups.append((manifest, grouped))
    return groups


def overlap_audit(groups):
    """Exact supervision overlaps, without arbitrary diversity pass thresholds."""
    rows = []
    for i, (a, _) in enumerate(groups):
        for b, _ in groups[i+1:]:
            row = dict(left=a['group'], right=b['group'])
            for part in ('train', 'stop'):
                left, right = set(a[part + '_native_ids']), set(b[part + '_native_ids'])
                row[part + '_intersection'] = len(left & right)
                row[part + '_union'] = len(left | right)
            rows.append(row)
    return rows
