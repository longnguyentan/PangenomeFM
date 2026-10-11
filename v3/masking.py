"""Leakage-safe masked allele observations; truth remains separate from inputs."""
import torch
MISSING, MASK = -1, -2

def mask_alleles(truth, structural_sites, *, generator, rate=.15, spans=(50, 500), span_starts=None, imputation_rows=None):
    if truth.ndim != 2 or structural_sites.shape != truth.shape[1:] or not 0 < rate < 1:
        raise ValueError('bound whole haplotype/site matrix required')
    if (truth < MISSING).any(): raise ValueError('input truth cannot already contain masks')
    eligible = truth >= 0
    selected = torch.rand(truth.shape, generator=generator, device='cpu').to(truth.device) < rate
    if span_starts is not None:
        if not (1 <= spans[0] <= spans[1]): raise ValueError('valid whole span bounds required')
        for row, start, length in span_starts:
            if not 0 <= row < truth.shape[0] or not 0 <= start < truth.shape[1] or not spans[0] <= length <= spans[1]:
                raise ValueError('exact prespecified span required')
            selected[row, start:min(start + length, truth.shape[1])] = True
    if imputation_rows is not None:
        for row in imputation_rows:
            if not 0 <= row < truth.shape[0]: raise ValueError('exact query row required')
            selected[row] = structural_sites
    selected &= eligible
    observed = truth.clone(); observed[selected] = MASK
    return observed, selected

def query_panel(train, query, assignments, query_assignment):
    if set(assignments) != {'TRAIN'} or query_assignment not in ('TRAIN_CROSSFIT', 'DEVELOPMENT', 'EXTERNAL'):
        raise ValueError('DEV/SEALED/query donors may not enter the reference panel')
    if train.ndim != 2 or query.ndim != 2 or query.shape[0] != 1 or train.shape[1] != query.shape[1]:
        raise ValueError('all frozen TRAIN reference rows plus ONE query haplotype required')
    return torch.cat((train, query), 0), train.shape[0]


def nested_target_closure(selected, ancestors, overlaps):
    """Mask structural target, ancestors, descendants and overlapping query sites."""
    result = set(selected)
    for i in list(result): result.update(overlaps.get(i, ()))
    changed = True
    while changed:
        previous = set(result)
        for child, parent in ancestors.items():
            if child in result or parent in result: result.update((child, parent))
        for i in list(result): result.update(overlaps.get(i, ()))
        changed = result != previous
    return result
