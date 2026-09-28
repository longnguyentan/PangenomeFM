"""Prespecified, validation-only masked NT objective on the native encoder."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import time
import zlib

import numpy as np
import pandas as pd
import torch

from evaluation.external import _namespace_from_checkpoint
from evaluation.splits import normalize_chrom, validate_chromosome_split
from graph.features import build_oid_metadata_from_segments
from graph.slicing import build_global_index
from models.dual_stream_gat import DualStreamPangenomeGAT
from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import verified_fingerprint, write_json
from training.masked_features import MaskedFeatureObjective, segment_mask, segment_target_statistics
from training.pretrain import encoder_parameter_digest, load_slice, seed_everything, tensorize_slice


def make_encoder(args):
    return DualStreamPangenomeGAT(in_dim=519, hidden_dim=args.hidden_dim,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
        edge_mlp_dim=args.hidden_dim*2, use_multiscale_rope=True, n_rope_scales=3,
        use_orientation=True, stream_mode=args.stream_mode, graph_message_direction='bidirectional')


def mask_seed(seed: int, name: str, epoch: int, view: int = 0) -> int:
    return (zlib.crc32(name.encode()) + seed*1000003 + epoch*1000033 + view*1000037) % (2**32)


def validation(objective, slices, args, plan):
    objective.eval()
    rows = []
    with torch.no_grad():
        for raw in slices:
            sd = tensorize_slice(raw, torch.device(args.device), args)
            for view in range(3):
                mask = segment_mask(sd['node_oids'], plan['mask_rate'], mask_seed(args.seed, raw['name'], 0, view))
                loss, n = objective(sd, mask)
                rows.append(dict(window=raw['name'], view=view, n_masked_segments=n, loss=float(loss)))
    frame = pd.DataFrame(rows)
    value = float(np.average(frame.loss, weights=frame.n_masked_segments))
    if not np.isfinite(value):
        raise ValueError('Nonfinite validation reconstruction loss')
    return value, frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'template-checkpoint', 'manifest', 'full-segments', 'nt-cache', 'out-dir']:
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--device', default='cuda')
    cli = ap.parse_args()
    plan = json.loads(cli.config.read_text())
    if cli.seed not in plan['seeds'] or cli.arm not in plan['arms']:
        raise ValueError('Undeclared initialization or arm')
    sources = [verified_fingerprint(cli.full_segments, plan['full_segments_sha256']),
        verified_fingerprint(cli.manifest, plan['manifest_sha256']),
        verified_fingerprint(cli.nt_cache, plan['nt_cache_sha256']), fingerprint(cli.template_checkpoint)]
    template = torch.load(cli.template_checkpoint, map_location='cpu', weights_only=False)
    args = _namespace_from_checkpoint(template, cli.seed)
    expected = dict(plan['encoder'], node_extra_features='cache', node_structure_source='visible',
        junction_geometry_match='signed_gap_bins', junction_geometry_bin_ratio=1.25,
        split_seed=20260806, pop_cond=False, use_edge_features=False, adaptive_window=False,
        window_k=None, no_rope=False, no_fusion_gate=False)
    if any(getattr(args, k, None) != v for k, v in expected.items()) or template['in_dim'] != 519:
        raise ValueError('Template does not have the fixed Q-branch architecture/input policy')
    reference = json.loads(Path('configs/server_full_multicohort_20260806.json').read_text())
    fold = next(f for f in reference['rotating_chromosome_folds'] if f['name'] == plan['fold'])
    if set(args.test_chrs) != set(fold['test']) or set(args.val_chrs) != set(fold['validation']):
        raise ValueError('Template chromosome partition differs')
    validate_chromosome_split(val_chrs=args.val_chrs, test_chrs=args.test_chrs)
    args.device, args.node_feature_cache, args.node_feature_min_coverage = cli.device, str(cli.nt_cache), 1.
    args.stream_mode = 'coordinate' if cli.arm.startswith('coordinate') else 'full'
    args.freeze_encoder = cli.arm.endswith('random')
    args.canonical_conflict_policy = 'exclude'
    args.objective = 'junction_repair'  # Loader eligibility only; not this objective's loss.
    args.extraction_mode = False
    args.validation_only = True
    args.drop_edge, args.drop_edge_rate, args.mask_query_edges = False, 0., False
    args.linear_predictor, args.pair_geometry = False, False
    args.manifest, args.full_segments, args.out_dir = str(cli.manifest), str(cli.full_segments), str(cli.out_dir)
    for key, value in plan['optimization'].items():
        setattr(args, key, value)
    cli.out_dir.mkdir(parents=True, exist_ok=False)
    record = dict(status='loading', plan=plan, config=fingerprint(cli.config), sources=sources,
        seed=cli.seed, arm=cli.arm, code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        biological_labels_used=False, test_windows_loaded=0, template_weights_loaded=False,
        source_implementation=fingerprint(Path(__file__)), started_at=time.time())
    write_json(cli.out_dir/'status.json', record)
    try:
        manifest = pd.read_csv(cli.manifest)
        selected = manifest.loc[manifest.closure.eq('1hop') & ~manifest.target_sn.map(normalize_chrom).isin(args.test_chrs)]
        segments = pd.read_csv(cli.full_segments, usecols=lambda c: c != 'seq')
        index, _ = build_global_index(segments)
        md = build_oid_metadata_from_segments(segments, index)
        train, val, audits = [], [], []
        for _, row in selected.iterrows():
            audit = dict(window=row['name'], chrom=normalize_chrom(row['target_sn']))
            raw = load_slice(row, index, md, segments, args, audit)
            audits.append(audit)
            if raw is not None:
                (val if audit['chrom'] in args.val_chrs else train).append(raw)
        pd.DataFrame(audits).to_csv(cli.out_dir/'windows.csv', index=False)
        if not train or not val:
            raise ValueError('Empty pretraining partition')
        ids, mean, scale = segment_target_statistics(train)
        val_ids = np.unique(np.concatenate([r['nodes']//2 for r in val]))
        if np.intersect1d(ids, val_ids).size:
            raise ValueError('A segment occurs in both train and validation windows')
        np.savez_compressed(cli.out_dir/'target_moments.npz', training_segment_ids=ids, mean=mean, scale=scale)
        record.update(status='training', n_train_windows=len(train), n_val_windows=len(val),
            n_unique_training_segments=len(ids), train_val_segment_overlap=0)
        write_json(cli.out_dir/'status.json', record)
        # Loading/preprocessing consumes no initialization-dependent randomness.
        seed_everything(cli.seed)
        encoder = make_encoder(args)
        initial = encoder_parameter_digest(encoder)
        objective = MaskedFeatureObjective(encoder, 519, args.hidden_dim, args.n_heads,
            mean, scale, args.freeze_encoder).to(cli.device)
        opt = plan['optimization']
        optimizer = torch.optim.AdamW([p for p in objective.parameters() if p.requires_grad],
            lr=opt['lr'], weight_decay=opt['weight_decay'])
        best, stale, history = float('inf'), 0, []
        for epoch in range(1, opt['epochs']+1):
            objective.train()
            optimizer.zero_grad(set_to_none=True)
            order = np.random.default_rng(cli.seed+epoch).permutation(len(train))
            total = 0.
            for position, idx in enumerate(order):
                raw = train[idx]
                sd = tensorize_slice(raw, torch.device(cli.device), args)
                mask = segment_mask(sd['node_oids'], plan['mask_rate'], mask_seed(cli.seed, raw['name'], epoch))
                loss, _ = objective(sd, mask)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                group_start = (position//opt['accum_steps'])*opt['accum_steps']
                denominator = min(opt['accum_steps'], len(order)-group_start)
                (loss/denominator).backward()
                total += float(loss.detach())
                if (position+1) % opt['accum_steps'] == 0 or position+1 == len(order):
                    torch.nn.utils.clip_grad_norm_(objective.parameters(), opt['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            score, per_window = validation(objective, val, args, plan)
            history.append(dict(epoch=epoch, train_loss=total/len(train), validation_loss=score, elapsed_seconds=time.time()-record['started_at']))
            pd.DataFrame(history).to_csv(cli.out_dir/'epochs.csv', index=False)
            print(json.dumps(history[-1]), flush=True)
            if score < best:
                best, stale = score, 0
                saved_args = copy.copy(args)
                saved_args.objective = 'masked_nt_features'
                payload = dict(model_state=encoder.state_dict(), predictor_state=None, in_dim=519, edge_feat_dim=0,
                    args=vars(saved_args), closure='1hop', stream_mode=args.stream_mode,
                    pretraining_objective='segment-grouped standardized frozen NT reconstruction',
                    best_validation_loss=best, epochs_run=epoch, initial_encoder_sha256=initial,
                    final_encoder_sha256=encoder_parameter_digest(encoder), plan=plan)
                torch.save(payload, cli.out_dir/'checkpoint.pt')
                torch.save(objective.state_dict(), cli.out_dir/'objective.pt')
                per_window.to_csv(cli.out_dir/'selected_validation.csv', index=False)
            else:
                stale += 1
            torch.save(dict(epoch=epoch, model=objective.state_dict(), optimizer=optimizer.state_dict(),
                plan=plan, history=history), cli.out_dir/'recovery.pt')
            if stale >= opt['patience']:
                break
        objective.load_state_dict(torch.load(cli.out_dir/'objective.pt', map_location=cli.device, weights_only=True))
        replay, _ = validation(objective, val, args, plan)
        if not np.isclose(replay, best, rtol=0, atol=1e-7):
            raise ValueError('Selected reconstruction score does not replay')
        final = encoder_parameter_digest(encoder)
        if args.freeze_encoder and final != initial:
            raise ValueError('Frozen random backbone changed')
        if not args.freeze_encoder and final == initial:
            raise ValueError('Trained backbone did not update')
        record.update(status='complete', best_validation_loss=best, replayed_validation_loss=replay,
            epochs_completed=len(history), checkpoint=fingerprint(cli.out_dir/'checkpoint.pt'),
            initial_encoder_sha256=initial, final_encoder_sha256=final, finished_at=time.time())
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(cli.out_dir/'status.json', record)


if __name__ == '__main__':
    main()
