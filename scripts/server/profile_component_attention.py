#!/usr/bin/env python3
"""Optimizer-free forward/backward smoke on the largest prepared real context.

This is resource/correctness profiling, not pretraining or biological evaluation.
Use a separate process per mode to make CPU peak RSS interpretable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import torch

from evaluation.external import _namespace_from_checkpoint, _build_model_from_checkpoint
from graph.features import build_oid_metadata_from_segments
from graph.slicing import build_global_index
from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import verified_fingerprint, write_json
from training.masked_features import MaskedFeatureObjective, segment_mask
from training.pretrain import encoder_parameter_digest, load_slice, seed_everything, tensorize_slice


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contexts', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--mode', choices=['chunked_exact','chunked_window'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--chunk-size', type=int, default=512)
    ap.add_argument('--window-k', type=int, default=128)
    cli = ap.parse_args()
    if cli.threads < 1:
        raise ValueError('Positive CPU thread limit required')
    cli.out_dir.mkdir(parents=True, exist_ok=False)
    started=time.time()
    record=dict(status='running', scope='Resource profiling only; no optimizer, label access or checkpoint selection',
        biological_labels_used=False, weight_updates=0, mode=cli.mode,device=cli.device,
        checkpoint=fingerprint(cli.checkpoint),implementation=fingerprint(Path(__file__)))
    write_json(cli.out_dir/'status.json',record)
    try:
        torch.set_num_threads(cli.threads)
        audit=json.loads((cli.contexts/'audit.json').read_text())
        if audit['status']!='complete' or audit['failed_windows'] or not audit['materialized']:
            raise ValueError('Complete materialized context audit required')
        plan=audit['plan']
        for key in ['full_segments','sequence_cache']:
            verified_fingerprint(Path(plan[key]),plan[key+'_sha256'])
        windows=pd.read_csv(cli.contexts/'manifest.csv')
        row=windows.sort_values(['n_oriented_handles','name'],ascending=[False,True]).iloc[0]
        record.update(window=str(row['name']),n_segments=int(row.n_segments),n_handles=int(row.n_oriented_handles),
            original_dense_score_bytes=int(row.dense_score_bytes),
            context_audit=fingerprint(cli.contexts/'audit.json'),manifest=fingerprint(cli.contexts/'manifest.csv'),
            segments=fingerprint(Path(row.segments_path)),links=fingerprint(Path(row.links_path)))
        checkpoint=torch.load(cli.checkpoint,map_location='cpu',weights_only=False)
        args=_namespace_from_checkpoint(checkpoint,42)
        if checkpoint['in_dim']!=519 or args.hidden_dim!=48 or args.n_layers!=2 or args.n_heads!=4:
            raise ValueError('Expected the fixed 519-input / 48D / two-layer / four-head encoder')
        args.objective='masked_nt_features'
        args.extraction_mode=True
        args.canonical_conflict_policy='exclude'
        args.node_feature_cache=plan['sequence_cache']
        args.node_feature_min_coverage=1.
        args.coordinate_attention_mode=cli.mode
        args.window_k=cli.window_k if cli.mode=='chunked_window' else None
        args.adaptive_window=False
        args.attention_chunk_size=cli.chunk_size
        args.device=cli.device
        segments=pd.read_csv(plan['full_segments'],usecols=lambda c:c!='seq')
        index,_=build_global_index(segments)
        metadata=build_oid_metadata_from_segments(segments,index)
        raw=load_slice(row,index,metadata,segments,args)
        if raw is None or len(raw['nodes'])!=int(row.n_oriented_handles):
            raise ValueError('Native loader lost nodes from the prepared graph')
        device=torch.device(cli.device)
        seed_everything(42)
        encoder,_=_build_model_from_checkpoint(checkpoint,args,device)
        initial=encoder_parameter_digest(encoder)
        # Fixed zero/unit moments avoid data-fitted normalization in this profile.
        # They are NOT moments for any new scientific pretraining experiment.
        objective=MaskedFeatureObjective(encoder,519,48,4,np.zeros(512,np.float32),np.ones(512,np.float32)).to(device)
        objective.train()
        sd=tensorize_slice(raw,device,args)
        mask=segment_mask(sd['node_oids'],.3,42)
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        compute=time.time()
        loss,n=objective(sd,mask)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite full-context loss')
        loss.backward()
        gradients=[p.grad for p in objective.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError('Missing or nonfinite full-context gradients')
        if device.type=='cuda':
            torch.cuda.synchronize(device)
        if encoder_parameter_digest(encoder)!=initial:
            raise ValueError('Profiling changed encoder weights')
        record.update(status='complete',loss=float(loss.detach()),n_masked_segments=n,
            all_gradients_finite=True,encoder_weights_unchanged=True,
            forward_backward_seconds=time.time()-compute,wall_seconds=time.time()-started,
            cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
            torch_version=torch.__version__,chunk_size=cli.chunk_size,window_k=args.window_k,
            limitations=['Single optimizer-free step; not a convergence or biological-performance result.',
                'CPU peak RSS includes graph/cache loading and is not GPU peak memory.',
                'Exact mode preserves global mathematical attention, but floating-point/dropout order differs.',
                'Window mode changes context and repairs local-attention semantics; requires a separately fixed protocol.'])
        write_json(cli.out_dir/'audit.json',record)
    except Exception as exc:
        record.update(status='failed',error=repr(exc),wall_seconds=time.time()-started)
        raise
    finally:
        write_json(cli.out_dir/'status.json',record)


if __name__=='__main__':
    main()
