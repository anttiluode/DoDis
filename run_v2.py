"""DoDis v2 — visible-local Sihti patch memory vs whole-image memory.

Frozen CNN, no retraining; 100 correctly recognized unseen CIFAR test birds by
default. Results are saved per bird for safe restart and paired uncertainty.
Usage: python run_v2.py --sihti-root ../SihtiGraphModel --device cuda
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from io import BytesIO
import base64

import numpy as np
from dodis import (band_pieces,render,features,Normalizer,BindingGraph,
                   nearest_donors,Settler)
from local_memory import LocalPatchMemory
from run import fit_purpose,score_cnn

MODES=('masked_mean','global_memory','local_memory','local_binding',
       'local_shuffled','local_purpose')

def atomic_json(path:Path,obj:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(obj,indent=2,allow_nan=False)+'\n'
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(payload,encoding='utf-8')
    os.replace(tmp,path)

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def paired_stats(a,b,seed=41,B=5000):
    """Distribution of paired mean(a-b). Negative = a wins if MSE."""
    x=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    if x.ndim!=1 or not len(x):raise ValueError('No paired cases')
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(x),(B,len(x)))
    diffs=np.mean(x[idx],axis=1)
    flips=rng.choice(np.array([-1.,1.]),size=(B,len(x)))
    random_means=np.mean(x[None]*flips,axis=1)
    observed=float(np.mean(x))
    p=(1+np.count_nonzero(np.abs(random_means)>=abs(observed)))/(B+1)
    return {'n':len(x),'mean_difference_a_minus_b':observed,
            'ci95_bootstrap':[float(z) for z in np.percentile(diffs,[2.5,97.5])],
            'paired_signflip_p_two_sided':float(p),
            'a_lower_than_b_count':int(np.sum(x<0)),
            'ties':int(np.sum(x==0)),
            'note':'Exploratory paired bootstrap and sign-flip test; fixed run/seed, no population claim'}

def png(x):
    from PIL import Image
    image=(np.clip(x,0,1).transpose(1,2,0)*255+0.5).astype('uint8')
    f=BytesIO()
    Image.fromarray(image).resize((256,256),Image.Resampling.NEAREST).save(f,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(f.getvalue()).decode('ascii')

def score_repair(pieces,truth,full,visible,corrupt_margin,model,device):
    x=render(pieces,clip=True)
    margin=float(score_cnn(model,x[None],device)[0])
    return x,{'pixel_mse':float(np.mean((x-truth)**2)),
              'hidden_residue_mse':float(np.mean((pieces[~visible]-full[~visible])**2)),
              'cnn_margin':margin,'cnn_margin_gain_over_corrupt':margin-corrupt_margin}

def choose_test_indices(model,ds,device,n,minimum=100):
    import torch
    from sihtigraph.teacher import normalize
    found=[]
    with torch.inference_mode():
        for i in range(minimum,len(ds)):
            if ds.targets[i]!=2:continue
            image=ds[i][0].unsqueeze(0).to(device)
            if int(model(normalize(image)).argmax(1).item())==2:
                found.append(i)
                if len(found)>=n:break
    if len(found)!=n:raise RuntimeError(f'Needed {n} correctly classified test birds, found {len(found)}')
    return found

def run(args):
    import torch
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    repo=Path(args.sihti_root).resolve()
    if not (repo/'sihtigraph'/'teacher.py').is_file():
        raise SystemExit(f'Missing SihtiGraphModel checkout at {repo}')
    sys.path.insert(0,str(repo))
    from sihtigraph.teacher import load_teacher
    ck=(Path(args.checkpoint).resolve() if args.checkpoint else
        repo/'runs/checkpoints/cifar_tiny_cnn.pt')
    if not ck.exists():raise SystemExit(f'Frozen checkpoint not found: {ck}')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA not available')
    model,model_info=load_teacher(str(ck),device)
    data_root=Path(args.data).resolve() if args.data else repo/'data'
    train=CIFAR10(str(data_root),train=True,download=False,transform=ToTensor())
    test=CIFAR10(str(data_root),train=False,download=False,transform=ToTensor())
    birds=[i for i,y in enumerate(train.targets) if y==2]
    others=[i for i,y in enumerate(train.targets) if y!=2]
    split_rng=np.random.default_rng(41)
    split_rng.shuffle(birds);split_rng.shuffle(others)
    na,nb,nc=args.memory_size,args.binding_size,args.purpose_birds
    if na+nb+nc+args.purpose_val_birds>len(birds):
        raise SystemExit('Not enough disjoint bird training examples')
    mem_ids=birds[:na]
    bind_ids=birds[na:na+nb]
    purpose_ids=birds[na+nb:na+nb+nc]+others[:args.purpose_others]
    validation_ids=(birds[na+nb+nc:na+nb+nc+args.purpose_val_birds]+
                    others[args.purpose_others:args.purpose_others+args.purpose_val_others])
    eval_ids=choose_test_indices(model,test,device,args.eval_birds)
    def tensors(ds,ids):
        return np.stack([ds[int(i)][0].numpy() for i in ids])
    print('Frozen teacher; disjoint train groups:',len(mem_ids),len(bind_ids),
          len(purpose_ids),len(validation_ids),flush=True)
    mem=band_pieces(tensors(train,mem_ids),device=device)
    bind_train=band_pieces(tensors(train,bind_ids),device=device)
    purpose_train=band_pieces(tensors(train,purpose_ids),device=device)
    purpose_valid=band_pieces(tensors(train,validation_ids),device=device)
    local_index=LocalPatchMemory(mem,train_ids=mem_ids)
    norm=Normalizer.learn(features(bind_train))
    binding=BindingGraph.learn(norm.apply(features(bind_train)),max_edges=80)
    purpose,quality=fit_purpose(model,purpose_train,purpose_valid,norm,binding,device)
    print(f'Purpose validation R²={purpose.val_r2:.4f}; '
          f'local donor source bank={len(mem)*16} patches per band',flush=True)
    signature={'protocol':'dodis_v2_visible_local_1','checkpoint_sha256':sha256(ck),
               'seed':args.seed,'test_ids':eval_ids,
               'memory':mem_ids,'binding':bind_ids,
               'purpose_train':purpose_ids,'purpose_validation':validation_ids,
               'params':{k:v for k,v in vars(args).items() if k not in ('docs','out','sihti_root','data','checkpoint','device')}}
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding='utf-8'))!=signature:
            raise SystemExit('Existing run manifest differs: use a different --out directory')
    else:
        if list(out.glob('bird_*.json')):
            raise SystemExit('Results without a manifest: choose a new --out directory')
        atomic_json(manifest,signature)
    records=[]
    visual_path=out/'first_visual.json'
    visual=(json.loads(visual_path.read_text(encoding='utf-8'))
            if visual_path.exists() else None)
    for pos,index in enumerate(eval_ids):
        file=out/f'bird_{index:05d}.json'
        # For transparency, this loop can resume after any fully saved test bird.
        if file.exists():
            record=json.loads(file.read_text(encoding='utf-8'))
            print(f'[{pos+1}/{len(eval_ids)}] bird {index} restored from saved receipt',flush=True)
            records.append(record)
            continue
        truth=test[index][0].numpy()
        full=band_pieces(truth[None],device=device)[0]
        if np.max(np.abs(render(full)-truth))>2e-6:
            raise AssertionError('Exact additive sieve reconstruction failed')
        rng=np.random.default_rng(np.random.SeedSequence([args.seed,index,774]))
        visible=(rng.random(64)<.5)
        if not visible.any() or visible.all():raise AssertionError('Degenerate random mask')
        observed=np.zeros_like(full)
        observed[visible]=full[visible]
        local=local_index.retrieve(observed,visible,k=args.donors)
        donor_idx,donor_dist=nearest_donors(observed,visible,mem,k=args.donors)
        corrupt=render(observed,clip=True)
        corrupt_margin=float(score_cnn(model,corrupt[None],device)[0])
        original_margin=float(score_cnn(model,truth[None],device)[0])
        modes={}
        images={}
        base_img,base_stat=score_repair(observed,truth,full,visible,corrupt_margin,model,device)
        modes['masked_mean']={**base_stat,'moves':0,'monotone':True,
                              'energy_trace':[],'energy_before':None,'energy_after':None}
        images['masked_mean']=base_img
        for mode in MODES[1:]:
            if mode=='global_memory':
                donors=mem[donor_idx];dist=donor_dist
            else:
                donors=local.candidates;dist=local.distances
            graph=binding if mode in ('local_binding','local_purpose') else (
                binding.permuted(91) if mode=='local_shuffled' else None)
            pg=purpose if mode=='local_purpose' else None
            solver=Settler(observed,visible,donors,dist,norm,graph,pg,
                           beta=args.memory_weight,
                           gamma=args.binding_weight,
                           lam=args.purpose_weight if pg is not None else 0.,
                           smooth=args.smooth_weight)
            result=solver.solve(passes=args.passes,seed=args.seed+index)
            if not result['monotone'] or not np.array_equal(result['pieces'][visible],observed[visible]):
                raise AssertionError('Settler violated monotonicity or observed data')
            img,stat=score_repair(result['pieces'],truth,full,visible,corrupt_margin,model,device)
            modes[mode]={**stat,'energy_before':result['energy_before'],
                         'energy_after':result['energy_after'],
                         'energy_terms':result['parts'],'energy_trace':result['trace'],
                         'moves':result['moves'],'monotone':result['monotone']}
            images[mode]=img
        hidden=np.flatnonzero(~visible)
        record={'index':index,'test_label':'bird','visible':int(visible.sum()),
                'hidden':len(hidden),'teacher_true_margin':original_margin,
                'teacher_corrupt_margin':corrupt_margin,
                'mask':visible.astype(int).tolist(),
                'local_context_zero_count':int(sum(local.observed_context_counts[j]==0 for j in hidden)),
                'local_context_counts':[local.observed_context_counts[j] for j in hidden],
                'local_donor_provenance':{str(j):local.provenance[j] for j in hidden},
                'global_donor_train_ids':[int(mem_ids[i]) for i in donor_idx],
                'modes':modes}
        atomic_json(file,record)
        records.append(record)
        print(f'[{pos+1}/{len(eval_ids)}] bird {index} / '+
              ' / '.join(f'{key}: {modes[key]["pixel_mse"]:.5f}' for key in MODES),flush=True)
        if pos==0:
            visual={'index':index,'true':png(truth),'corrupted':png(corrupt),
                    'reconstructions':{m:png(images[m]) for m in MODES},
                    'modes':modes,'visible_pieces':visible.astype(int).tolist(),
                    'teacher_margin':original_margin}
            atomic_json(visual_path,visual)
    def vals(key):
        return np.asarray([r['modes'][key]['pixel_mse'] for r in records])
    summary={}
    for name in MODES:
        arr=vals(name)
        summary[name]={'mean_pixel_mse':float(np.mean(arr)),
                       'median_pixel_mse':float(np.median(arr)),
                       'mean_hidden_residue_mse':float(np.mean([
                           r['modes'][name]['hidden_residue_mse'] for r in records])),
                       'mean_cnn_margin':float(np.mean([
                           r['modes'][name]['cnn_margin'] for r in records])),
                       'mse_improved_vs_masked_mean_count':int(np.sum(arr<vals('masked_mean'))),
                       'mse_improved_vs_local_memory_count':int(np.sum(arr<vals('local_memory')))}
    comparisons={}
    for a,b in [('local_memory','masked_mean'),
                ('global_memory','masked_mean'),
                ('local_memory','global_memory'),
                ('local_binding','local_memory'),
                ('local_binding','local_shuffled'),
                ('local_purpose','local_binding')]:
        comparisons[f'{a}_vs_{b}']=paired_stats(vals(a),vals(b),seed=args.seed+201)
    gates={'G0_external_content_and_fixed_visible':True,
           'G1_all_modes_monotone':all(
               r['modes'][m]['monotone'] for r in records for m in MODES),
           'G2_local_beats_masked_mean':bool(summary['local_memory']['mean_pixel_mse']<
                                            summary['masked_mean']['mean_pixel_mse']),
           'G3_learned_beats_shuffled':bool(
               summary['local_binding']['mean_pixel_mse']<
               summary['local_shuffled']['mean_pixel_mse']),
           'G3_pairwise_CI_excludes_zero':bool(
               comparisons['local_binding_vs_local_shuffled']['ci95_bootstrap'][1]<0),
           'G4_purpose_raises_margin':bool(summary['local_purpose']['mean_cnn_margin']>
                                            summary['local_binding']['mean_cnn_margin'])}
    result={'protocol':signature['protocol'],'complete':True,
            'sample_size':len(records),'split':{'training_roles_disjoint':True,
                  'memory_count':len(mem_ids),'binding_count':len(bind_ids),
                  'purpose_fit_count':len(purpose_ids),'purpose_validation_count':len(validation_ids),
                  'test_indices':eval_ids},
            'source_checkpoint_sha256':sha256(ck),'teacher_info':model_info,
            'purpose_validation':quality,'modes':list(MODES),
            'summary':summary,'comparisons':comparisons,'gates':gates,
            'per_image':records,
            'notes':['No target hidden residue enters retrieval or settling.',
                     'Per-piece local donor patches may come from different source tile positions.',
                     'Mean masks are a baseline, not automatically beaten by completion.',
                     'Reported paired p-values are exploratory on one fixed model and random seed.',
                     'A higher CNN margin is not a ground-truth image reconstruction score.']}
    atomic_json(out/'results.json',result)
    if visual is None:
        # A receipt can be absent only if the first image was not completed; 
        # do not manufacture a replacement image from a later example.
        print('No new first-bird visualization; prior local viewer receipt is unaffected.',flush=True)
    else:
        visual.update({'protocol':result['protocol'],'n_test':len(records),
                       'summary':summary,'gates':gates,
                       'comparisons':comparisons,'purpose':quality})
        atomic_json(Path(args.docs),visual)
    print('Complete:',out/'results.json')
    print('MSE',json.dumps({k:v['mean_pixel_mse'] for k,v in summary.items()}))
    print('Paired learned-vs-shuffled',json.dumps(
        comparisons['local_binding_vs_local_shuffled']))
    return result

def main():
    p=argparse.ArgumentParser(description='DoDis v2 local Sihti patch retrieval')
    p.add_argument('--sihti-root',default='../SihtiGraphModel')
    p.add_argument('--checkpoint',default='')
    p.add_argument('--data',default='')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--out',default='runs/v2_local_100_birds')
    p.add_argument('--docs',default='docs/receipt_v2.json')
    p.add_argument('--eval-birds',type=int,default=100)
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--memory-size',type=int,default=256)
    p.add_argument('--binding-size',type=int,default=320)
    p.add_argument('--purpose-birds',type=int,default=380)
    p.add_argument('--purpose-others',type=int,default=220)
    p.add_argument('--purpose-val-birds',type=int,default=70)
    p.add_argument('--purpose-val-others',type=int,default=70)
    p.add_argument('--donors',type=int,default=4)
    p.add_argument('--passes',type=int,default=3)
    p.add_argument('--memory-weight',type=float,default=.05)
    p.add_argument('--smooth-weight',type=float,default=.75)
    p.add_argument('--binding-weight',type=float,default=.4)
    p.add_argument('--purpose-weight',type=float,default=.12)
    args=p.parse_args()
    if args.eval_birds<2 or args.donors<2:p.error('Need at least 2 test birds and 2 donors')
    run(args)

if __name__=='__main__':
    main()
