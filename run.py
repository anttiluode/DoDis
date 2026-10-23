"""DoDis real held-out experiment (RTX 3060 friendly).

Requires a sibling checkout of SihtiGraphModel with its trained CIFAR CNN
and dataset. Does not download, train, or alter the CNN. No target hidden
Sihti piece is used for candidate retrieval, graph training or settling.
"""
from __future__ import annotations
import argparse
import base64
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from dodis import (MEAN,band_pieces,render,features,Normalizer,BindingGraph,
                   PurposeGraph,nearest_donors,Settler)

def score_cnn(model,images,device,batch=128):
    import torch
    from sihtigraph.teacher import normalize
    a=np.asarray(images,dtype=np.float32)
    out=[]
    with torch.inference_mode():
        for start in range(0,len(a),batch):
            x=torch.from_numpy(np.clip(a[start:start+batch],0,1)).to(device)
            logits=model(normalize(x))
            bird=logits[:,2]
            neg=logits.clone();neg[:,2]=-1e9
            out.append((bird-neg.max(1).values).cpu().numpy())
    return np.concatenate(out).astype(float)

def seed_masks(rng,n,kind):
    if kind=='half':return rng.random(n)<.5
    if kind=='full':return np.ones(n,dtype=bool)
    raise ValueError(kind)

def fit_purpose(model,train_pieces,val_pieces,norm,binding,device,seed=41):
    rng=np.random.default_rng(seed+7821)
    def gather(parts):
        x=[];f=[]
        for p in parts:
            for prob in (.5,.75,.9,1.):
                mask=rng.random(64)<prob
                masked=p*mask[:,None,None,None]
                f.append(norm.apply(features(masked)))
                x.append(render(masked,clip=True))
        scores=score_cnn(model,np.stack(x),device)
        return np.stack(f),scores
    uf,yfit=gather(train_pieces)
    uv,yval=gather(val_pieces)
    graph=PurposeGraph.fit(uf,yfit,uv,yval,binding.edges)
    return graph,{'fit_observations':len(yfit),'validation_observations':len(yval),
                  'validation_r2':graph.val_r2,'teacher_margin_sd':graph.teacher_sd,
                  'note':'Shared teacher-distilled regression on training images, NOT v3 image-specific graph'}

def data_image(tensor):
    from PIL import Image
    a=(np.clip(tensor,0,1).transpose(1,2,0)*255+0.5).astype(np.uint8)
    fp=BytesIO()
    Image.fromarray(a).resize((256,256),resample=Image.Resampling.NEAREST).save(fp,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(fp.getvalue()).decode('ascii')

def run(args):
    import torch
    sihti=Path(args.sihti_root).resolve()
    if not (sihti/'sihtigraph'/'teacher.py').is_file():
        raise SystemExit(f'Missing SihtiGraphModel checkout at {sihti}; set --sihti-root')
    sys.path.insert(0,str(sihti))
    from sihtigraph.teacher import load_teacher,normalize
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    ck=Path(args.checkpoint).resolve() if args.checkpoint else sihti/'runs/checkpoints/cifar_tiny_cnn.pt'
    if not ck.is_file():raise SystemExit(f'No trained CNN checkpoint: {ck}')
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else args.device
    if device=='auto':device='cpu'
    if device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA requested but not available')
    model,meta=load_teacher(str(ck),device)
    data_root=Path(args.data).resolve() if args.data else sihti/'data'
    training=CIFAR10(str(data_root),train=True,download=False,transform=ToTensor())
    testing=CIFAR10(str(data_root),train=False,download=False,transform=ToTensor())
    birds=[i for i,y in enumerate(training.targets) if y==2]
    others=[i for i,y in enumerate(training.targets) if y!=2]
    rng=np.random.default_rng(41)
    # Immutable disjoint TRAIN images for all 3 roles.
    rng.shuffle(birds);rng.shuffle(others)
    na,nb,nc=args.memory_size,args.binding_size,args.purpose_birds
    if na+nb+nc+args.purpose_val_birds>len(birds):raise SystemExit('Too many train birds')
    memory_idx=birds[:na];binding_idx=birds[na:na+nb]
    purpose_idx=(birds[na+nb:na+nb+nc]+others[:args.purpose_others])
    purpose_val_idx=(birds[na+nb+nc:na+nb+nc+args.purpose_val_birds]+
                     others[args.purpose_others:args.purpose_others+args.purpose_val_others])
    def take(ds,ids):
        return np.stack([ds[int(i)][0].numpy() for i in ids])
    print(f'Memory {len(memory_idx)} separate train birds; graph A {len(binding_idx)} train birds; '
          f'graph B {len(purpose_idx)} train originals; B-validation {len(purpose_val_idx)}',flush=True)
    memory=band_pieces(take(training,memory_idx),device=device)
    binding_pieces=band_pieces(take(training,binding_idx),device=device)
    purpose_train=band_pieces(take(training,purpose_idx),device=device)
    purpose_val=band_pieces(take(training,purpose_val_idx),device=device)
    norm=Normalizer.learn(features(binding_pieces))
    bind=BindingGraph.learn(norm.apply(features(binding_pieces)),max_edges=80)
    purpose,quality=fit_purpose(model,purpose_train,purpose_val,norm,bind,device)
    print(f'Graph A {len(bind.edges)} learned edges; graph B independent '
          f'validation R²={purpose.val_r2:.4f}',flush=True)
    # Test targets are separate CIFAR TEST images, not the train memory or graph fit.
    selected=[]
    with torch.inference_mode():
        for idx in range(100,len(testing)):
            if testing.targets[idx]!=2:continue
            image=testing[idx][0].unsqueeze(0).to(device)
            if int(model(normalize(image)).argmax(1).item())==2:
                selected.append(idx)
                if len(selected)>=args.eval_birds:break
    if len(selected)<args.eval_birds:raise SystemExit('Insufficient correctly classified test birds')
    conditions=('smooth_memory','binding','shuffled_binding','binding_plus_purpose')
    records=[]
    for number,index in enumerate(selected):
        target=testing[index][0].numpy()
        full=band_pieces(target[None],device=device)[0]
        assert np.max(np.abs(render(full)-target))<2e-6
        # This is the only point full residues appear: produce observations,
        # then pass only *observed* patches to the solver.
        selector=np.random.default_rng(np.random.SeedSequence([args.seed,index,774]))
        visible=seed_masks(selector,64,'half')
        observed=np.zeros_like(full)
        observed[visible]=full[visible]
        donor_ids,dist=nearest_donors(observed,visible,memory,k=args.donors)
        donors=memory[donor_ids]
        target_margin=float(score_cnn(model,target[None],device)[0])
        baseline=render(observed,clip=True)
        start_margin=float(score_cnn(model,baseline[None],device)[0])
        modes={}
        images={}
        for label in conditions:
            graph=bind if label in ('binding','binding_plus_purpose') else (
                bind.permuted(91) if label=='shuffled_binding' else None)
            pg=purpose if label=='binding_plus_purpose' else None
            lam=args.purpose_weight if pg is not None else 0.
            solver=Settler(observed,visible,donors,dist,norm,graph,pg,
                           beta=args.memory_weight,gamma=args.binding_weight,
                           lam=lam,smooth=args.smooth_weight)
            settled=solver.solve(passes=args.passes,seed=args.seed+index)
            reconstructed=render(settled['pieces'],clip=True)
            recon_margin=float(score_cnn(model,reconstructed[None],device)[0])
            mse=float(np.mean((reconstructed-target)**2))
            # Ground truth hidden residues used ONLY for post-hoc evaluation.
            hidden_mse=float(np.mean((settled['pieces'][~visible]-full[~visible])**2))
            modes[label]={
                'pixel_mse':mse,'hidden_residue_mse':hidden_mse,
                'cnn_margin':recon_margin,
                'cnn_margin_gain_over_corrupt':recon_margin-start_margin,
                'margin_error_vs_truth':abs(recon_margin-target_margin),
                'energy_before':settled['energy_before'],'energy_after':settled['energy_after'],
                'monotone':settled['monotone'],'moves':settled['moves'],
                'energy_terms':settled['parts'],
                'energy_trace':settled['trace']}
            images[label]=reconstructed
            if not settled['monotone']:raise AssertionError('Coordinate descent increased its energy')
        receipt={
            'index':index,'test_label':'bird','visible':int(visible.sum()),
            'hidden':int((~visible).sum()),
            'memory_donor_train_ids':[int(memory_idx[i]) for i in donor_ids],
            'teacher_true_margin':target_margin,'teacher_corrupt_margin':start_margin,
            'corrupt_pixel_mse':float(np.mean((baseline-target)**2)),
            'modes':modes}
        records.append(receipt)
        print(f'Bird {index}: '+', '.join(k+' mse='+format(v['pixel_mse'],'.5f')+
              ' margin='+format(v['cnn_margin'],'.3f') for k,v in modes.items()),flush=True)
        if number==0:
            visual={'index':index,'teacher_margin':target_margin,
                    'true':data_image(target),'corrupted':data_image(baseline),
                    'reconstructions':{k:data_image(v) for k,v in images.items()},
                    'modes':modes,'visible_pieces':visible.astype(int).tolist()}
    def aggregate(key):
        return {mode:{'mean_pixel_mse':float(np.mean([r['modes'][mode]['pixel_mse'] for r in records])),
                      'mean_hidden_residue_mse':float(np.mean([r['modes'][mode]['hidden_residue_mse'] for r in records])),
                      'mean_cnn_margin':float(np.mean([r['modes'][mode]['cnn_margin'] for r in records])),
                      'mean_margin_gain':float(np.mean([r['modes'][mode]['cnn_margin_gain_over_corrupt'] for r in records])),
                      'mse_improved_vs_smooth_count':int(sum(
                        r['modes'][mode]['pixel_mse']<r['modes']['smooth_memory']['pixel_mse'] for r in records))}
                for mode in conditions}
    outcome={'protocol':'dodis_v1_2026-10-23',
        'experimental_scope':'test-time reconstruction of 50% missing multiscale residues',
        'note':'Hidden target pieces are only used to score completed reconstructions, never to choose memory candidates.',
        'teacher_checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
        'cnn_checkpoint':meta,'source_repo':str(sihti),
        'split':{'memory_bird_count':na,'binding_bird_count':nb,
                 'purpose_training_count':len(purpose_idx),
                 'purpose_validation_count':len(purpose_val_idx),
                 'test_bird_indices':selected,'disjoint_train_roles':True},
        'binding':{'edges_count':len(bind.edges),
                   'first_edges':bind.edges[:20].tolist(),
                   'training_is_teacher_independent':True},
        'purpose':quality,'params':vars(args),'per_image':records,
        'summary':aggregate('pixel_mse'),
        'gates':{'content_origin_other_train_images':True,
                 'monotone_energy_all_modes':all(
                     r['modes'][m]['monotone'] for r in records for m in conditions),
                 'binding_reduces_mean_mse':aggregate('pixel_mse')['binding']['mean_pixel_mse']<
                     aggregate('pixel_mse')['smooth_memory']['mean_pixel_mse'],
                 'purpose_increases_mean_margin':aggregate('pixel_mse')['binding_plus_purpose']['mean_cnn_margin']>
                     aggregate('pixel_mse')['binding']['mean_cnn_margin']}}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n',encoding='utf8')
    visual.update({'summary':outcome['summary'],'gates':outcome['gates'],
                   'purpose':outcome['purpose'],'n_test':len(records)})
    docs=Path(args.docs);docs.mkdir(parents=True,exist_ok=True)
    (docs/'receipt.json').write_text(json.dumps(visual,separators=(',',':'),allow_nan=False),encoding='utf8')
    print('Saved',out/'results.json',docs/'receipt.json')
    print('Gates',outcome['gates'])
    return outcome

def main():
    parser=argparse.ArgumentParser(description='DoDis three-part completion, no CNN training')
    parser.add_argument('--sihti-root',default='../SihtiGraphModel')
    parser.add_argument('--checkpoint',default='')
    parser.add_argument('--data',default='')
    parser.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    parser.add_argument('--out',default='runs/v1')
    parser.add_argument('--docs',default='docs')
    parser.add_argument('--seed',type=int,default=41)
    parser.add_argument('--memory-size',type=int,default=256)
    parser.add_argument('--binding-size',type=int,default=320)
    parser.add_argument('--purpose-birds',type=int,default=380)
    parser.add_argument('--purpose-others',type=int,default=220)
    parser.add_argument('--purpose-val-birds',type=int,default=70)
    parser.add_argument('--purpose-val-others',type=int,default=70)
    parser.add_argument('--eval-birds',type=int,default=12)
    parser.add_argument('--donors',type=int,default=4)
    parser.add_argument('--passes',type=int,default=3)
    parser.add_argument('--memory-weight',type=float,default=.05)
    parser.add_argument('--smooth-weight',type=float,default=.75)
    parser.add_argument('--binding-weight',type=float,default=.4)
    parser.add_argument('--purpose-weight',type=float,default=.12)
    args=parser.parse_args()
    if args.eval_birds<1 or args.donors<2:parser.error('eval-birds >=1, donors>=2')
    run(args)

if __name__=='__main__':main()
