"""DoDis v2: translation-tolerant, visible-context local Sihti memory.

Crucial rule: a missing target residue is NEVER used to form its address.
The query is built only from visible nearby pieces and visible cross-scale
pieces, including genuinely observed coarse-state context. Donor patches are
real residues from other CIFAR training images, indexed at all 16 locations.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from dodis import features


def context_for_piece(j: int, visible: np.ndarray):
    """Observed pieces near j: 8 local neighbors + same-tile other bands
    + neighboring visible coarser band. Returns (band, dr, dc, weight)."""
    b, tile = divmod(int(j),16)
    r,c=divmod(tile,4)
    result=[]
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr==dc==0 or not 0<=r+dr<4 or not 0<=c+dc<4:continue
            k=b*16+(r+dr)*4+c+dc
            if visible[k]:
                result.append((b,dr,dc,1. if abs(dr)+abs(dc)==1 else .5))
    for other in range(4):
        if other==b:continue
        k=other*16+tile
        if visible[k]:
            result.append((other,0,0,1.7 if other>b else 1.))
    if b<3:
        other=b+1
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            if not 0<=r+dr<4 or not 0<=c+dc<4:continue
            k=other*16+(r+dr)*4+c+dc
            if visible[k]:
                result.append((other,dr,dc,.5))
    return result


@dataclass
class LocalRetrieval:
    candidates: np.ndarray  # [K,64,3,8,8] each K slot can come from different image/tile per piece
    distances: np.ndarray   # [K,64] piece-specific context distances
    provenance: list        # [64][K] train index, donor tile and confidence
    observed_context_counts: list


class LocalPatchMemory:
    """Index every tile of each training image; use band-standardized features.

    Coordinates are compared relative to the missing target piece. Source
    patches may come from another absolute tile position; this makes the
    address independent of global bird placement.
    """
    def __init__(self, training_pieces:np.ndarray,train_ids=None):
        bank=np.asarray(training_pieces,dtype=np.float32)
        if bank.ndim!=5 or bank.shape[1:]!=(64,3,8,8) or len(bank)<2:
            raise ValueError('Training memory must be [N,64,3,8,8] with N>=2')
        self.bank=bank
        self.train_ids=np.arange(len(bank)) if train_ids is None else np.asarray(train_ids)
        if len(self.train_ids)!=len(bank):raise ValueError('train ID count mismatch')
        descriptors=features(bank).reshape(len(bank),4,16,6)
        self.center=np.mean(descriptors,axis=(0,2)).astype(np.float32)
        self.std=np.maximum(np.std(descriptors,axis=(0,2)),.01).astype(np.float32)
        self.bank_u=np.clip((descriptors-self.center[None,:,None,:])/
                            self.std[None,:,None,:],-4.,4.).astype(np.float32)
        self.images=np.repeat(np.arange(len(bank),dtype=np.int32),16)
        self.tiles=np.tile(np.arange(16,dtype=np.int32),len(bank))
        self.rows=self.tiles//4;self.cols=self.tiles%4
        self.prior_rms=np.sqrt(np.mean(bank*bank,axis=(2,3,4))).reshape(len(bank),4,16)

    def retrieve(self, observed:np.ndarray,visible:np.ndarray,k:int=4):
        observed=np.asarray(observed,dtype=np.float32)
        visible=np.asarray(visible,dtype=bool)
        if observed.shape!=(64,3,8,8) or visible.shape!=(64,):
            raise ValueError('Expected observed [64,3,8,8] and visible [64]')
        if not visible.any() or int(k)<1 or int(k)>len(self.bank):
            raise ValueError('Need visible evidence and 1..N donors')
        if np.any(observed[~visible]!=0):
            raise ValueError('Missing residues must be ZERO, not hidden ground truth')
        tf=features(observed).reshape(4,16,6)
        query=np.clip((tf-self.center[:,None,:])/self.std[:,None,:],-4,4)
        candidates=np.zeros((k,64,3,8,8),dtype=np.float32)
        distances=np.zeros((k,64),dtype=np.float32)
        provenance=[[] for _ in range(64)]
        counts=[0]*64
        for j in range(64):
            band,tile=divmod(j,16);r,c=divmod(tile,4)
            context=context_for_piece(j,visible)
            counts[j]=len(context)
            if visible[j]:
                # No visible target piece is ever substituted by this memory.
                candidates[:,j]=observed[j]
                continue
            score=np.zeros(len(self.images),dtype=np.float64)
            total=0.
            for other,dr,dc,weight in context:
                src_r=self.rows+dr
                src_c=self.cols+dc
                valid=(src_r>=0)&(src_r<4)&(src_c>=0)&(src_c<4)
                src_tile=(src_r[valid]*4+src_c[valid]).astype(np.int32)
                target_tile=(r+dr)*4+c+dc
                diff=(self.bank_u[self.images[valid],other,src_tile]-
                      query[other,target_tile])
                score[valid]+=weight*np.mean(diff*diff,axis=1)
                # Penalize donors missing a neighbor that the target actually sees.
                score[~valid]+=weight*2.
                total+=weight
            if total>0:
                score/=total
            else:
                # Without visible neighbors, no justified structural address exists.
                # A weak energy-only default is explicitly recorded as zero evidence.
                score.fill(1.)
            # Small prior against injecting high-amplitude unsupported residues.
            score+=.02*self.prior_rms[self.images,band,self.tiles]
            # Distinct training photographs avoid top-K near-duplicate tiles from one image.
            order=np.argsort(score,kind='stable')
            chosen=[];used=set()
            for flat in order:
                im=int(self.images[flat])
                if im in used:continue
                used.add(im);chosen.append(int(flat))
                if len(chosen)==k:break
            for slot,flat in enumerate(chosen):
                im=int(self.images[flat]);source_tile=int(self.tiles[flat])
                src_index=band*16+source_tile
                candidates[slot,j]=self.bank[im,src_index]
                distances[slot,j]=float(score[flat])
                provenance[j].append({'train_image_id':int(self.train_ids[im]),
                                      'source_tile':source_tile,
                                      'target_tile':tile,'band':band,
                                      'distance':float(score[flat]),
                                      'visible_context_count':len(context)})
        return LocalRetrieval(candidates,distances,provenance,counts)
