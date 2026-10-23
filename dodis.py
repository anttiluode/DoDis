"""DoDis: candidate content, learned consistency, optional CNN-purpose graph.

No hidden target pieces are available to the solver. All candidate residues come
from other training images. All observed residues are held fixed. Discrete
coordinate descent enumerates every candidate and never increases its energy.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MEAN = np.array([.4914,.4822,.4465],dtype=np.float32)

def render(parts:np.ndarray,clip:bool=False)->np.ndarray:
    """Exact additive inverse of 4 bands x 4 x 4 tiles of 3x8x8."""
    p=np.asarray(parts,dtype=np.float32)
    if p.shape!=(64,3,8,8):raise ValueError(f'Expected 64x3x8x8, got {p.shape}')
    x=MEAN[:,None,None]+p.reshape(4,4,4,3,8,8).sum(axis=0).transpose(2,0,3,1,4).reshape(3,32,32)
    return np.clip(x,0,1) if clip else x

def band_pieces(images,device='cpu',batch=64)->np.ndarray:
    """Uses SihtiGraphModel's exact torch area/bilinear Laplacian-like sieve."""
    import torch
    import torch.nn.functional as F
    if not len(images):return np.empty((0,64,3,8,8),np.float32)
    out=[]
    for start in range(0,len(images),batch):
        img=torch.as_tensor(np.asarray(images[start:start+batch]),dtype=torch.float32,device=device)
        if img.ndim!=4 or img.shape[1:]!=(3,32,32):
            raise ValueError('Images must be [N,3,32,32] in [0,1]')
        gs=[img]
        for factor in (2,4,8):
            coarse=F.avg_pool2d(img,factor)
            gs.append(F.interpolate(coarse,size=(32,32),mode='bilinear',align_corners=False))
        bands=torch.stack([gs[i]-gs[i+1] for i in range(3)]+[gs[3]],dim=1)
        bands[:,3]-=torch.tensor(MEAN,device=device).view(1,3,1,1)
        # [N,4,3,32,32] -> [N,4,4,4,3,8,8] -> [N,64,3,8,8]
        patches=bands.reshape(len(img),4,3,4,8,4,8).permute(0,1,3,5,2,4,6)
        out.append(patches.reshape(len(img),64,3,8,8).cpu().numpy().copy())
    return np.concatenate(out)

def features(parts:np.ndarray)->np.ndarray:
    """Six per-piece observations (RGB averages, x/y contrast, RMS)."""
    a=np.asarray(parts,dtype=np.float32)
    if a.shape[-4:]!=(64,3,8,8):raise ValueError('Expected [...,64,3,8,8]')
    color=a.mean(axis=(-1,-2))
    lum=(a[...,0,:,:]+a[...,1,:,:]+a[...,2,:,:])/3
    dx=np.mean(np.abs(np.diff(lum,axis=-1)),axis=(-1,-2))
    dy=np.mean(np.abs(np.diff(lum,axis=-2)),axis=(-1,-2))
    rms=np.sqrt(np.mean(a*a,axis=(-1,-2,-3))+1e-8)
    return np.concatenate([color,dx[...,None],dy[...,None],rms[...,None]],axis=-1)

@dataclass
class Normalizer:
    center:np.ndarray
    scale:np.ndarray
    @classmethod
    def learn(cls,train_features:np.ndarray):
        center=np.mean(train_features,axis=0)
        std=np.std(train_features,axis=0)
        return cls(center,np.maximum(std,.01))
    def apply(self,f:np.ndarray):
        return np.clip((np.asarray(f)-self.center)/self.scale,-3.,3.)

def candidate_edges(n=4):
    """Spatial 4-neighbours + adjacent octave same tile + far-octave same tile."""
    edges=set()
    for band in range(n):
        for r in range(4):
            for c in range(4):
                i=band*16+r*4+c
                if r<3:edges.add((i,i+4))
                if c<3:edges.add((i,i+1))
    for band in range(n-1):
        for tile in range(16):edges.add((band*16+tile,(band+1)*16+tile))
    for tile in range(16):edges.add((tile,48+tile))
    return sorted(edges)

@dataclass
class BindingGraph:
    edges:np.ndarray              # [E,2]
    matrices:np.ndarray           # [E,6,6], from images not teacher
    strengths:np.ndarray
    @classmethod
    def learn(cls,u:np.ndarray,max_edges=80):
        if u.ndim!=3 or u.shape[1:]!=(64,6):raise ValueError('Expected [images,64,6]')
        if len(u)<10:raise ValueError('Need multiple distinct training images')
        candidates=[]
        # Paired vs permuted controls ensure correlations originate in co-occurrence.
        shifted=np.roll(u,len(u)//2,axis=0)
        for i,j in candidate_edges():
            joint=(u[:,i].T@u[:,j])/len(u)
            shuffled=(u[:,i].T@shifted[:,j])/len(u)
            mat=joint-shuffled
            strength=float(np.linalg.norm(mat))
            candidates.append((strength,i,j,mat))
        candidates.sort(key=lambda a:(-a[0],a[1],a[2]))
        selected=candidates[:max_edges]
        edges=np.asarray([[i,j] for _,i,j,_ in selected],dtype=np.int32)
        mats=np.asarray([m/(np.linalg.norm(m)+1e-8) for _,_,_,m in selected],dtype=np.float32)
        return cls(edges,mats,np.array([s for s,_,_,_ in selected],np.float32))
    def score(self,u:np.ndarray)->float:
        if not len(self.edges):return 0.
        pairs=u[self.edges]
        terms=np.einsum('ed,edf,ef->e',pairs[:,0],self.matrices,pairs[:,1],optimize=True)
        return float(np.mean(np.tanh(terms/6.0)))
    def permuted(self,seed=91):
        rng=np.random.default_rng(seed)
        permutation=rng.permutation(len(self.matrices))
        return BindingGraph(self.edges.copy(),self.matrices[permutation].copy(),self.strengths.copy())

def purpose_design(u:np.ndarray,edges:np.ndarray)->np.ndarray:
    """Visible, interpretable unary observations + bound scalar interactions."""
    a=np.asarray(u,dtype=np.float64)
    if a.ndim!=3 or a.shape[1:]!=(64,6):raise ValueError('Expected [batch,64,6]')
    unary=a.reshape(len(a),-1)
    scalar=a[:,:,5]
    pair=scalar[:,edges[:,0]]*scalar[:,edges[:,1]]
    return np.concatenate([np.ones((len(a),1)),unary,pair],axis=1)

@dataclass
class PurposeGraph:
    weights:np.ndarray
    edges:np.ndarray
    val_r2:float
    teacher_sd:float
    def predict(self,u:np.ndarray)->float:
        if u.ndim==2:u=u[None]
        return float((purpose_design(u,self.edges)@self.weights)[0])
    @classmethod
    def fit(cls,ufit,yfit,uval,yval,edges,ridge=50.):
        a=purpose_design(ufit,edges)
        b=purpose_design(uval,edges)
        penalty=np.eye(a.shape[1])*ridge
        penalty[0,0]=0.
        w=np.linalg.solve(a.T@a+penalty,a.T@np.asarray(yfit,dtype=np.float64))
        yp=b@w
        v=np.asarray(yval)
        r2=1.-np.sum((v-yp)**2)/(np.sum((v-v.mean())**2)+1e-12)
        return cls(w,edges.copy(),float(r2),max(float(np.std(yfit)),1e-6))

def nearest_donors(observed:np.ndarray,visible:np.ndarray,
                   memory:np.ndarray,k=4)->tuple[np.ndarray,np.ndarray]:
    """Distance computed ONLY on visible piece data; hidden truth is absent."""
    vis=np.where(visible)[0]
    if len(vis)==0:raise ValueError('At least one visible piece required')
    scales=np.mean(memory[:,vis]**2,axis=(0,2,3,4))+1e-4
    error=np.mean((memory[:,vis]-observed[vis][None])**2,axis=(2,3,4))
    distance=np.mean(error/scales[None],axis=1)
    idx=np.argsort(distance,kind='stable')[:min(k,len(distance))]
    return idx,distance[idx]

def stitch_choices(observed:np.ndarray,visible:np.ndarray,candidates:np.ndarray,choice:np.ndarray):
    out=np.array(observed,copy=True)
    hidden=np.flatnonzero(~visible)
    out[hidden]=candidates[choice[hidden],hidden]
    return out

def seams(image:np.ndarray)->float:
    """Squared discontinuity ONLY across 8px tile boundaries, fixed normalization."""
    a=image
    cols=np.mean((a[:,:,7:31:8]-a[:,:,8:32:8])**2)
    rows=np.mean((a[:,7:31:8,:]-a[:,8:32:8,:])**2)
    return float((cols+rows)*.5)

class Settler:
    def __init__(self, observed, visible, donor_pieces, donor_distance,
                 normalizer:Normalizer, binding:BindingGraph|None=None,
                 purpose:PurposeGraph|None=None, beta=.05,gamma=.40,lam=0.,
                 smooth=.75,allow_blank=True):
        self.visible=np.asarray(visible,dtype=bool)
        self.observed=np.asarray(observed,dtype=np.float32).copy()
        if self.observed.shape!=(64,3,8,8):raise ValueError('bad observed patches')
        if not np.array_equal(self.observed[~self.visible],np.zeros_like(self.observed[~self.visible])):
            raise ValueError('Hidden observed pieces must be zero, not ground truth')
        self.candidates=np.asarray(donor_pieces,dtype=np.float32)
        distances=np.asarray(donor_distance,dtype=np.float32)
        if self.candidates.shape[1:]!=(64,3,8,8) or len(distances)!=len(self.candidates):
            raise ValueError('candidate shape mismatch')
        if distances.ndim not in (1,2) or (distances.ndim==2 and distances.shape[1]!=64):
            raise ValueError('donor distances must be [K] or per-piece [K,64]')
        # One candidate per donor slot; in v2 each slot can hold a DIFFERENT
        # training patch for each missing piece. Blank is an explicit candidate.
        if allow_blank:
            self.candidates=np.concatenate([self.candidates,np.zeros_like(self.candidates[:1])])
            blank=(distances.max(axis=0)+1.) if distances.ndim==2 else distances.max()+1.
            distances=np.concatenate((distances,np.asarray(blank,dtype=np.float32)[None]),axis=0) if distances.ndim==2 else np.append(distances,blank)
        if distances.ndim==2:
            self.penalty=(distances-distances.min(axis=0,keepdims=True))/(np.std(distances,axis=0,keepdims=True)+1e-3)
        else:
            self.penalty=(distances-distances.min())/(np.std(distances)+1e-3)
        self.normalizer=normalizer
        self.bind=binding
        self.purpose=purpose
        self.beta=beta;self.gamma=gamma;self.lam=lam;self.smooth=smooth
        self.hidden=np.flatnonzero(~self.visible)
        self.cache_u=normalizer.apply(features(self.candidates))
        self.fixed_u=normalizer.apply(features(self.observed))
    def unpack(self,choice):
        return stitch_choices(self.observed,self.visible,self.candidates,choice)
    def cost(self,choice):
        p=self.unpack(choice)
        image=render(p)
        penalties=(self.penalty[choice[self.hidden],self.hidden] if self.penalty.ndim==2
                   else self.penalty[choice[self.hidden]])
        unary=self.beta*float(np.mean(penalties))
        edge=self.smooth*seams(image)*20.
        u=self.fixed_u.copy()
        u[self.hidden]=self.cache_u[choice[self.hidden],self.hidden]
        bind=-self.gamma*self.bind.score(u) if self.bind is not None else 0.
        purpose=-(self.lam/self.purpose.teacher_sd)*self.purpose.predict(u) if self.purpose is not None else 0.
        return unary+edge+bind+purpose,{'memory':unary,'smoothness':edge,'binding':bind,'purpose':purpose}
    def solve(self,passes=3,seed=7):
        rng=np.random.default_rng(seed)
        choice=np.zeros(64,dtype=np.int32)
        cost,parts=self.cost(choice)
        before=cost
        log=[cost];moves=0
        for _ in range(passes):
            improved=0
            for j in rng.permutation(self.hidden):
                best=choice[j];best_cost=cost;best_parts=parts
                for c in range(len(self.candidates)):
                    if c==choice[j]:continue
                    choice[j]=c
                    proposed,decomp=self.cost(choice)
                    if proposed<best_cost-1e-9:
                        best=c;best_cost=proposed;best_parts=decomp
                choice[j]=best
                if best_cost<cost-1e-9:
                    moves+=1;improved+=1
                cost=best_cost;parts=best_parts;log.append(cost)
            if not improved:break
        return {'choice':choice,'pieces':self.unpack(choice),'energy_before':before,
                'energy_after':cost,'parts':parts,'moves':moves,
                'trace':log,'monotone':bool(np.all(np.diff(log)<=1e-8))}
