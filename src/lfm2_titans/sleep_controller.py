"""Small stateful replay schedulers; memory contents stay in physical pages.

The MaleCNS circuit is an engineering prior, not a simulation of animal sleep.
Every variant uses the same observable signals, state size and replay budget.
No controller sees withheld answers, deletes history or writes into foreground
causal snapshots. State belongs to the caller, never a global singleton.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SIGNALS=('distortion','novelty','replayed_fraction','sensory_load','age','interference')


def degree_rewire(src,dst,*,seed,swaps_per_edge=10):
    """Directed endpoint swaps preserve every in/out degree and source weight."""
    src=np.array(src,dtype=np.int64,copy=True);dst=np.array(dst,dtype=np.int64,copy=True)
    before=dst.copy();pairs=set(zip(src.tolist(),dst.tolist()));rng=np.random.default_rng(seed)
    accepted=0
    for _ in range(len(src)*swaps_per_edge):
        a,b=rng.integers(len(src),size=2)
        u,v=int(src[a]),int(src[b]);x,y=int(dst[a]),int(dst[b])
        if a==b or u==v or x==y or u==x or v==y or u==y or v==x or (u,y) in pairs or (v,x) in pairs:continue
        pairs.remove((u,x));pairs.remove((v,y));pairs.add((u,y));pairs.add((v,x))
        dst[a],dst[b]=y,x;accepted+=1
    return src,dst,dict(accepted_swaps=accepted,changed_destinations=int((before!=dst).sum()),
                        directed_degrees_preserved=True)


@dataclass(frozen=True)
class SleepState:
    activity: torch.Tensor
    debt: float=0.
    ticks: int=0

    def detach(self):return SleepState(self.activity.detach(),self.debt,self.ticks)


class ReplayController(nn.Module):
    """Persistent neural activity plus source-specific, transient ranking reads.

    Feature-to-cell assignments and initial signs are declared engineering
    choices. Connectivity and anatomical identities come from the graph file.
    A bounded queue supplies source features; the controller holds no catalogue.
    """
    def __init__(self,graph,*,kind='malecns',seed=1):
        super().__init__()
        if kind not in ('malecns','rewired','mlp'):raise ValueError('unknown controller')
        self.kind=kind
        types=list(graph['types']);n=len(types)
        src,dst=np.array(graph['src']),np.array(graph['dst'])
        self.rewire_receipt=None
        if kind=='rewired':src,dst,self.rewire_receipt=degree_rewire(src,dst,seed=seed)
        self.register_buffer('src',torch.tensor(src,dtype=torch.long))
        self.register_buffer('dst',torch.tensor(dst,dtype=torch.long))
        weights=torch.log1p(torch.tensor(graph['weight'],dtype=torch.float32))
        total=torch.zeros(n,dtype=torch.float32).index_add_(0,self.dst,weights).clamp_min(1)
        # Only two explicitly declared inhibitory initial priors; learned
        # edge gains can change sign. Others have no transmitter inference.
        sign=torch.tensor([-1. if types[i] in ('DPM','MBON03') else 1. for i in src],dtype=torch.float32)
        initial=.5*weights/total[self.dst]*sign
        self.register_buffer('edge_scale',initial)
        drives=torch.zeros(len(SIGNALS),n,dtype=torch.float32)
        targets=[('MBON12',),('ER5',),('MBON03',),('ExR1',),('FB6A_a','FB6A_b','FB6A_c'),('DPM',)]
        for i,names in enumerate(targets):
            selected=torch.tensor([t in names for t in types])
            if not selected.any():raise ValueError(f'missing signal target {names}')
            drives[i,selected]=1.
        self.register_buffer('drive_mask',drives)
        self.nodes=n
        self.input_gain=nn.Parameter(torch.ones(len(SIGNALS),dtype=torch.float32))
        self.leak_logit=nn.Parameter(torch.tensor(-1.4,dtype=torch.float32))
        self.readout=nn.Parameter(torch.zeros(n,dtype=torch.float32))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            nn.init.normal_(self.readout,std=.02)
            if kind=='mlp':
                # Budget-near MLP replaces E gains; state dimension is unchanged.
                hidden=max(1,len(src)//(2*n+1))
                self.recurrent=nn.Sequential(nn.Linear(n,hidden,bias=False),nn.Tanh(),nn.Linear(hidden,n,bias=False))
            else:self.edge_gain=nn.Parameter(torch.ones(len(src),dtype=torch.float32))

    def initial_state(self):return SleepState(self.readout.new_zeros(self.nodes))

    def _tick(self,activity,drive):
        if self.kind=='mlp':recurrent=self.recurrent(activity)
        else:
            recurrent=torch.zeros_like(activity).index_add(-1,self.dst,
                activity[...,self.src]*self.edge_scale*self.edge_gain)
        rate=self.leak_logit.sigmoid()
        return (1-rate)*activity+rate*torch.tanh(drive+recurrent)

    def forward(self,signals,state):
        if signals.ndim!=2 or signals.shape[1]!=len(SIGNALS) or not torch.isfinite(signals).all():
            raise ValueError('finite [candidate,signal] matrix required')
        if state.activity.shape!=(self.nodes,):raise ValueError('controller state shape mismatch')
        # Bounded monotonic transform, not a cap on the number of memories.
        x=torch.sign(signals)*torch.log1p(signals.abs())
        drive=(x*self.input_gain)@self.drive_mask
        candidate=state.activity.unsqueeze(0).expand(len(x),-1)
        for _ in range(3):candidate=self._tick(candidate,drive)
        logits=candidate@self.readout/math.sqrt(self.nodes)
        # One persistent state update after ranking the fixed snapshot.
        activity=self._tick(state.activity,drive.mean(0))
        return logits,SleepState(activity,state.debt,state.ticks+1)

    def observe(self,signals,state,*,new_information):
        if not math.isfinite(new_information) or new_information<0:raise ValueError('invalid new-information signal')
        _,updated=self(signals,state)
        return SleepState(updated.activity,state.debt+new_information,updated.ticks).detach()

    def ready(self,state,*,threshold=1.):
        if threshold<=0:raise ValueError('threshold must be positive')
        return state.debt>=threshold

    @staticmethod
    def complete(state,*,consolidated_work):
        if not math.isfinite(consolidated_work) or consolidated_work<0:raise ValueError('invalid consolidation work')
        return SleepState(state.activity,max(0.,state.debt-consolidated_work),state.ticks)

    def architecture(self):
        return dict(kind=self.kind,nodes=self.nodes,edges=len(self.src),signals=SIGNALS,
            parameters=sum(p.numel() for p in self.parameters()),persistent_activity_scalars=self.nodes,
            microsteps=3,rewire=self.rewire_receipt,
            biological_scope='measured topology prior with engineered signal mapping and trainable dynamics')
