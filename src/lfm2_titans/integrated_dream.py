"""Model-owned VAE heads and MaleCNS sleep controller for physical units."""
import torch
from dataclasses import replace
from torch import nn
from torch.nn import functional as F
from .dream_memory import DreamWeightVAE,PortFeatureVAE
from .sleep_controller import ReplayController
from .episodic_adapters import AdapterMemory,PhysicalMemoryUnit
from .multiport_connectome import ConnectomeState


class IntegratedDreamMemory(nn.Module):
    def __init__(self,spec,port_dims):
        super().__init__();self.spec=spec
        self.weight_vae=DreamWeightVAE(**spec['weight_vae'])
        self.feature_vae=PortFeatureVAE(port_dims,**spec['feature_vae'])
        self.controller=ReplayController(spec['controller_graph'],kind='malecns',seed=spec['seed'])

    def reconstruct_unit(self,model,unit,*,sample=False):
        """Decode graph and every physical FFN tensor; never mutate the teacher."""
        prior=model.physical_memory.factors(create_graph=False)
        delta,graph_terms=self.weight_vae.reconstruct(unit.graph.fast.detach()-model.memory.slow_weights.detach(),sample=sample)
        graph=ConnectomeState(model.memory.slow_weights.detach()+delta,unit.graph.momentum.detach(),unit.graph.commits)
        factors={};terms=[graph_terms]
        for key,value in unit.adapters.factors.items():
            decoded,term=self.weight_vae.reconstruct(value.detach()-prior[key].detach(),sample=sample)
            factors[key]=prior[key].detach()+decoded;terms.append(term)
        adapters=AdapterMemory(factors,unit.adapters.first_moment,unit.adapters.second_moment,unit.adapters.commits)
        distortion=torch.stack([t['distortion'] for t in terms]).mean()
        kl=torch.stack([t['kl'] for t in terms]).mean()
        return replace(unit,adapters=adapters,graph=graph),dict(distortion=distortion,kl=kl)

    def training_loss(self,graph_delta,features):
        _,terms=self.weight_vae.reconstruct(graph_delta.detach().float(),sample=self.training)
        if not features:raise ValueError('dream training requires actual observed native features')
        feature_results={n:self.feature_vae.loss_terms(n,x,beta=self.spec['kl_weight']) for n,x in features.items()}
        feature_loss=torch.stack([loss for loss,_ in feature_results.values()]).mean()
        # This auxiliary only ranks an observed item above a zero-signal item.
        # It is not supervision for actual consolidation benefit or a learned
        # comparison between compressed and physical recall.
        error=terms['distortion'].detach()
        signals=torch.zeros(2,6,device=error.device);signals[0,0]=error
        signals[0,1]=1.;signals[0,4]=1.
        logits,_=self.controller(signals,self.controller.initial_state())
        schedule=F.cross_entropy(logits[None],torch.zeros(1,device=error.device,dtype=torch.long))
        loss=terms['distortion']+self.spec['kl_weight']*terms['kl']+feature_loss+.01*schedule
        return loss,dict(**terms,feature_vae=feature_loss,scheduler=schedule,
            feature_terms={n:{key:value.detach() for key,value in item.items()} for n,(_,item) in feature_results.items()})

    def weight_replay_loss(self,deltas):
        """Train the weight codec on real FFN updates from the preceding batch."""
        terms=[self.weight_vae.reconstruct(value.detach().float(),sample=self.training)[1] for value in deltas]
        if not terms:return next(self.parameters()).new_zeros(())
        return torch.stack([t['distortion']+self.spec['kl_weight']*t['kl'] for t in terms]).mean()

    @torch.no_grad()
    def replay_latents(self,latents,*,source_id,temperature=.3,min_cosine=.8):
        """Dream from persisted posterior codes after native features are gone.

        The gate measures codec cycle consistency, not historical truth.
        Publication still requires the session's separate functional verifier.
        """
        if self.training or not source_id:raise ValueError('frozen codec and source identity required')
        if not 0<=temperature<=1 or not -1<=min_cosine<=1:raise ValueError('invalid dream replay settings')
        accepted={};receipts={}
        for name,port in latents.items():
            head=self.feature_vae.heads[name]
            z=port.mu+temperature*torch.randn_like(port.mu)*(.5*port.logvar).exp()
            generated=head.decoder(z)
            reconstructed,_=head.posterior(head.encoder(generated)).chunk(2,-1)
            similarity=F.cosine_similarity(reconstructed,port.mu,dim=-1)
            valid=torch.isfinite(generated).all(-1)&(similarity>=min_cosine)
            if valid.any():accepted[name]=(generated*port.scale+port.mean)[valid]
            receipts[name]=dict(proposed=port.count,accepted=int(valid.sum()),source_id=source_id,
                min_cosine=min_cosine,source='persisted_vae_posterior',criterion='latent_cycle_consistency_only')
        return accepted,receipts

    @torch.no_grad()
    def replay_features(self,features,*,source_id,temperature=.3,min_cosine=.8):
        """Source-conditioned dreams retain each native feature's scale."""
        if self.training or not source_id:raise ValueError('frozen dream module and source identity required')
        if not 0<=temperature<=1 or not -1<=min_cosine<=1:raise ValueError('invalid dream replay settings')
        accepted={};receipts={}
        for name,value in features.items():
            x=value.detach().float().reshape(-1,value.shape[-1]);mean=x.mean(-1,keepdim=True)
            scale=(x.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt();normalized=(x-mean)/scale
            head=self.feature_vae.heads[name]
            _,mu,logvar=head(normalized,sample=False)
            generated=head.decoder(mu+temperature*torch.randn_like(mu)*(.5*logvar).exp())
            similarity=F.cosine_similarity(generated,normalized,dim=-1)
            valid=torch.isfinite(generated).all(-1)&(similarity>=min_cosine)
            if valid.any():accepted[name]=(generated*scale+mean)[valid]
            receipts[name]=dict(proposed=len(x),accepted=int(valid.sum()),source_id=source_id,min_cosine=min_cosine)
        return accepted,receipts
