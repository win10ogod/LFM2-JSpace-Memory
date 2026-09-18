import torch
from lfm2_titans.dream_memory import PortFeatureVAE


def test_codec_objective_preserves_repeated_identity_and_trains_both_heads():
    torch.manual_seed(19)
    codec=PortFeatureVAE({'native_input':16},hidden_size=8,latent_size=2).train()
    features=torch.randn(12,16,requires_grad=True)
    actual,terms=codec.loss_terms('native_input',features)
    expected=terms['distortion']+terms['mean_distortion']+.25*terms['discrimination']+.001*terms['kl']
    actual.backward()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters())
    assert features.grad is None
    assert all(torch.isfinite(x) for x in terms.values())
    codec.eval()
    with torch.no_grad():
        _,deterministic=codec.loss_terms('native_input',features)
        _,repeated=codec.loss_terms('native_input',features[:1].expand(12,-1))
    torch.testing.assert_close(deterministic['distortion'],deterministic['mean_distortion'])
    assert repeated['discrimination']==0


def test_categorical_codec_covers_every_position_and_keeps_dictionary_frozen():
    from lfm2_titans.latent_memory import encode_observations
    from torch.nn import functional as F
    torch.manual_seed(17)
    embedding=torch.nn.Embedding(37,16)
    codec=PortFeatureVAE({'native_input':16},hidden_size=8,latent_size=2).eval()
    ids=torch.tensor([1,3,5,7,1,9,11])
    code=encode_observations(codec,{'native_input':embedding(ids)},create_graph=True)['native_input']
    restored=codec.heads['native_input'].decoder(code.mu)*code.scale+code.mean
    reference=F.cross_entropy(F.normalize(restored,dim=-1)@F.normalize(embedding.weight.detach(),dim=-1).T/.02,ids)
    params=tuple(codec.parameters())
    expected=torch.autograd.grad(reference,params,retain_graph=True)
    actual=codec.token_reconstruction_loss('native_input',code,ids,embedding.weight,chunk_size=3)
    gradients=torch.autograd.grad(actual,params+(embedding.weight,),allow_unused=True)
    torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-6)
    for a,b in zip(gradients[:-1],expected):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    assert gradients[-1] is None
    codec.train();code=encode_observations(codec,{'native_input':embedding(ids)},create_graph=True)['native_input']
    codec.token_reconstruction_loss('native_input',code,ids,embedding.weight,chunk_size=3).backward()
    assert torch.isfinite(codec.heads['native_input'].posterior.weight.grad).all()
    assert codec.heads['native_input'].posterior.weight.grad[2:].abs().sum()>0
    assert embedding.weight.grad is None
