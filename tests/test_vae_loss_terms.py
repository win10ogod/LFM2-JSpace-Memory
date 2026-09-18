from copy import deepcopy
import torch
from torch.nn import functional as F
from lfm2_titans.dream_memory import PortFeatureVAE


def test_codec_metrics_preserve_sampled_objective_gradients_and_rng():
    torch.manual_seed(19)
    codec=PortFeatureVAE({'native_input':16},hidden_size=8,latent_size=2).train()
    reference=deepcopy(codec);features=torch.randn(12,16,requires_grad=True)
    normalized=F.layer_norm(features.detach(),(16,))
    torch.manual_seed(23)
    decoded,mu,logvar=reference.heads['native_input'](normalized,sample=True)
    expected=F.mse_loss(decoded,normalized)+.001*.5*(mu.square()+logvar.exp()-1-logvar).mean()
    expected.backward();rng=torch.get_rng_state()
    torch.manual_seed(23)
    actual,terms=codec.loss_terms('native_input',features)
    actual.backward()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),rng)
    for a,b in zip(codec.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    assert features.grad is None
    assert all(torch.isfinite(x) for x in terms.values())
    assert not terms['mean_distortion'].requires_grad
    codec.eval()
    with torch.no_grad():
        _,deterministic=codec.loss_terms('native_input',features)
    torch.testing.assert_close(deterministic['distortion'],deterministic['mean_distortion'])
