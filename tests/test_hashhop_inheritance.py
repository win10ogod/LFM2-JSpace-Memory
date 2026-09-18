import sys
from pathlib import Path
import torch
from torch.nn.utils import parametrize

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from prepare_hashhop_model import capture_parameters,verify_inherited_parameters
from lfm2_titans.sft_lora import TensorLoRA


def test_registration_version_change_preserves_original_and_effective_weights():
    layer=torch.nn.Linear(5,3,bias=False)
    weight=layer.weight.detach().clone();snapshot=capture_parameters(layer)
    parametrize.register_parametrization(layer,'weight',TensorLoRA(layer.weight,2,4))
    result=verify_inherited_parameters(layer,snapshot)
    assert result['status']=='passed'
    assert result['version_changes_without_value_changes']==1
    torch.testing.assert_close(layer.weight,weight,rtol=0,atol=0)


def test_numerical_mutations_and_nonzero_adapters_are_rejected():
    layer=torch.nn.Linear(5,3,bias=False);snapshot=capture_parameters(layer)
    parametrize.register_parametrization(layer,'weight',TensorLoRA(layer.weight,2,4))
    original=layer.parametrizations.weight.original
    version=original._version
    original.data.add_(1)  # A _version-only test misses this real modification.
    assert original._version==version
    result=verify_inherited_parameters(layer,snapshot)
    assert result['status']=='failed' and result['changed_parameters']==['weight']
    snapshot=capture_parameters(layer)
    layer.parametrizations.weight[0].adapter_B.data.fill_(1)
    result=verify_inherited_parameters(layer,snapshot)
    assert result['status']=='failed' and result['nonzero_adapter_outputs']
