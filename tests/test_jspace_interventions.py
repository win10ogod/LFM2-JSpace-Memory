from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
from test_jspace_functions import swap_delta,reference_swap_delta,suite


def test_coordinate_swap_preserves_orthogonal_computation():
    torch.manual_seed(51)
    d=torch.nn.functional.normalize(torch.randn(2,16),dim=-1);x=torch.randn(4,16)
    before=x@torch.linalg.pinv(d);after=x+swap_delta(x,d)
    torch.testing.assert_close(after@torch.linalg.pinv(d),before.flip(-1))
    projection=torch.linalg.pinv(d)@d
    torch.testing.assert_close(after-after@projection,x-x@projection)
    torch.testing.assert_close(after+swap_delta(after,d),x)


def test_functional_suite_pairs_have_distinct_answers_and_blind_intermediates():
    protocol=suite()
    assert len(protocol['items'])==16
    for p in protocol['items']:
        assert p['answer']!=p['counterfactual']
        assert p['original'].lower() not in p['prompt'].lower()
        assert p['target'].lower() not in p['prompt'].lower()


def test_reference_swap_is_idempotent_and_preserves_orthogonal_content():
    torch.manual_seed(81)
    d=torch.nn.functional.normalize(torch.randn(2,16),dim=-1)
    reference=torch.randn(5,16);hidden=reference+torch.randn(5,16)*.1
    edited=hidden+reference_swap_delta(hidden,d,reference)
    torch.testing.assert_close(edited+reference_swap_delta(edited,d,reference),edited)
    torch.testing.assert_close(edited@torch.linalg.pinv(d),(reference@torch.linalg.pinv(d)).flip(-1))
    projection=torch.linalg.pinv(d)@d
    torch.testing.assert_close(edited-edited@projection,hidden-hidden@projection)
