import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from prepare_counterfactual_hashhop import examples,audit


def test_queries_require_hops_and_memory_contents_and_holdout_is_disjoint():
    rows,graphs,symbols=examples(917203,8)
    validation,vg,vs=examples(617307,4,symbols)
    result=audit(rows,graphs)
    assert not symbols&vs
    assert result['counterfactual_query_pairs']==len(rows)//2
    assert 0<result['ignore_hop_count_solver_correct']<len(rows)//4
    assert 0<result['interior_start_records']<len(rows)
    by_question=defaultdict(list)
    for row in rows:by_question[row['messages'][2]['content']].append(row)
    for pair in by_question.values():
        assert len(pair)==2
        assert pair[0]['messages'][0]['content']!=pair[1]['messages'][0]['content']
        assert pair[0]['messages'][-1]['content']!=pair[1]['messages'][-1]['content']
    audit(validation,vg)
