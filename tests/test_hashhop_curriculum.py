import sys
from pathlib import Path
import random
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from hashhop_curriculum import training_examples,evaluation_cases,validate_case,training_token_lengths
from evaluate_memory_generation import hash_score,coherence_diagnostics


def test_shuffled_chains_are_valid_and_eval_symbols_are_disjoint():
    rows,graphs,symbols=training_examples()
    sources,questions,evaluation,heldout=evaluation_cases(forbidden=symbols)
    assert len(rows)==256 and not symbols&heldout
    assert all([m['role'] for m in row['messages']]==['user','assistant','user','assistant'] for row in rows)
    assert all('COMPLETION' not in row['messages'][0]['content'] and 'Using' not in row['messages'][0]['content'] for row in rows)
    assert {r['hops'] for r in rows}=={1,2,4,8}
    mapping={g['id']:g for g in graphs}
    for row in rows:
        g=mapping[row['graph_id']];edges=list(g['edges']);random.Random(81).shuffle(edges)
        validate_case(edges,g['chain'][0],row['hops'],row['task'],row['messages'][-1]['content'])
    source_ids={s['id'] for s in sources}
    assert all(set(q['required_sources'])<=source_ids for q in questions)
    assert any(q.get('hops')==10 for q in questions)
    assert all(len(q['required_sources'])>1 for q in questions if q['id'].startswith('cross'))
    assert {r['hash_length'] for r in rows}=={8,16}
    for g in graphs+evaluation:
        table=dict(g['edges']);roots=set(table)-set(table.values());lengths=[]
        for root in roots:
            n=0
            while root in table:root=table[root];n+=1
            lengths.append(n)
        assert len(set(lengths))==1  # No distinctive long target chain.
    for q in questions:
        if q['kind']=='hash':assert 'table ' not in q['question'] and all(s['id'] not in q['question'] for s in sources)


def test_missing_edges_and_incorrect_targets_never_become_random_answers():
    edges=[('a','b'),('b','c')]
    with pytest.raises(ValueError,match='Missing required edge'):validate_case(edges,'a',3,'jump','anything')
    with pytest.raises(ValueError,match='Incorrect'):validate_case(edges,'a',2,'jump','b')
    with pytest.raises(ValueError,match='Ambiguous'):validate_case(edges+[('a','d')],'a',1,'jump','d')


def test_native_chat_token_count_uses_ids_not_batchencoding_field_count():
    class Tokenizer:
        def apply_chat_template(self,*a,**kw):return {'input_ids':list(range(4097)),'attention_mask':[1]*4097}
    with pytest.raises(RuntimeError,match='exceeds native cutoff'):
        training_token_lengths(Tokenizer(),[{'messages':[]}])


def test_hash_grading_keeps_case_and_does_not_extract_answers_from_prose():
    expected='AbcD1234 wXyZ5678'
    assert hash_score(expected,expected)['exact']
    assert not hash_score(expected.lower(),expected)['exact']
    assert not hash_score('The answer is '+expected,expected)['valid_hash_format']
    assert hash_score('AbcD1234 ABCD5678',expected)['correct_prefix_values']==1


def test_coherence_diagnostics_do_not_claim_semantic_correctness():
    facts=dict(code='AbcD1234',file='wXyZ5678.bin',timeout=17,retries=2,total_attempts=3)
    result=coherence_diagnostics('AbcD1234 使用 wXyZ5678.bin；17 秒後最多重試兩次，總共三次。',[1,2,3,4]*20,facts)
    assert result['exact_code_preserved'] and result['exact_filename_preserved']
    assert result['total_attempts_mentioned'] and result['repeated_4gram_fraction']>.8
    assert result['manual_semantic_review_required'] and 'coherent' not in result
