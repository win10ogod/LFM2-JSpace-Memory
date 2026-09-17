"""Runtime export keeps lazy dependencies and fails before partial changes."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('export_artifacts',
    Path(__file__).resolve().parents[1]/'experiments/lfm2_export_artifacts.py')
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


def test_sync_includes_lazy_and_circular_dependencies_without_training_files(tmp_path):
    source, export = tmp_path/'source', tmp_path/'export'
    source.mkdir(); export.mkdir()
    config = {'auto_map': {'AutoModel': 'model.Model', 'AutoConfig': 'config.Config'}}
    (export/'config.json').write_text(json.dumps(config))
    (source/'model.py').write_text('from .config import Config\ndef recall():\n    from .archive import mount\n')
    (source/'config.py').write_text('class Config: pass\n')
    (source/'archive.py').write_text('from .model import recall\ndef mount(): pass\n')
    (source/'training.py').write_text('raise RuntimeError("not runtime")\n')
    hashes = artifacts.sync_runtime(source, export)
    assert set(hashes) == {'model.py', 'config.py', 'archive.py'}
    assert sorted(p.name for p in export.iterdir()) == ['archive.py', 'config.json', 'config.py', 'model.py']
    assert (export/'archive.py').read_bytes() == (source/'archive.py').read_bytes()


def test_missing_dependency_does_not_partially_replace_existing_runtime(tmp_path):
    source, export = tmp_path/'source', tmp_path/'export'
    source.mkdir(); export.mkdir()
    (export/'config.json').write_text(json.dumps({'auto_map': {'AutoModel': 'model.Model'}}))
    (export/'model.py').write_text('old model\n')
    (source/'model.py').write_text('from .missing import memory\n')
    with pytest.raises(FileNotFoundError):
        artifacts.sync_runtime(source, export)
    assert (export/'model.py').read_text() == 'old model\n'
