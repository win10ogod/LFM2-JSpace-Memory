"""Keep model exports limited to their recursive runtime dependencies."""
import ast
import hashlib
import json
from pathlib import Path
import shutil


def runtime_files(source, config):
    """Resolve local AutoClass entry points, including imports inside methods."""
    source = Path(source)
    pending = set()
    for value in config['auto_map'].values():
        for entry in value if isinstance(value, (tuple, list)) else [value]:
            if entry is None:
                continue
            module = entry.split('--')[-1].rsplit('.', 1)[0]
            if '.' in module or '/' in module:
                raise ValueError(f'Expected a flat runtime module: {module}')
            pending.add(module + '.py')
    found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        tree = ast.parse((source / name).read_text(encoding='utf-8'))
        found.add(name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1:
                    raise ValueError(f'Unsupported relative runtime import in {name}')
                dependencies = ([node.module.split('.')[0]] if node.module
                                else [alias.name for alias in node.names])
                pending.update(dep + '.py' for dep in dependencies)
    return sorted(found)


def sync_runtime(source, export):
    """Copy only code used by the exported architecture; return an audit."""
    source, export = Path(source), Path(export)
    config = json.loads((export / 'config.json').read_text())
    files = runtime_files(source, config)  # Validate the full closure before copying.
    hashes = {}
    for name in files:
        temporary = export / (name + '.updating')
        shutil.copy2(source / name, temporary)
        temporary.replace(export / name)
        hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    return hashes
