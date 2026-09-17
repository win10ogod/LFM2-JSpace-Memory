"""Deterministic checksum for serialized physical-memory tensors."""
import hashlib

def _checksum(tensors):
    result=hashlib.sha256()
    for name,tensor in sorted(tensors.items()):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()
