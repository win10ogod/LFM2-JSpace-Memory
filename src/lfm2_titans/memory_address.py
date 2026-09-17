"""One native sparse-concept address format; no generated-label branch."""
import hashlib,json
from .native_concepts import NativeConceptAddress,validate_native_address,native_tensors,native_shapes,load_native_address

def validate_address(model,address):
    if not isinstance(address,NativeConceptAddress):raise ValueError('native concept address required')
    return validate_native_address(model,address)

def address_tensors(address):return {} if address is None else native_tensors(address)
def address_shapes(model,metadata):return {} if metadata is None else native_shapes(model,metadata)
def load_address(model,metadata,tensors):return None if metadata is None else load_native_address(model,metadata,tensors)
def memory_hash(identity,tensor_checksum,address_metadata):
    value=json.dumps(dict(identity=identity,tensors=tensor_checksum,address=address_metadata),ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(value).hexdigest()
