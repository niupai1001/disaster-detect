"""latent_response_sensor_memory_v1, spec_revision=1.2.

Phase 1 contains only modules whose contract is frozen by the specifications:
one P32 per SensorKey, shared multi-slot memory, masked dynamic convolution,
module queries and exact native Sub-Patch/micro-cell geometry. Fusion, spatial
backbone, decoder and adapters stay unimplemented until I1-I5 are confirmed.
"""
from .types import SensorKey, ChannelMeta, NativeGrid, ChannelObservation
from .registry import SensorRegistry, ChannelCatalog
from .sensor_memory import SensorMemory
from .module_query import ModuleQuery, KINDS
from .dynamic_conv import DynamicKernelBank, masked_native_conv
from .native_patch import reference_gsd, subpatch_size, aggregate_microcells

ARCHITECTURE_ID = 'latent_response_sensor_memory_v1'
SPEC_REVISION = '1.2'

__all__ = [
    'ARCHITECTURE_ID', 'SPEC_REVISION', 'SensorKey', 'ChannelMeta', 'NativeGrid',
    'ChannelObservation', 'SensorRegistry', 'ChannelCatalog', 'SensorMemory',
    'ModuleQuery', 'KINDS', 'DynamicKernelBank', 'masked_native_conv',
    'reference_gsd', 'subpatch_size', 'aggregate_microcells',
]
