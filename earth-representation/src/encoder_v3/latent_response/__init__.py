"""latent_response_sensor_memory_v1, spec_revision=1.2.

IR1/IR5/IR6 confirmed: joint 309-wide decoder, shared metadata projection,
scene-relative position. Complete model and E1 entry are independent of legacy training.
"""
from .types import SensorKey, ChannelMeta, NativeGrid, ChannelObservation
from .registry import SensorRegistry, ChannelCatalog
from .sensor_memory import SensorMemory
from .memory_read import read_memory
from .module_query import ModuleQuery, KINDS
from .dynamic_conv import DynamicKernelBank, masked_native_conv
from .native_patch import reference_gsd, subpatch_size, aggregate_microcells

ARCHITECTURE_ID = 'latent_response_sensor_memory_v1'
SPEC_REVISION = '1.3'

__all__ = [
    'ARCHITECTURE_ID', 'SPEC_REVISION', 'SensorKey', 'ChannelMeta', 'NativeGrid',
    'ChannelObservation', 'SensorRegistry', 'ChannelCatalog', 'SensorMemory', 'read_memory',
    'ModuleQuery', 'KINDS', 'DynamicKernelBank', 'masked_native_conv',
    'reference_gsd', 'subpatch_size', 'aggregate_microcells',
]
