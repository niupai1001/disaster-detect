"""Reuse admitted native pixels and frozen TRAIN statistics; add physical conditions."""
from pathlib import Path
import json
import math
from encoder_v3.data import sha256
from encoder_v3.latent_response.data import LRDataset, load_catalogs
from encoder_v3.latent_response.enmap_data import NativeEnmapDataset, CombinedDataset, KEY
from encoder_v3.latent_response.registry import ChannelCatalog
from encoder_v3.latent_response.checkpoint import digest
from .metadata import MetadataResolver
from .model import NativeChannel, Sample

ROOT = Path(__file__).resolve().parents[3]


def load_data():
    config = json.loads((ROOT/'src/encoder_v3/config_optical_selfsup_v1.json').read_text(encoding='utf-8'))['data']
    catalogs = load_catalogs(ROOT/'outputs/encoder_v3/latent_response/channel_catalog_frozen.json')
    catalogs[KEY] = ChannelCatalog.from_ordered([f'B{i:03d}' for i in range(1, 225)])
    datasets = {role: LRDataset(config['release'], config['contracts'], role=role, catalogs=catalogs) for role in ('TRAIN', 'VAL')}
    for dataset in datasets.values():dataset.buffered_npz=True
    enmap_path = json.loads((ROOT/'configs/config_lr_e0_v1.json').read_text())['enmap_release']
    datasets['TRAIN'] = CombinedDataset(datasets['TRAIN'], NativeEnmapDataset(enmap_path, catalogs))
    return datasets, MetadataResolver(ROOT/'data/physics/sensor_physics.json')


def sample_from_dataset(dataset, index, resolver, device):
    observation = dataset[index]
    row = observation.metadata
    bands = {b['band_key']: b for b in row['bands']}
    # The existing adapter enforces source identity, QA, units and TRAIN-only z statistics.
    normalized = dataset.channels(observation, device)
    channels = [NativeChannel(c.values, c.valid, c.grid, resolver.resolve(row, bands[c.channel_meta.channel_id]),
                              c.channel_meta.channel_id) for c in normalized]
    return Sample(observation.identifier, row['role'], channels)


def identity(config, datasets, resolver):
    folders = [ROOT/'src/encoder_v3/shared_mae', ROOT/'src/encoder_v3/latent_response']
    files = {str(p.relative_to(ROOT)): sha256(p) for folder in folders for p in folder.glob('*.py')}
    for name in ('data.py', 'tasks.py', 'model.py', 'enmap_admission.py'):
        p = ROOT/'src/encoder_v3'/name;files[str(p.relative_to(ROOT))] = sha256(p)
    return dict(code=digest(files), code_files=files, config=digest(config), physical_catalog=resolver.catalog_sha,
                data={role: ds.identities for role, ds in datasets.items()},
                members={role: digest([(r['id'], r['role']) for r in ds.rows]) for role, ds in datasets.items()})


class EpochPlan:
    """Include single-channel observations; they have spatial A but no B1 task."""
    def __init__(self, rows, seed, batch_size):
        from encoder_v3.latent_response.epoch_batches import EpochBatches
        if any(r['role'] != 'TRAIN' or not r['bands'] for r in rows):
            raise ValueError('Epoch plan requires nonempty TRAIN observations only')
        # Reuse the tested shuffle/tail/resume logic without its legacy two-channel eligibility gate.
        eligible = [dict(r, bands=[None, None]) for r in rows]
        self.plan = EpochBatches(eligible, seed, batch_size)
        self.rows = {r['id']: r for r in rows}
        self.batch_size = batch_size

    def __len__(self):return len(self.plan)
    def batch(self, cursor):return [self.rows[r['id']] for _, r in self.plan.batch(cursor)]
    def progress(self, cursor):return self.plan.progress(cursor)
    def observations_seen(self, cursor):return self.plan.observations_seen(cursor)
    def validation_due(self, cursor, checks):return self.plan.validation_due(cursor, checks)


def validate_config(config, *, require_scale=True):
    for key in ('epochs', 'batch_size', 'micro_batch_observations', 'micro_batch_pixels', 'validations_per_epoch', 'checkpoint_every_batches'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError('Positive integer '+key+' required')
    if config.get('objective') != 'observation_equal' or config.get('architecture') != 'shared_physical_mae':
        raise ValueError('Explicit shared_physical_mae observation-equal configuration required')
    if require_scale and (not isinstance(config.get('patch_metres'), (int, float)) or not math.isfinite(config['patch_metres']) or config['patch_metres'] <= 0):
        raise ValueError('Set physical patch length L in patch_metres; p=L/GSD, N=ceil(W*gx/L)*ceil(H*gy/L). No scale inferred from satellite ID.')
    if not 0 <= config['warmup_epochs'] < config['epochs']:
        raise ValueError('Invalid warmup_epochs')


def build_model(config):
    from .model import SharedMAE
    return SharedMAE(**{k: config[k] for k in ('patch_metres', 'width', 'layout', 'encoder_depth', 'decoder_depth', 'mask_ratio', 'decoder_width')})
