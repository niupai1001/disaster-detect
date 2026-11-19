"""Use the historical whole-scene training pipeline with catalogued Viewer names."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import train_pd_here_s60_whole_scene as trainer
from index_pd_here_s60_versions import confirmed_version, sync_viewer_labels, viewer_labels


def named_writer(directory, *, experiment, variant, config, **kwargs):
    version = confirmed_version(config['model_id'])
    parents = config['parents']
    if kwargs.get('resume'):
        sync_viewer_labels(directory, version, parents)
    experiment_label, variant_label = viewer_labels(version, parents)
    return trainer_writer(directory, experiment=experiment | experiment_label,
                          variant=variant | variant_label, config=config, **kwargs)


trainer_writer = trainer.RunWriter


if __name__ == '__main__':
    trainer.RunWriter = named_writer
    trainer.main()
