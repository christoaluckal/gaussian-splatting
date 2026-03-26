import argparse
import os
import re
import subprocess
import sys


DEFAULT_NAIVE_LOD_STAGE_ITERATIONS = 5000
WANDB_PROJECT = 'gaussian-splatting-lod'
EXPERIMENTS = [
    {"label": "baseline", "start_scale": 2, "resolution_scales": [2], "match_resolution": False},
    {"label": "naive-lod", "start_scale": 2, "resolution_scales": [2, 4, 8], "match_resolution": False},
    {"label": "matched-naive-lod", "start_scale": 2, "resolution_scales": [2, 4, 8], "match_resolution": True},
    # {"label": "baseline", "start_scale": 4, "resolution_scales": [4], "match_resolution": False},
    # {"label": "naive-lod", "start_scale": 4, "resolution_scales": [4, 8], "match_resolution": False},
    # {"label": "matched-naive-lod", "start_scale": 4, "resolution_scales": [4, 8], "match_resolution": True},
    # {"label": "baseline", "start_scale": 8, "resolution_scales": [8], "match_resolution": False},
]


def warn(message):
    print(f"[WARN] {message}")


def _normalize_source_path(source_path):
    return os.path.abspath(os.path.expanduser(source_path))


def _warn_if_model0_missing(source_path, role):
    if os.path.basename(source_path) != 'model0':
        warn(
            f"Expected {role} source to point to a `model0` directory. "
            f"Received `{source_path}`."
        )


def _warn_if_base_convention_missing(base_source):
    scene_dir = os.path.basename(os.path.dirname(base_source))
    if not re.match(r'^.+_base$', scene_dir):
        warn(
            "Expected base source parent directory to follow `<scene>_base`. "
            f"Received `{scene_dir}`."
        )


def _derive_split_config(split_source, final_extension_iteration):
    split_source = _normalize_source_path(split_source)
    _warn_if_model0_missing(split_source, 'split')

    scene_dir = os.path.basename(os.path.dirname(split_source))
    match = re.match(r'^(?P<prefix>.+)_split(?P<split_idx>[0-9]+)$', scene_dir)
    if not match:
        warn(
            "Expected split source parent directory to follow `<scene>_splitN` "
            "so the extension count can be inferred. "
            f"Received `{scene_dir}`."
        )
        raise ValueError(
            'Cannot infer split extension count from split source. '
            'Expected `<scene>_splitN/model0`.'
        )

    split_idx = int(match.group('split_idx'))
    if split_idx < 2:
        raise ValueError('Split source must encode at least one extension via `_splitN` with N >= 2.')

    return {
        'label': 'split',
        'scene_name': scene_dir,
        'source_path': split_source,
        'xtend': split_idx - 1,
        'default': False,
        'splitter_itr': final_extension_iteration // (split_idx - 1),
    }


def build_scene_configs(base_source, split_source, final_extension_iteration):
    if final_extension_iteration < 0:
        raise ValueError('--final_extension_iteration must be non-negative.')

    base_source = _normalize_source_path(base_source)
    split_source = _normalize_source_path(split_source)

    _warn_if_model0_missing(base_source, 'base')
    _warn_if_base_convention_missing(base_source)

    return [
        {
            'label': 'base',
            'scene_name': os.path.basename(os.path.dirname(base_source)),
            'source_path': base_source,
            'xtend': 0,
            'default': True,
            'splitter_itr': None,
        },
        _derive_split_config(split_source, final_extension_iteration),
    ]


def _resolve_stage_iterations(scene_config, experiment):
    if experiment['label'] == 'baseline':
        return DEFAULT_NAIVE_LOD_STAGE_ITERATIONS

    if scene_config['splitter_itr'] is None:
        return DEFAULT_NAIVE_LOD_STAGE_ITERATIONS

    total_levels = len(experiment['resolution_scales'])
    if total_levels <= 0:
        raise ValueError('resolution_scales must contain at least one level.')

    return max(1, scene_config['splitter_itr'] // total_levels)


def build_experiment_name(scene_config, experiment):
    return (
        f"{scene_config['scene_name']}-{experiment['label']}"
        f"-r{experiment['start_scale']}"
    )


def build_command(scene_config, experiment):
    experiment_name = build_experiment_name(scene_config, experiment)
    output_path = f"output/{experiment_name}"
    naive_lod_stage_iterations = _resolve_stage_iterations(scene_config, experiment)
    command = [
        sys.executable,
        'train_nomask.py',
        '-s',
        scene_config['source_path'],
        '-m',
        output_path,
        '-r',
        str(experiment['start_scale']),
        '--resolution_scales',
        *[str(scale) for scale in experiment['resolution_scales']],
        '--naive_lod_stage_iterations',
        str(naive_lod_stage_iterations),
        '--pkl_name',
        f"{output_path}/result.pkl",
        '--wandb_project',
        WANDB_PROJECT,
        '--wandb_name',
        experiment_name,
        '-x',
        str(scene_config['xtend']),
        '--eval',
    ]
    if scene_config['default']:
        command.append('--default')
    if scene_config['splitter_itr'] is not None:
        command.extend(['--splitter_itr', str(scene_config['splitter_itr'])])
    if experiment.get('match_resolution', False):
        command.append('--match_resolution')
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_source',
        type=str,
        required=True,
        help='Path to the non-split source scene, ideally ending in `<scene>_base/model0`.',
    )
    parser.add_argument(
        '--split_source',
        type=str,
        required=True,
        help='Path to the split source scene, expected to end in `<scene>_splitN/model0`.',
    )
    parser.add_argument(
        '--final_extension_iteration',
        type=int,
        required=True,
        help='Iteration where the final extension should occur for the split variant.',
    )
    args = parser.parse_args()

    base_source = _normalize_source_path(args.base_source)
    split_source = _normalize_source_path(args.split_source)
    if not os.path.isdir(base_source):
        raise FileNotFoundError(f"Base source scene does not exist: `{base_source}`.")
    if not os.path.isdir(split_source):
        raise FileNotFoundError(f"Split source scene does not exist: `{split_source}`.")

    scene_configs = build_scene_configs(
        base_source,
        split_source,
        args.final_extension_iteration,
    )
    for scene_config in scene_configs:
        for experiment in EXPERIMENTS:
            command = build_command(scene_config, experiment)
            print('Running:', ' '.join(command))
            try:
                subprocess.run(command, check=True)
            except Exception:
                pass


if __name__ == '__main__':
    main()
