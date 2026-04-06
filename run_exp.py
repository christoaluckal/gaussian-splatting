import argparse
import csv
import os
import re
import subprocess
import sys
from statistics import mean


DEFAULT_NAIVE_LOD_STAGE_ITERATIONS = 5000
DEFAULT_FINAL_EXTENSION_ITERATION = 7500
DEFAULT_PAIRS_ROOT = os.path.join(os.path.dirname(__file__), 'kraken_stuff')
WANDB_PROJECT = 'gaussian-splatting-lod'
EXPERIMENTS = [
    {'label': 'baseline', 'start_scale': 4, 'resolution_scales': [4], 'match_resolution': False},
    {'label': 'naive-lod', 'start_scale': 4, 'resolution_scales': [ 4, 8], 'match_resolution': False},
    {'label': 'matched-naive-lod', 'start_scale': 4, 'resolution_scales': [4, 8], 'match_resolution': True},
]
REPORT_METRICS = [
    ('l1', 'L1', 'reduction'),
    ('psnr', 'PSNR', 'improvement'),
    ('final_num_gaussians', 'Final #Gaussians', 'delta'),
    ('gaussian_model_mb', 'Gaussian Model MB', 'delta'),
    ('peak_allocated_mb', 'Peak Allocated MB', 'delta'),
    ('peak_reserved_mb', 'Peak Reserved MB', 'delta'),
    ('total_training_time_sec', 'Total Train Time (s)', 'delta'),
    ('avg_iter_time_ms', 'Avg Iter Time (ms)', 'delta'),
]


def warn(message):
    print(f'[WARN] {message}')


def _normalize_source_path(source_path):
    return os.path.abspath(os.path.expanduser(source_path))


def _warn_if_model0_missing(source_path, role):
    if os.path.basename(source_path) != 'model0':
        warn(
            f'Expected {role} source to point to a `model0` directory. '
            f'Received `{source_path}`.'
        )


def _warn_if_base_convention_missing(base_source):
    scene_dir = os.path.basename(os.path.dirname(base_source))
    if not re.match(r'^.+_base$', scene_dir):
        warn(
            'Expected base source parent directory to follow `<scene>_base`. '
            f'Received `{scene_dir}`.'
        )


def _derive_split_config(split_source, final_extension_iteration):
    split_source = _normalize_source_path(split_source)
    _warn_if_model0_missing(split_source, 'split')

    scene_dir = os.path.basename(os.path.dirname(split_source))
    match = re.match(r'^(?P<prefix>.+)_split(?P<split_idx>[0-9]+)$', scene_dir)
    if not match:
        warn(
            'Expected split source parent directory to follow `<scene>_splitN` '
            'so the extension count can be inferred. '
            f'Received `{scene_dir}`.'
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


def discover_scene_pairs(pairs_root):
    pairs_root = _normalize_source_path(pairs_root)
    if not os.path.isdir(pairs_root):
        raise FileNotFoundError(f'Pairs root does not exist: `{pairs_root}`.')

    base_sources = {}
    split_sources = {}
    for entry in sorted(os.listdir(pairs_root)):
        entry_path = os.path.join(pairs_root, entry)
        if not os.path.isdir(entry_path):
            continue

        base_match = re.match(r'^(?P<prefix>.+)_base$', entry)
        split_match = re.match(r'^(?P<prefix>.+)_split(?P<split_idx>[0-9]+)$', entry)
        model0_path = os.path.join(entry_path, 'model0')
        if not os.path.isdir(model0_path):
            warn(f'Skipping `{entry_path}` because `model0` is missing.')
            continue

        if base_match:
            prefix = base_match.group('prefix')
            base_sources[prefix] = model0_path
        elif split_match:
            prefix = split_match.group('prefix')
            split_sources.setdefault(prefix, []).append(model0_path)

    pairs = []
    all_prefixes = sorted(set(base_sources) | set(split_sources))
    for prefix in all_prefixes:
        base_source = base_sources.get(prefix)
        prefix_splits = sorted(split_sources.get(prefix, []))
        if not base_source:
            warn(f'Skipping `{prefix}` because no `<scene>_base/model0` directory was found.')
            continue
        if not prefix_splits:
            warn(f'Skipping `{prefix}` because no `<scene>_splitN/model0` directory was found.')
            continue
        for split_source in prefix_splits:
            pairs.append(
                {
                    'prefix': prefix,
                    'base_source': base_source,
                    'split_source': split_source,
                }
            )

    if not pairs:
        raise RuntimeError(f'No valid base/split pairs found under `{pairs_root}`.')
    return pairs


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
    return f"{scene_config['scene_name']}-{experiment['label']}-r{experiment['start_scale']}"


def build_command(scene_config, experiment):
    experiment_name = build_experiment_name(scene_config, experiment)
    output_path = os.path.join('output', experiment_name)
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
        f'{output_path}/result.pkl',
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
    return command, output_path


def _load_csv_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline='') as csv_file:
        return list(csv.DictReader(csv_file))


def _parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value):
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _max_metric(rows, key):
    values = [_parse_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values)


def _last_row_by_iteration(rows, split_name=None):
    filtered = rows
    if split_name is not None:
        filtered = [row for row in rows if row.get('split') == split_name]
    if not filtered:
        return None
    return max(filtered, key=lambda row: _parse_int(row.get('iteration')) or -1)


def _event_row(rows, event_name):
    for row in reversed(rows):
        if row.get('event') == event_name:
            return row
    return None


def summarize_run(run_record):
    output_path = run_record['output_path']
    train_rows = _load_csv_rows(os.path.join(output_path, 'train_metrics.csv'))
    eval_rows = _load_csv_rows(os.path.join(output_path, 'eval_metrics.csv'))
    runtime_rows = _load_csv_rows(os.path.join(output_path, 'runtime_metrics.csv'))

    final_train = _last_row_by_iteration(train_rows)
    final_eval = _last_row_by_iteration(eval_rows, split_name='test')
    training_complete = _event_row(runtime_rows, 'training_complete')

    avg_iter_time_ms = None
    iter_time_values = [_parse_float(row.get('iter_time_ms')) for row in train_rows]
    iter_time_values = [value for value in iter_time_values if value is not None]
    if iter_time_values:
        avg_iter_time_ms = mean(iter_time_values)

    return {
        'pair_name': run_record['pair_name'],
        'scene_label': run_record['scene_label'],
        'scene_name': run_record['scene_name'],
        'variant_label': run_record['variant_label'],
        'match_resolution': run_record['match_resolution'],
        'output_path': output_path,
        'success': run_record['success'],
        'returncode': run_record['returncode'],
        'l1': _parse_float(final_eval.get('l1')) if final_eval else None,
        'psnr': _parse_float(final_eval.get('psnr')) if final_eval else None,
        'final_num_gaussians': _parse_int(final_train.get('num_gaussians')) if final_train else None,
        'gaussian_model_mb': _parse_float(training_complete.get('gaussian_model_mb')) if training_complete else _max_metric(runtime_rows, 'gaussian_model_mb'),
        'peak_allocated_mb': _max_metric(runtime_rows, 'gpu_peak_allocated_mb'),
        'peak_reserved_mb': _max_metric(runtime_rows, 'gpu_peak_reserved_mb'),
        'total_training_time_sec': _parse_float(training_complete.get('total_training_time_sec')) if training_complete else None,
        'avg_iter_time_ms': avg_iter_time_ms,
    }


def _format_value(value, digits=4):
    if value is None:
        return 'N/A'
    if isinstance(value, int):
        return f'{value:,}'
    return f'{value:.{digits}f}'


def _format_delta(value, digits=4):
    if value is None:
        return 'N/A'
    return f'{value:+.{digits}f}'


def _format_percent(value, digits=2):
    if value is None:
        return 'N/A'
    return f'{value:+.{digits}f}%'


def _compute_delta(mode, baseline_value, run_value):
    if baseline_value is None or run_value is None:
        return None
    if mode == 'improvement':
        return run_value - baseline_value
    if mode == 'reduction':
        return baseline_value - run_value
    return run_value - baseline_value


def _compute_percent_delta(mode, baseline_value, run_value):
    if baseline_value is None or run_value is None or baseline_value == 0:
        return None
    return (_compute_delta(mode, baseline_value, run_value) / baseline_value) * 100.0


def _absolute_report_row(summary, baseline_summary):
    row = [
        summary['scene_label'],
        summary['variant_label'],
        'yes' if summary['match_resolution'] else 'no',
    ]
    for metric_key, _label, mode in REPORT_METRICS:
        run_value = summary.get(metric_key)
        baseline_value = baseline_summary.get(metric_key)
        row.append(_format_value(run_value, digits=4))
        row.append(_format_delta(_compute_delta(mode, baseline_value, run_value), digits=4))
    return '| ' + ' | '.join(row) + ' |'


def _percent_report_row(summary, baseline_summary):
    row = [
        summary['scene_label'],
        summary['variant_label'],
        'yes' if summary['match_resolution'] else 'no',
    ]
    for metric_key, _label, mode in REPORT_METRICS:
        run_value = summary.get(metric_key)
        baseline_value = baseline_summary.get(metric_key)
        row.append(_format_percent(_compute_percent_delta(mode, baseline_value, run_value), digits=2))
    return '| ' + ' | '.join(row) + ' |'


def write_report(report_path, pair_name, summaries):
    successful = [summary for summary in summaries if summary['success']]
    baseline_summary = next(
        (
            summary for summary in successful
            if summary['scene_label'] == 'base' and summary['variant_label'] == 'baseline'
        ),
        None,
    )

    lines = [
        '# Gaussian Splatting Experiment Report',
        '',
        f'## Pair: `{pair_name}`',
        '',
    ]

    if baseline_summary is None:
        lines.extend([
            'No successful `base` + `baseline` run was found for this pair.',
            '',
        ])
    else:
        lines.extend([
            '## 1. Baseline Reference',
            '',
            f"- Run: `{baseline_summary['scene_name']}-{baseline_summary['variant_label']}`",
            f"- Output: `{baseline_summary['output_path']}`",
            '',
            '## 2. Final Metrics vs Baseline',
            '',
            '| Scene Variant | Training Variant | Match Resolution | L1 | L1 Reduction vs Baseline | PSNR | PSNR Improvement vs Baseline | Final #Gaussians | Delta #Gaussians vs Baseline | Gaussian Model MB | Delta Gaussian Model MB vs Baseline | Peak Allocated MB | Delta Peak Allocated MB vs Baseline | Peak Reserved MB | Delta Peak Reserved MB vs Baseline | Total Train Time (s) | Delta Train Time vs Baseline | Avg Iter Time (ms) | Delta Avg Iter Time vs Baseline |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        ])
        for summary in successful:
            lines.append(_absolute_report_row(summary, baseline_summary))
        lines.extend([
            '',
            '## 3. Percentage Change vs Baseline',
            '',
            '| Scene Variant | Training Variant | Match Resolution | L1 Reduction % vs Baseline | PSNR Improvement % vs Baseline | #Gaussians Delta % vs Baseline | Gaussian Model MB Delta % vs Baseline | Peak Allocated MB Delta % vs Baseline | Peak Reserved MB Delta % vs Baseline | Train Time Delta % vs Baseline | Avg Iter Time Delta % vs Baseline |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        ])
        for summary in successful:
            lines.append(_percent_report_row(summary, baseline_summary))
        lines.extend([
            '',
            '## 4. Raw Run Values',
            '',
            '| Scene Variant | Training Variant | Match Resolution | Output Path | L1 | PSNR | Final #Gaussians | Gaussian Model MB | Peak Allocated MB | Peak Reserved MB | Total Train Time (s) | Avg Iter Time (ms) |',
            '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        ])
        for summary in successful:
            lines.append(
                '| ' + ' | '.join([
                    summary['scene_label'],
                    summary['variant_label'],
                    'yes' if summary['match_resolution'] else 'no',
                    f"`{summary['output_path']}`",
                    _format_value(summary['l1'], digits=4),
                    _format_value(summary['psnr'], digits=4),
                    _format_value(summary['final_num_gaussians'], digits=0),
                    _format_value(summary['gaussian_model_mb'], digits=4),
                    _format_value(summary['peak_allocated_mb'], digits=4),
                    _format_value(summary['peak_reserved_mb'], digits=4),
                    _format_value(summary['total_training_time_sec'], digits=4),
                    _format_value(summary['avg_iter_time_ms'], digits=4),
                ]) + ' |'
            )
        lines.append('')

    failed = [summary for summary in summaries if not summary['success']]
    if failed:
        lines.extend([
            '## 5. Failed Runs',
            '',
            '| Scene Variant | Training Variant | Match Resolution | Output Path | Return Code |',
            '|---|---|---|---|---:|',
        ])
        for summary in failed:
            return_code = 'N/A' if summary['returncode'] is None else str(summary['returncode'])
            lines.append(
                '| ' + ' | '.join([
                    summary['scene_label'],
                    summary['variant_label'],
                    'yes' if summary['match_resolution'] else 'no',
                    f"`{summary['output_path']}`",
                    return_code,
                ]) + ' |'
            )
        lines.append('')

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as report_file:
        report_file.write('\n'.join(lines))


def _report_filename(pair_name):
    return f'{pair_name}_report.md'


def run_pair(pair, final_extension_iteration):
    scene_configs = build_scene_configs(
        pair['base_source'],
        pair['split_source'],
        final_extension_iteration,
    )
    pair_name = f"{scene_configs[0]['scene_name']}__{scene_configs[1]['scene_name']}"

    run_records = []
    for scene_config in scene_configs:
        for experiment in EXPERIMENTS:
            command, output_path = build_command(scene_config, experiment)
            print('Running:', ' '.join(command))
            success = True
            returncode = 0
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                success = False
                returncode = exc.returncode
            except Exception:
                success = False
                returncode = None

            run_records.append(
                {
                    'pair_name': pair_name,
                    'scene_label': scene_config['label'],
                    'scene_name': scene_config['scene_name'],
                    'variant_label': experiment['label'],
                    'match_resolution': experiment.get('match_resolution', False),
                    'output_path': output_path,
                    'success': success,
                    'returncode': returncode,
                }
            )

    summaries = [summarize_run(run_record) for run_record in run_records]
    report_path = os.path.join('output', _report_filename(pair_name))
    write_report(report_path, pair_name, summaries)
    print(f'Report written to {report_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pairs_root',
        type=str,
        default=DEFAULT_PAIRS_ROOT,
        help='Root directory containing `<scene>_base` and `<scene>_splitN` subdirectories.',
    )
    parser.add_argument(
        '--final_extension_iteration',
        type=int,
        default=DEFAULT_FINAL_EXTENSION_ITERATION,
        help='Iteration where the final extension should occur for split variants. Default: 7500.',
    )
    args = parser.parse_args()

    pairs = discover_scene_pairs(args.pairs_root)
    for pair in pairs:
        run_pair(pair, args.final_extension_iteration)


if __name__ == '__main__':
    main()
