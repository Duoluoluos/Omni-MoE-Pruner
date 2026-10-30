#!/usr/bin/env python3
"""Create a reversible loss-curve boundary before a checkpoint resume."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def parse_iteration(record):
    value = record.get('iteration')
    if not isinstance(value, str):
        return None
    step_text, separator, _ = value.partition('/')
    if not separator:
        return None
    try:
        return int(step_text)
    except ValueError:
        return None


def read_checkpoint_step(checkpoint_dir: Path) -> int:
    tracker = checkpoint_dir / 'latest_checkpointed_iteration.txt'
    if not tracker.is_file():
        raise FileNotFoundError(f'Missing checkpoint tracker: {tracker}')
    return int(tracker.read_text(encoding='utf-8').strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = Path(args.checkpoint).resolve()
    checkpoint_step = read_checkpoint_step(checkpoint_dir)
    if not (checkpoint_dir / 'adapter_model.safetensors').is_file():
        raise FileNotFoundError(f'Missing LoRA adapter: {checkpoint_dir / "adapter_model.safetensors"}')

    log_path = output_dir / 'logging.jsonl'
    marker_path = output_dir / f'resume-from-checkpoint-{checkpoint_step}.json'
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        if marker.get('checkpoint') == str(checkpoint_dir):
            print(f'Resume curve boundary already prepared: {marker_path}')
            return 0
        raise RuntimeError(f'Refusing to reuse conflicting resume marker: {marker_path}')
    if not log_path.is_file():
        raise FileNotFoundError(f'Missing loss log: {log_path}')

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup_path = output_dir / f'logging.before-resume-{checkpoint_step}-{timestamp}.jsonl'
    shutil.copy2(log_path, backup_path)

    kept_lines = []
    dropped = 0
    prior_max_step = None
    with log_path.open('r', encoding='utf-8') as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                kept_lines.append(line)
                continue
            iteration = parse_iteration(record)
            if iteration is not None:
                prior_max_step = max(prior_max_step or iteration, iteration)
                if iteration > checkpoint_step:
                    dropped += 1
                    continue
            kept_lines.append(line)

    fd, temporary_path = tempfile.mkstemp(prefix='.logging.resume.', suffix='.jsonl', dir=output_dir)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as destination:
            destination.writelines(kept_lines)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, log_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    marker = {
        'checkpoint': str(checkpoint_dir),
        'checkpoint_step': checkpoint_step,
        'backup': str(backup_path),
        'prior_max_step': prior_max_step,
        'dropped_post_checkpoint_records': dropped,
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    marker_path.write_text(json.dumps(marker, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(marker, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'Failed to prepare resume curve: {exc}', file=sys.stderr)
        raise
