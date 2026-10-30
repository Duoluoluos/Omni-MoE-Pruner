#!/usr/bin/env python3
import glob
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path('/mnt/bn/isp-traindata-lf-1/wq/OmniZip')
sys.path.insert(0, str(PROJECT_ROOT / 'lmms-eval'))

from lmms_eval.tasks.videomme.utils import videomme_aggregate_results  # noqa: E402


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path((PROJECT_ROOT / 'logs/latest_parallel_dir.txt').read_text().strip())
    sample_files = sorted(glob.glob(str(base / 'shard_*' / '**' / '*_samples_videomme.jsonl'), recursive=True))
    if not sample_files:
        print(f'No sample files found under {base}', file=sys.stderr)
        return 1

    by_doc_id = {}
    duplicates = 0
    for file_path in sample_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                doc_id = sample.get('doc_id')
                metric = sample.get('videomme_perception_score')
                if metric is None:
                    continue
                if doc_id in by_doc_id:
                    duplicates += 1
                by_doc_id[doc_id] = metric

    results = [by_doc_id[k] for k in sorted(by_doc_id)]
    score = videomme_aggregate_results(results)
    summary = {
        'base': str(base),
        'sample_files': sample_files,
        'num_samples': len(results),
        'duplicates': duplicates,
        'videomme_perception_score': score,
    }
    out = base / 'aggregated_videomme_results.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
