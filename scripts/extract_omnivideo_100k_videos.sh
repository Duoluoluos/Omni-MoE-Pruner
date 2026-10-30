#!/usr/bin/env bash
# Extract the official split archive without first materializing a second .tar.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${1:-${project_root}/ft_data}"

parts=(
  "${data_dir}/videos.tar.part_aa"
  "${data_dir}/videos.tar.part_ab"
  "${data_dir}/videos.tar.part_ac"
  "${data_dir}/videos.tar.part_ad"
  "${data_dir}/videos.tar.part_ae"
)

for part in "${parts[@]}"; do
  if [[ ! -s "${part}" ]]; then
    printf 'Missing or empty archive part: %s\n' "${part}" >&2
    exit 1
  fi
done

if [[ -e "${data_dir}/videos" ]]; then
  printf 'Refusing to overwrite existing path: %s\n' "${data_dir}/videos" >&2
  printf 'Inspect it before retrying extraction.\n' >&2
  exit 1
fi

printf 'Extracting OmniVideo-100K videos into %s\n' "${data_dir}"
cat "${parts[@]}" | tar -xpf - -C "${data_dir}" --no-same-owner

video_count="$(find "${data_dir}/videos" -type f -name '*.mp4' -print | wc -l)"
printf 'Extraction finished: %s video files.\n' "${video_count}"
