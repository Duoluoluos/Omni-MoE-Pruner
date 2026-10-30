#!/usr/bin/env bash
set -u

printf 'identity: '
id
for tool in python3.12 python3.11 python3 uv conda mamba micromamba; do
  if command -v "${tool}" >/dev/null 2>&1; then
    printf '%s: %s\n' "${tool}" "$(command -v "${tool}")"
    "${tool}" --version 2>&1 | head -n 1
  else
    printf '%s: unavailable\n' "${tool}"
  fi
done

for variable in UV_PYTHON_INSTALL_MIRROR UV_ASTRAL_MIRROR_URL UV_PYTHON_INSTALL_DIR; do
  printf '%s=' "${variable}"
  printenv "${variable}" || true
done

for candidate in /usr/bin/python3.12 /usr/local/bin/python3.12 /opt/conda/bin/python /opt/pyvenv/bin/python; do
  if [[ -x "${candidate}" ]]; then
    printf 'candidate: %s\n' "${candidate}"
    "${candidate}" --version
  fi
done

echo 'workspace disk:'
df -h /mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
