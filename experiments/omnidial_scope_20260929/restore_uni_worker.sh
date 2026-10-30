#!/usr/bin/env bash
# One-shot recovery on the 2026-09-29 worker; fixed copy PID must not be replayed.
set -euo pipefail
cd /mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner
python3 - <<'PY'
from pathlib import Path
import os,signal,time
p=Path('/proc/1792/cmdline')
if p.exists():
 args=p.read_bytes().split(b'\0')
 assert args[:5]==[b'tar',b'--use-compress-program=zstd',b'-C',b'/home/tiger',b'-xf'],args
 os.kill(1792,signal.SIGTERM)
 for _ in range(100):
  if not p.exists():break
  time.sleep(.1)
 assert not p.exists(),'Superseded environment extraction still alive'
PY
mkdir /tmp/omnidial_runtime_20260929
tar --use-compress-program=zstd -C /tmp/omnidial_runtime_20260929 -xf .runtime/uni_scope_env_20260929.tar.zst
python3 - <<'PY'
from pathlib import Path
base=Path('/tmp/omnidial_runtime_20260929')
home=Path('/home/tiger')
assert (base/'uni_moe_venv/bin/python').exists()
assert (base/'Uni-MoE/Uni-MoE-2/uni_moe').exists()
for name in ['uni_moe_venv','Uni-MoE']:
 dst=home/name
 if dst.exists():
  # Only the incomplete directory created by this recovery attempt is present.
  backup=home/(name+'.partial_scope_20260929')
  assert not dst.is_symlink() and not backup.exists()
  dst.rename(backup)
 dst.symlink_to(base/name,target_is_directory=True)
print('Exact existing environment restored on worker local NVMe')
PY
/home/tiger/uni_moe_venv/bin/python -c 'import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.is_available())'
bash experiments/omnidial_scope_20260929/launch.sh unimoe
