#!/bin/bash
# Unified environment for the flash-attention benchmark harness.
#   source env.sh
#
# Machine-specific bits (notably the tilelang-ascend install path) live in
# env.local.sh, which is NOT committed (see .gitignore). Copy env.local.sh.example
# to env.local.sh and edit it for your machine.

FA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FA_ROOT

# --- per-machine config (tilelang activation, etc.) -- not in the repo ------- #
if [ -f "${FA_ROOT}/env.local.sh" ]; then
    source "${FA_ROOT}/env.local.sh"
else
    echo "env.sh: missing ${FA_ROOT}/env.local.sh (per-machine config)." >&2
    echo "        cp env.local.sh.example env.local.sh  and set your tilelang path." >&2
fi

# --- CANN toolkit ----------------------------------------------------------- #
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

# --- torch / torch_npu shared libs (libc10.so etc.) ------------------------- #
_torch_lib="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null)"
_tnpu_lib="$(python -c 'import torch_npu,os;print(os.path.join(os.path.dirname(torch_npu.__file__),"lib"))' 2>/dev/null)"
export LD_LIBRARY_PATH="${_torch_lib}:${_tnpu_lib}:${LD_LIBRARY_PATH}"

# --- the harness itself ----------------------------------------------------- #
export PYTHONPATH="${FA_ROOT}:${PYTHONPATH}"
