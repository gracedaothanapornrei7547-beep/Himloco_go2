#!/usr/bin/env bash
set -euo pipefail

readonly HIMLOCO_ROOT="/home/rum/himloco_lab"
readonly HIMLOCO_PYTHON="/home/rum/env_isaaclab_himloco/bin/python"
readonly HIMLOCO_ASSETS="/home/rum/himloco_assets/unitree_ros"

if [[ ! -x "${HIMLOCO_PYTHON}" ]]; then
    echo "Missing isolated Python: ${HIMLOCO_PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${HIMLOCO_ASSETS}/robots/go2_description/urdf/go2_description.urdf" ]]; then
    echo "Missing local Go2 URDF under: ${HIMLOCO_ASSETS}" >&2
    exit 2
fi
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <python-script> [arguments ...]" >&2
    echo "Example: $0 scripts/himloco_rsl_rl/train.py --task Unitree-Go2-Velocity --headless" >&2
    exit 2
fi

cd "${HIMLOCO_ROOT}"
export OMNI_KIT_ACCEPT_EULA="YES"
export UNITREE_ROS_DIR="${HIMLOCO_ASSETS}"
export PXR_WORK_THREAD_LIMIT="${PXR_WORK_THREAD_LIMIT:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 默认剥掉 DISPLAY（训练/评测都该 headless 跑）。需要肉眼看图形界面时显式
# HIMLOCO_KEEP_DISPLAY=1 —— 只影响这一个进程，默认行为完全不变。
if [[ "${HIMLOCO_KEEP_DISPLAY:-0}" == "1" ]]; then
    echo "[run_himloco] 保留 DISPLAY=${DISPLAY:-<未设置>}（图形界面模式）" >&2
    exec "${HIMLOCO_PYTHON}" "$@"
fi
exec env -u DISPLAY "${HIMLOCO_PYTHON}" "$@"
