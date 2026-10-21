#!/usr/bin/env bash
# AutoDL 服务器环境安装与自检。镜像选 PyTorch 2.x + Python 3.11/3.12 + CUDA 12.x（RTX 50 系列需 CUDA 12.8 镜像）。
# 用法：bash setup.sh
set -e
cd "$(dirname "$0")"
[ -f /etc/network_turbo ] && source /etc/network_turbo || true          # AutoDL 学术加速（可选）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
python -c "import torch, sys; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# 不覆盖镜像自带的 torch
grep -v '^torch' requirements.txt > /tmp/req.txt
pip install -r /tmp/req.txt
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
for repo in ("amazon/chronos-2", "google/timesfm-3.0-pytorch"):
    p = snapshot_download(repo)
    print("模型已下载：", repo, "->", p)
import torch, lightgbm, sklearn, statsmodels, arch, chronos, timesfm3, pandas, numpy, scipy
print("依赖检查通过 | lightgbm", lightgbm.__version__, "| sklearn", sklearn.__version__, "| pandas", pandas.__version__,
      "| numpy", numpy.__version__, "| scipy", scipy.__version__)
EOF
cd code && python -c "import util; print('容器可用 CPU 核数：', util.n_cpus())"
nvidia-smi --query-gpu=name,memory.total --format=csv
free -g | head -2
df -h . | tail -1
