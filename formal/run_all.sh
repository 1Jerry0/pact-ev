#!/usr/bin/env bash
# 正式实验一键运行（可中断后重跑，已完成的部分自动跳过）。用法：nohup bash run_all.sh > logs/run_all.out 2>&1 &
# 资源安排：GPU 先跑 MP 再跑 UrbanEV 的基础模型；同时 CPU 用全部核跑统计与机器学习模型，之后依次跑决策、机制实验。
cd "$(dirname "$0")/code"
mkdir -p ../logs
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
N=$(python -c "import util; print(util.n_cpus())")
echo "$(date '+%F %T') 开始，CPU 核数 $N，冒烟模式 ${SMOKE:-0}"
rm -f ../logs/.gpu_mp_done ../logs/.gpu_uev_done          # 每次启动重新判断 GPU 阶段（已完成的部分会被自动跳过）

step() { echo "$(date '+%F %T') >>> $*"; "$@" || { echo "$(date '+%F %T') !!! 失败：$*（可直接重跑 run_all.sh 续跑）"; exit 1; }; }

# A. GPU：基础模型（后台）
( python fc_tsfm.py mp && touch ../logs/.gpu_mp_done && python fc_tsfm.py uev && touch ../logs/.gpu_uev_done ) \
  > ../logs/gpu.out 2>&1 &
GPU_PID=$!

# B. CPU：统计与机器学习模型（GPU 进程占 1 个核）
export N_JOBS=$((N > 2 ? N - 1 : 1))
step python fc_stat.py mp
step python fc_stat.py uev

# C. 等 MP 基础模型完成 → 组合与校准 → MP 决策与机制实验（此时 GPU 仍在跑 UrbanEV）
while [ ! -f ../logs/.gpu_mp_done ]; do
  if ! kill -0 $GPU_PID 2>/dev/null; then echo "GPU 进程已退出但 MP 未完成，请查看 logs/gpu.out"; exit 1; fi
  sleep 30
done
step python fc_post.py mp
export N_JOBS=$N
step python dec_mp.py
step python mech.py

# D. UrbanEV
wait $GPU_PID || { echo "GPU 进程失败，请查看 logs/gpu.out"; exit 1; }
step python fc_post.py uev
step python uev.py
step python agg.py

# E. 校验与预注册分析
step python check.py
step python analysis.py
echo "$(date '+%F %T') 全部完成。结果摘要：out/tables/summary.md"
