# Visual Foothold Chain 修复与服务器验证

## 范围与证据

本次仅修改 Visual Foothold 几何规划、事件生命周期、诊断和回放；保持 Task ID、
41 维新增观测、29 关节动作、PPO/MoE/AMP、相机参数、物理阈值及训练入口不变。
依照 NVIDIA debugging-rl-training、building-environments、using-sensors-actuators、
training-rl-agents Skills 审查；命令以本项目 Isaac Sim 5.1 的脚本为准。

历史缓存采集于 2026-10-12 12:09:45（北京时间），最后 step=16050；15100–16050
有效目标比例约 0.336%。这是历史训练参考，不是本次修复后的实测数据，也不是同条件冻结
策略基线。远端旧源码哈希未核实，不能将本地复现直接等同于远端每次失败的原因。

## 修改

* `perception.py`：足印、可达性、高差、方向及走廊覆盖共同过滤后才排序；按 128 个候选
  分块限制临时张量。走廊从实际摆动足起步，与参考轨迹相同。原中后段六个采样点覆盖要求、
  深度未知处理及物理安全阈值不变；初始三个采样点仍沿用原规则，不构成完整路径安全证明。
* `controller.py`：锁定的触地误差/风险属于事件，地图清理不再取消它；接触取消、结算和
  episode reset 才结束该事件。记录每足已支付的 command slot，防止反弹或重规划重复支付。
  不放宽 1.5 cm 锚点转换一致性条件；不可靠地图仍清空。
* 增加阶段失败、地图/目标清理、历史使用、触地确认与支付计数。
  `mdp.py` 只在回放记录开启时复制快照。`play.py` 保存实际实现哈希、完整状态/计数、
  重置前误差、终止原因、episode 时长及各项加权 reward rate 的积分。
* `diagnostics.py` 将加总量转换成含明确分子/分母的 JSON 指标，空分母输出 `null`。

## 已执行的本地验证

环境：Windows，Python 3.7.4 / Torch 1.13.1+cpu，CUDA 不可用；现有 Python 3.11.15
环境没有 Torch。两个解释器均未安装 Isaac Sim/Isaac Lab。没有安装/升级依赖。

| 检查 | 结果 |
|---|---|
| 原 Visual Foothold CPU 测试 | 修改前 7/7 通过 |
| A/B 新回归 | 修改前 2 项按预期失败，分别为无效目标及丢失支付 |
| 修复后的 `test_core.py` | 13/13 通过 |
| `test_motion_pool.py` | 4/4 通过，包含恢复时拒绝专家权重改变 |
| `foot_state_demo/test_foot_state.py` | 6/6 通过 |
| `foot_state_demo/test_foothold_reference.py` | 7/7 通过 |
| 摆脚起点退化的内存 mutation | 将走廊起点改回支撑足后，盲区回归按预期失败 |
| 静态检查 | 6 个修改 Python 文件语法通过；7 个 Bash 命令块通过 `bash -n`；`git diff --check` 通过 |
| 扩展 `test_mixed_parkour.py` | 10/11 通过；已有源码结构断言期待 2 条语句、实际 3 条，未修改 HEAD 临时副本亦失败 |
| 扩展 `test_stair_training.py` | Torch 解释器下 8/9 通过；1 项缺少 trimesh，未安装依赖 |

Mixed 既有失败和楼梯可选依赖缺失不在本次修复范围，没有修改这些测试或任务来隐藏失败。
CPU 合成场景的可比结果：

| 场景 | 修改前 | 修改后 |
|---|---|---|
| 地面仅 x≥0.12 m 可见，最低成本目标走廊未知 | 选 (0.24, 0.20)，返回无效 | 选 (0.28, 0.20)，返回有效 |
| 首次落点误差 2 cm，确认期间移动 5 mm | 支付 1 次，积分 0.939413 | 支付 1 次，积分 0.939413 |
| 首次落点误差 2 cm，确认期间移动 20 mm | 支付 0 次 | 支付 1 次，积分 0.939413；地图/目标仍清除 |
| x≥0.32 m 可见、摆脚从 x=0 开始、无历史 | 无效 | 仍无效 |
| 同盲区，有可靠且未过期的历史观测 | 未纳入原回归 | 有效；历史过期后再次无效 |

这些是算法与记账验证，不是 Isaac Sim、CUDA、checkpoint 加载或策略行为验证。
修复后真实有效目标比例、落点误差及摔倒率目前均为 N/A。

## 诊断口径

训练 TensorBoard 的 `VisualFoothold/raw/<name>` 是本次 reset 环境组的加总量。
Runner 可以对多个 reset 组求均值：计算对应分子 tag / 分母 tag 的比值，
不要再平均每回合比值。`episodes` 为 reset 样本数；`ticks_available`、
`touchdown_samples_available` 和 `risk_samples_available` 明确空样本。
旧的零填充误差/命中率标签已移除。

| 指标/阶段 | 分子或计数 | 分母/解释 |
|---|---|---|
| 有效目标覆盖 | valid_ticks | ticks；训练保留同名近似图，精确聚合用 raw 比值 |
| 移动时覆盖 | valid_ticks | moving_ticks |
| 生成率 | targets_generated | planning_attempts |
| 单步保留率 | target_retained_ticks | target_exposure_ticks；同一目标上一步有效且本步未清除 |
| 无观测 | no_observations | planning_attempts |
| 无合法支撑足印 | no_safe_candidates | 有观测但没有安全足印 |
| 可达性等过滤失败 | feasibility_failures | 有安全足印但共同几何约束全部失败 |
| 走廊未知 | corridor_unknown | 有可达足印但全部走廊覆盖失败 |
| 走廊覆盖 | corridor_observed_samples | corridor_required_samples；只计几何可行候选的中后段采样点 |
| 历史输入 | history_points_used、memory_frames_used | 每次规划累计输入量；前者年龄大于一个捕获周期，不等于历史一定覆盖了走廊 |
| 清除原因 | clear_anchor_loss、clear_transfer、clear_phase、clear_stopped | targets_cleared；cleared_unlanded 指清理时尚未结算，可能在同帧随后结算 |
| 确认与支付 | touchdown_candidates、candidate_aborts、confirmed_commanded_touchdowns、reward_payments | 物理 touchdowns 包括无视觉命令触地；有效确认与支付应一致 |
| 误差 | error_sum_m | commanded_touchdowns；无样本 N/A |
| 5 cm 命中率 | hits | commanded_touchdowns；无样本 N/A |
| 视觉风险 | visual_risk | risk_samples；未知落点不当作安全样本 |

回放 `summary.json` 保存 `chain_totals` 和带分子、分母的 `chain_metrics`。
`weighted_reward_integrals` 是本项目 MultiRewardManager 的加权 reward rate × dt 总和；
`visual_trace.csv` 的 `weighted_rate_*` 是加权率，`step_*` 是本步原始计数/积分增量。
原 `Episode_Reward/*`、终止日志照常保留。完成的 episode 时长与末尾未完成 episode 分开。
`status=complete` 仅证明指定步数运行完毕，不代表完成地形路线。

## 容器与同步

以下命令由用户在服务器执行，本次没有远程部署、停止旧训练或启动 GPU 任务。
旧训练使用旧逻辑，不应在它运行中途切换源码；先保留 checkpoint/log 并结束该进程。
要求现有 `instinctlab` 容器已运行、已挂载项目与数据、沿用原来的 Isaac Sim 5.1 / Instinct-RL
Python 环境，且 `cuda:0` 是该容器内空闲的逻辑设备。不要升级依赖。

宿主机进入容器：

```bash
docker exec -it instinctlab bash
```

容器内执行；有未提交修改时先妥善保留，不使用 force/reset：

```bash
cd /workspace/instinctlab
git status --short
git fetch origin
git switch fix/visual-foothold-chain
git pull --ff-only origin fix/visual-foothold-chain
git log -1 --oneline
nvidia-smi

export TRAIN_DEVICE=cuda:0
export RUN=/workspace/instinctlab/logs/instinct_rl/g1_parkour_visual_foothold/20261011_091423_visual_author_grail_fresh_device-cuda:0_resume-False
export CHECKPOINT=model_16000.pt
test -s "$RUN/$CHECKPOINT"
test -s "$RUN/params/env.yaml"
test -s "$RUN/params/agent.yaml"
test -s "$RUN/params/motion_inventory.json"

python -B scripts/visual_foothold/test_core.py -v
python -B scripts/visual_foothold/test_motion_pool.py -v
```

## 1. 冻结策略验证：先执行

实际脚本固定使用 Visual Foothold Task，从旧 run 恢复 env/agent 配置；
只加载 actor/critic 权重与所需归一化状态，`strict=True`，不用 AMP 数据集或优化器。
观察/动作维度保持不变仅意味着结构上预期兼容；实际加载和行为兼容必须由此验证。

```bash
cd /workspace/instinctlab
python -u scripts/visual_foothold/play.py \
  --load_run "$RUN" --checkpoint "$CHECKPOINT" \
  --headless --device "$TRAIN_DEVICE" --num_envs 4 \
  --max_steps 1500 --seed 42 --video \
  --output_dir "outputs/visual_foothold/chain_fixed_$(date +%Y%m%d_%H%M%S)"
```

没有视频需求时去掉 `--video`，CSV/JSON 照常输出。检查保存的 `params/visual_foothold_runtime.json`
哈希确为新源码；查看 CSV 的目标锁定、实际摆脚位置/参考、接触 FSM、有效确认和支付，
将 `chain_metrics` 的误差样本数一并报告。比较时固定 checkpoint、4 环境、seed 42、
地形/命令配置和 1500 步；不要与历史 2048 环境训练的每回合平均值宣称严格同比。

## 2. 短时训练：分启动与恢复两步

历史运行实际是 GRAIL=0.75、author=0.25。下列命令保留该比例，
不要照最早粘贴的 0.5 恢复旧 AMP 判别器。现有入口会核对 motion inventory 的哈希与权重。
先执行随机初始化的 32 环境/5 次更新，验证有限 loss、动作维度、GPU 内存和最终保存：

```bash
cd /workspace/instinctlab
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Visual-Foothold-Amp-G1-v0 \
  --motion_root /workspace/instinctlab/data/grail_instinctlab \
  --author_motion_root /workspace/instinctlab/data/hiking_in_the_wild/parkour_motion_reference \
  --amp_grail_fraction 0.75 \
  --headless --num_envs 32 --max_iterations 5 \
  --device "$TRAIN_DEVICE" --seed 42 --run_name visual_chain_startup \
  agent.device="$TRAIN_DEVICE" agent.resume=False agent.log_interval=1
```

五次更新只证明启动。冻结回放通过后，以隔离新 run 执行 50 次恢复更新，
验证 optimizer/AMP/normalizer 的完整恢复和至少一个 20 s 回合窗口；不覆盖旧 run：

```bash
cd /workspace/instinctlab
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Visual-Foothold-Amp-G1-v0 \
  --motion_root /workspace/instinctlab/data/grail_instinctlab \
  --author_motion_root /workspace/instinctlab/data/hiking_in_the_wild/parkour_motion_reference \
  --amp_grail_fraction 0.75 \
  --resume --load_run "$RUN" --checkpoint "$CHECKPOINT" \
  --headless --num_envs 32 --max_iterations 50 \
  --device "$TRAIN_DEVICE" --seed 42 --run_name visual_chain_resume_smoke \
  agent.device="$TRAIN_DEVICE" agent.log_interval=1
```

用实际新 run 和实际保存的 model 文件再执行同一冻结回放命令。检查最终 checkpoint 的存在与
正常退出；编号以文件及运行日志为准，不能假设 5 次更新一定产生 `model_5.pt`。
`--warm_start` 当前只支持 Curb Crossing，不适用于本任务。

## 放行门槛与剩余风险

先确认没有非有限值、形状/加载错误，触地支付不重复、不跨 reset，确认与支付计数一致。
目标覆盖长期低于 1% 或没有有效触地误差样本时，继续查 `corridor_unknown`、
锚点/目标清除原因及历史使用，不启动长训练；1% 是排查门槛，不是训练成功标准。
在相同冻结评估条件下检查 episode 时长、物理失败终止和 reward terms 没有明显退化。
相机近场盲区与频繁锚点失效仍可能导致覆盖不足；固定 0.4 s 步相与旧策略步态可能不匹配。
旧 value/optimizer 面对新奖励语义的适应性未经验证。冻结旧 actor 的变化不能证明重新学习改善。

## 3. 正式恢复：仅在上面的 GPU 验证通过后执行

从保留的 `model_16000.pt` 恢复到新 run，不接着扩大随机初始化的启动测试。
`train.py` 将 `--max_iterations` 原样传给已安装 Runner 的 `num_learning_iterations`；
先只读核对该服务器 fork 的 `learn/load`，确认是“额外更新次数”以及完整恢复行为：

```bash
python - <<'PY'
import inspect
from instinct_rl.runners import OnPolicyRunner
print(inspect.getsource(OnPolicyRunner.learn))
print(inspect.getsource(OnPolicyRunner.load))
PY
```

若确认是额外更新次数，16000 后再跑 14000 次约对应原计划的 30000；
实际边界/编号以该 fork 为准。若语义不同，按实际实现换算，不能直接执行下方预算。
以下恢复命令保留 2048 环境和原 PPO/AMP 配置，不会由本次修复自动启动：

```bash
cd /workspace/instinctlab
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Visual-Foothold-Amp-G1-v0 \
  --motion_root /workspace/instinctlab/data/grail_instinctlab \
  --author_motion_root /workspace/instinctlab/data/hiking_in_the_wild/parkour_motion_reference \
  --amp_grail_fraction 0.75 \
  --resume --load_run "$RUN" --checkpoint "$CHECKPOINT" \
  --headless --num_envs 2048 --max_iterations 14000 \
  --device "$TRAIN_DEVICE" --seed 42 --run_name visual_chain_resume_verified \
  agent.device="$TRAIN_DEVICE"
```

建议先把额外更新次数限制为 1000，检查新增 raw 指标及冻结回放，再延长预算。
本项目仍每 50 次日志、每 1000 次 checkpoint，并保留训练结束显式保存。
如果 motion inventory 检查或完整恢复失败，不绕过检查；保留错误并用新 run 从零训练。
