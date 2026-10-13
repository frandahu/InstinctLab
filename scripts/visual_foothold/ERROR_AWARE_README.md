# 冻结策略的落脚误差与自适应补偿验证

本实现是 [ERROR_AWARE_PLAN.md](ERROR_AWARE_PLAN.md) 的先导实验：在已有
`Instinct-Parkour-Visual-Foothold-Amp-G1-v0` 上测量参考响应并验证历史误差反馈。
继续加载同一个 B 策略及其 normalizer，保持原 41 维落脚观测、29 维关节动作和保存配置。
不训练策略，不实现 H-LIP、CLF、新高层 RL 或 InEKF。

## 实现与坐标

`reference.py` 管理三个模式及事件账本，`perception.py` 执行原有可行候选集合内的投影，
`play.py` 冻结策略回放，`analyze_reference.py` 离线比较 CSV。

| 模式 | 参考提案 |
|---|---|
| `baseline`，默认 | 原规划落点，完全保留原候选排序 |
| `perturb` | 在原规划点上沿当前接触锚点 X 或 Y 加固定 ±2/±4 cm |
| `compensate` | 原规划点减去该环境、该脚的历史预测偏差，补偿提案向量范数最多 4 cm |

规划点 `planned`、修正提案 `requested`、投影后的最终落点 `applied` 分别记录。
策略的目标、目标误差及摆脚插值参考都使用 `applied`。投影保留足印覆盖、平面残差、坡度、
可达距离、前向/侧向范围、高差、方向限制和摆脚走廊观测等原有条件，不放宽阈值。
从同一候选集合中选择离提案近、优先同高的落点；等价候选优先保留原点。
摆脚 apex 根据最终落点的路径重算。

**当前网格为 4 cm。2 cm 提案在平地会与原点等距，被投影抹除。**
这属于投影分辨率限制：`correction_erased` 标记执行变化 ≤1 mm 的非零提案；
`projection_changed` 标记投影与提案相差 >1 mm；`correction_opposed` 标记执行变化与提案方向内积 ≤0，
且变化没有被完全抹除的情况。先用 ±4 cm 判断参考响应，±2 cm 用于量化投影限制。
这里的安全性仍是原局部视觉规则下的估计，不是整机安全证明。

实际触地点采用**首次候选接触时 URDF FK 的鞋底参考点**，不是独立的仿真世界真值。
确认接触后才完成测量，期间的脚移动不改写首次接触位置；候选取消不更新估计器。

```text
error_xyz = actual_first_contact_xyz - applied_xyz
bias[foot] = (1-alpha) * bias[foot] + alpha * error_heading_xy
requested_xy = planned_xy - bounded(predicted_bias_in_current_anchor_xy)
```

第一条有效样本直接初始化偏差，后续按默认 `alpha=0.2` 更新。
事件里的位置和 X/Y 有符号误差使用**首次接触时的锚点坐标系**，同一条记录内可直接相减。
计划 CSV 使用规划时锚点坐标系，触地 CSV 已将该命令各落点一起转换到首次接触坐标系；
不要跨 CSV 直接相减坐标。`anchor` 为左/右脚编号 0/1，`slot` 是原固定步钟编号，
不是“成功触地次数”；每次成功提交有独立 `reference_id`。

EWMA 使用重力对齐的躯干朝向 XY，以免换支撑脚后混用锚点轴。
规划时将预测偏差旋转到当前锚点轴，再投影。左右脚、各环境独立；每次 episode reset 清空历史。
只用已确认、有限且水平误差不超过 `--ewma_max_error_cm`（默认 50 cm）的标签更新。
默认每脚至少 3 条样本才补偿；参考生成在接触确认之后，因此同帧新提交的命令也可使用刚确认的历史误差。
未触地样本、取消样本和无可行参考均保留计数；没有误差样本时 JSON 为 `null`。

## 远程 Docker 验证命令

SSH 登录服务器后，进入现有容器；不需要重新构建或升级环境：

```bash
docker exec -it instinctlab bash
cd /workspace/instinctlab

# origin 应指向个人仓库 https://github.com/frandahu/InstinctLab.git。
git remote get-url origin
# 首次获取本功能分支。已有同名分支时，switch 会使用现有分支。
git fetch origin feat/visual-foothold-error-feedback
git switch feat/visual-foothold-error-feedback
git merge --ff-only origin/feat/visual-foothold-error-feedback
git rev-parse --short HEAD

python scripts/visual_foothold/test_core.py -v
python scripts/visual_foothold/test_reference.py -v
```

选择空闲 GPU；例如 `CUDA_VISIBLE_DEVICES=1` 后物理 GPU 1 映射为下面的逻辑 `cuda:0`。
将 `RUN` 和 `CKPT` 改成真实存在的 B 训练目录和模型文件：

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=1  # 按 nvidia-smi 结果选择实际空闲卡
RUN=/workspace/instinctlab/logs/instinct_rl/g1_parkour_visual_foothold/你的实际训练目录
CKPT=model_20000.pt
test -f "$RUN/$CKPT"
test -f "$RUN/params/env.yaml"
test -f "$RUN/params/agent.yaml"
OUT="outputs/visual_foothold/error_aware_$(date +%Y%m%d_%H%M%S)"

# 首先只验证 4 环境、500 步启动和记录；不训练、不保存新 checkpoint。
python -u scripts/visual_foothold/play.py \
  --load_run "$RUN" --checkpoint "$CKPT" --headless --device cuda:0 \
  --num_envs 4 --max_steps 500 --seed 42 --experiment_mode baseline \
  --output_dir "$OUT/smoke"
```

启动检查通过后，执行完整三模式对照。下面各组顺序运行，共用配置、模型、种子、环境数和预算：

```bash
COMMON=(--load_run "$RUN" --checkpoint "$CKPT" --headless --device cuda:0
        --num_envs 4 --max_steps 10000 --seed 42)

python -u scripts/visual_foothold/play.py "${COMMON[@]}" \
  --experiment_mode baseline --output_dir "$OUT/baseline"

for AXIS in x y; do
  for CM in -4 -2 2 4; do
    python -u scripts/visual_foothold/play.py "${COMMON[@]}" \
      --experiment_mode perturb --perturb_axis "$AXIS" --perturb_cm "$CM" \
      --output_dir "$OUT/perturb_${AXIS}_${CM}" || break 2
  done
done

# 先导实验从每脚首条有效标签起补偿，便于稀疏触地场景检查闭环是否激活。
# 改成 3 可观察默认 warmup 的影响；1 条样本的偏差预测可能很噪。
python -u scripts/visual_foothold/play.py "${COMMON[@]}" \
  --experiment_mode compensate --ewma_alpha 0.2 --ewma_min_samples 1 \
  --compensation_max_cm 4 --ewma_max_error_cm 50 \
  --output_dir "$OUT/compensate"

python scripts/visual_foothold/analyze_reference.py \
  "$OUT/baseline" "$OUT"/perturb_* "$OUT/compensate" \
  --output "$OUT/comparison.json"
```

回放目录已存在时程序会创建带编号的新目录，请按日志中的实际目录比较。
需要 MP4 时给单组回放加 `--video`；以上实验默认不录像。
若有效触地不足，先看样本数和参考覆盖；延长回放只能增加回合数，不会把不同 episode 的 EWMA 样本合并。

## 输出与判读

| 文件 | 内容 |
|---|---|
| `reference_plans.csv` | 每次有效提交的 planned/requested/applied、偏差与样本数、投影改变/抹除/反向标记 |
| `touchdown_events.csv` | 首次接触位置和时间、确认时间、XYZ 有符号误差、heading 误差、EWMA 更新结果；含 confirmed/aborted/reset_unconfirmed |
| `visual_trace.csv` | 原控制步轨迹和自动 reset 前状态，新增累加统计的步增量 |
| `summary.json` | 全部/左右脚误差均值、绝对误差、水平 P95/RMSE、样本分母、投影和未完成计数 |
| `params/visual_foothold_runtime.json` | 实验配置、坐标/测量定义、实现与 URDF 哈希 |
| `comparison.json` | 各组结果及同轴 ± 扰动的配对 slot 响应对照 |

先看 `corrections_requested`、`corrections_erased` 和实际 applied offset，确认干预到达策略输入。
随后看 `confirmed.samples`、左右脚分母、`horizontal_error_p95_m`、`signed_error_xy_mean_m`，
以及 termination、参考覆盖和 `commands_without_confirmed_touchdown`，避免因落脚失败样本减少而误判精度提升。
最后比较 baseline 与 compensate；补偿效果由数据决定，不能由 CPU 测试推断。

`response_gain_proxy` 是落点相对 planned 的位移在 applied 修正上的投影，包含原执行偏差。
`response_secant_proxy` 比较同种子、同环境/episode/slot/脚的 ± 扰动均值变化；
接近 1 表示观测到同向等幅响应，接近 0 表示响应弱，反号表示反向。
两次闭环轨迹及幸存触地样本可能已经不同，因此它们都是描述性诊断，不是因果辨识或稳定性证明。
没有有效干预或触地时输出 `null`。比较工具拒绝混用 checkpoint、保存配置、代码、种子、环境数或预算不同的运行。

本地完成 CPU 数学、时序、CSV 与离线分析测试及语法检查；Isaac Sim 5.1/Docker/GPU 回放由以上命令验证。
