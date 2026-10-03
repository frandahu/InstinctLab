# 从头训练 G1 深度相机楼梯策略

新任务：`Instinct-Parkour-Stairs-Amp-G1-v1`。日志目录：`logs/instinct_rl/g1_stairs_v1`。

现有视频/诊断说明旧模型的动作均值没有可靠步态，加入随机采样后能前进但严重抖动。它们没有单独证明 PPO 的优化公式有错误，也不能仅用 `is_alive` 与速度奖励的权重大小确定原因。这次保留深度相机、观测历史、MoE 网络、动作缩放和 PPO/AMP，修正已检查的奖励机制，并采用独立训练配置。

## 修改依据

| 项目 | 原行为 | 新配置 |
|---|---|---|
| AMP 与任务奖励尺度 | 本仓库任务奖励乘 `step_dt`；上游 PPO 的辅助奖励直接逐步相加 | AMP 按 `amp_reward_rate * step_dt` 设置，默认 `0.25 * 0.02 = 0.005` |
| 抬脚奖励 | 单脚支撑的持续时间没有上限，可持续累计 | 只奖励落地完成的摆腿，单脚单次最多 0.20，并要求实际向前运动 |
| 不前进惩罚 | 多个速度阈值的离散计数 | 指令与实际速度的连续相对欠速，范围 0–2 |
| 前进/存活 | 前进权重 2、存活权重 3 | 前进 4（核宽度 0.35）、存活 0.5 |
| 动作与上身抖动 | 动作差分 -0.005，上身偏离 -0.004 | 动作差分 -0.02，上身偏离 -0.08，关节速度 -0.001 |
| 探索与更新 | 初始标准差 1，无标准差上界，熵系数 0.006，初始学习率 0.001 自适应 | 初始标准差 0.35，投影到 0.05–0.50，熵系数 0.001，固定学习率 0.0003 |
| 训练地形 | 混合粗糙地形、沟槽、箱体、楼梯等；初始最多第 5 级 | 8 级直行路线：第 0 级平地，随后名义台阶高 3–18 cm；全部从地面开始，先上后下 |
| 地形升级 | 速度跟踪分数可以在接近目标/零指令时增高 | 完成路线、双脚到最终地面且直立才升级；跌倒同时到终点不升级 |

这里的 AMP 缩放是明确统一时间尺度的新配置选择，不表示上游 `0.25` 参数本身违反 PPO。服务器可能使用修改过的 Instinct-RL；每次新训练会将实际算法类、源码路径、哈希，以及各层 `process_env_step` / `compute_auxiliary_reward` / `update` 实现存进 `params/stair_runtime.json` 和相邻 `.txt`，便于核对服务器实际实现。

每条楼梯有 6 级上行、1.2 m 顶部平台、6 级下行，宽 2 m。踏面名义深度 28–40 cm；从第 3 个难度行开始，每段楼梯随机选 2/6 的台阶改变高度和宽度，幅度最高 20%。高度异常最大约 21.6 cm，下降总高度与上升相等。每个环境从平地逐级学习，超时/失败降级；初始化重置不会降级。完成最高级后沿用 Isaac Lab 的随机重分配难度逻辑。

保留参考动作数据和 AMP 判别器训练。`dataset_exhausted` 使用原有的 `reset_without_notice=True`，只循环专家参考片段，不重置机器人；删除这一处理会让 AMP 读到过期片段。缩小初始关节/姿态扰动与摩擦、弹性随机化，先建立可用步态。旧任务默认配置没有被替换。

## 服务器执行

更新代码：

```bash
cd /workspace/instinctlab
git pull --ff-only origin grail-training
```

先让新代码跑完 5 次迭代，以确认服务器能启动、更新、保存模型；这一步不检验步态是否学会：

```bash
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Stairs-Amp-G1-v1 \
  --headless --num_envs 32 --max_iterations 5 \
  --device cuda:1 --seed 42 --run_name startup \
  agent.device=cuda:1
```

正常结束后，从头正式训练（不使用 `--resume`、`--load_run`、旧 checkpoint）：

```bash
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Stairs-Amp-G1-v1 \
  --headless --num_envs 1024 --max_iterations 10000 \
  --device cuda:1 --seed 42 --run_name fresh_stairs \
  agent.device=cuda:1
```

正式运行会重新初始化全部策略、价值网络、判别器和优化器；启动检查目录与正式训练目录分开。每 50 次迭代打印日志、每 500 次保存 checkpoint，正常结束仍显式保存最后一个模型。迭代数是预算，不是成功保证。

以后续实际生成的目录/模型名替换下面占位符，导出未知楼梯 MP4，默认使用动作均值：

```bash
python -u scripts/instinct_rl/eval_stairs.py \
  --task Instinct-Parkour-Stairs-Amp-G1-v1 \
  --load_run /workspace/instinctlab/logs/instinct_rl/g1_stairs_v1/实际运行目录 \
  --checkpoint model_实际迭代数.pt \
  --device cuda:0 --headless --num_envs 8 --episodes_per_env 3 \
  --stair_mode up_down --seed 20261003 \
  --output_dir outputs/stair_eval/stairs_v1
```

训练地形与验证生成器、种子、级数不同，仍需用验证结果判断泛化。没有 `--sample`；随机采样能移动不能代替确定性步态。显卡 0 需空闲，否则换到空闲卡或等训练结束。

检查新训练指标时，原来的读取脚本现在也会显示 `route_progress`、`terrain_level`、动作平滑和上身偏离奖励。课程等级长期接近 0 说明尚未学会完成平地路线，应停止扩展训练预算；等级上升也不能代替独立验证成功率。

## 已验证与限制

本地 CPU 测试检查实际三角网格的顶面高度、起终点/难度行布局、不规则比例、升降总高度相等、地面相对跌倒判断、换级后的地面查询、完成/跌倒优先级、站立不升级、落地奖励上限、欠速惩罚、噪声投影和反向传播，以及已有 MP4/验证代码回归。

本地没有 Isaac Sim，尚未运行服务器环境构建或 PPO/AMP GPU 训练，不能声称这些配置已训练成功。具体权重和噪声上限是基于当前失败表现制定的起始配置，仍需要新训练的确定性验证来确认。

参考实现：[Instinct-RL PPO](https://github.com/project-instinct/instinct_rl/blob/main/instinct_rl/algorithms/ppo.py)、[Wasabi AMP](https://github.com/project-instinct/instinct_rl/blob/main/instinct_rl/algorithms/wasabi.py)。
