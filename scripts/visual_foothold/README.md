# 深度落脚参考：从零训练的 A/B 实验

B 任务：`Instinct-Parkour-Visual-Foothold-Amp-G1-v0`。
A 任务：`Instinct-Parkour-Visual-Foothold-Baseline-Amp-G1-v0`。

实现独立保存在
`source/instinctlab/instinctlab/tasks/parkour/config/g1/visual_foothold/`。
现有 Mixed、foot_state_demo、Curb Crossing 任务不变。继续使用现有
`scripts/instinct_rl/train.py`，保留 cuDNN bootstrap、数据准备和最终 checkpoint 保存。

## B 实现了什么

1. 使用同一个躯干相机，读取米制 `distance_to_image_plane`，裁剪与原策略一致的
   下半部中央 32×18 区域。使用项目实际的相机局部射线和 +X 光轴恢复点云，
   不读取 `ray_hits_w`、地形标签、网格、case 文件或机器人世界位姿。
2. 几何支路在高斯模糊前处理，增加 5 mm 标准差深度噪声，保留有效性掩码；
   最大量程 2.5 m 的饱和值为未知。几何支路与 actor 图像都延迟一个控制步。
   A 也使用固定一帧延迟；原 Mixed 的随机 0–1 帧延迟不变。
3. GPU 批量 URDF FK 包含腰部祖先关节，鞋底偏移 `(0.039, 0, -0.058)` m。
   FK 输入增加独立的关节测量噪声。速度是身体相对坐标的导数，不是世界滑移速度。
4. 20/10 N 接触迟滞、20 ms 接触/离地确认、10 Hz 速度低通。
   接触输入当前来自仿真的踝部向上合力；实机需要相应接触测量或经过验证的接触估计。
5. 支撑脚锚定、最多 0.6 s 的点云记忆，每 0.1 s 纳入一帧已延迟图像。
   在双支撑且足间相对运动一致时转换锚点；失去支撑、明显脚部转动或无法可靠转换时清空。
   地图是短时静止支撑近似，不是全局定位，也不能可靠检测单支撑滑移。
6. 4 cm 网格分离“可支撑表面”和“观测障碍最高点”。只填补至少四个同高邻点支持的
   单格空洞，大块未知区域保持未知。以保守 28×20 cm 足印窗口检查覆盖、平面残差和坡度。
   目标可达水平距离不超过 0.60 m，高差不超过 0.24 m。
7. 每 0.4 s 交替一只摆动脚，在前 31% 步相内选取并锁定目标。
   几何上可达的近距离同高台面优先直接跨越；远台面超出范围时先选择已观测地面。
   这里没有 curb 真值分类器。几何可达不保证机器人能执行。
8. 增加 41 维低维观测：双足位置、滤波速度、接触状态、支撑有效性、目标、误差、摆动脚、
   步相、目标/锚点有效性、摆动参考、重力方向和观测年龄。MoE 深度编码器、PPO/AMP、
   29 关节位置控制和原奖励保留。

初版主要针对前向行走和适度转向：目标方向相对当前锚点偏转超过 20 度时不生成有效目标。
起步时脚下区域可能尚未被相机观测；此时目标无效，仍由原 Mixed/AMP 奖励引导起步，
不会补入地形真值或盲目生成“安全”落脚点。观测不足时新奖励被屏蔽，但原速度任务继续有效。

## 奖励与数据解释

| 项目 | 初始权重 | 计算口径 |
|---|---:|---|
| `visual_touchdown` | +0.5 | 一次确认落脚计一次，使用首次候选接触时锁定的三维误差 |
| `visual_swing` | +2.0 | 摆动参考跟踪的连续奖励率 |
| `visual_edge` | −0.5 | 有效指令落脚时，视觉足印窗口的风险事件 |
| `visual_miss` | −0.25 | 已有有效目标却未完成的步周期 |

事件项返回值除以 `step_dt`，抵消本项目 MultiRewardManager 的 `*dt` 积分。
这些是初始实验权重，没有完成 GPU 训练或性能调参。
“视觉风险”是局部估计，并不是实际接触压力、完整支撑或摔倒标签。
摆动参考的净空来自观测到的障碍，不能保证未知区域没有障碍。
原环境奖励仍可使用仿真传感器/真值进行训练评分，但这些数据不流入视觉目标生成器。

日志新增 `VisualFoothold/` 指标：

- `valid_target_fraction`、`support_fraction`：有效目标与可用锚点覆盖率；
- `touchdowns`、`commanded_touchdowns`：全部/有有效指令的落脚数；
- `touchdown_error_m`、`touchdown_hit_5cm`：有有效指令时的 FK 落脚误差；
- `visual_risk_fraction`、`missed_steps`、`anchor_losses`。

没有有效指令时，误差指标的零值不代表准确，应同时看 `commanded_touchdowns`。
有效目标覆盖率长期接近零时不能把训练称为成功接入落脚学习，应先查深度覆盖与锚点有效性。
参数、运动数据清单和实现 SHA256 保存在运行的 `params/` 中，新增
`visual_foothold_runtime.json`。固定种子还不保证 CUDA 位级确定性。

## 执行：服务器原有 Isaac Sim 5.1 / Instinct-RL 环境

先将本次代码同步到服务器工作区；下面命令本身不会同步代码。
在包含这些改动的 `/workspace/instinctlab` 中执行。`cuda:0` 必须是实际空闲的逻辑 GPU。

```bash
cd /workspace/instinctlab
TRAIN_DEVICE=cuda:0

# B 的小规模启动检查：不是正式训练，也不恢复旧模型
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Visual-Foothold-Amp-G1-v0 \
  --headless --num_envs 32 --max_iterations 5 \
  --device "$TRAIN_DEVICE" --seed 42 --run_name visual_foothold_startup \
  agent.device="$TRAIN_DEVICE" agent.resume=False
```

检查 `[VISUAL]`、41 维新增观测、有限的 PPO/AMP 更新、运行记录与最终 checkpoint。
五次迭代只验证程序启动。通过后另开正式随机初始化运行：

```bash
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Visual-Foothold-Amp-G1-v0 \
  --headless --num_envs 1024 --max_iterations 30000 \
  --device "$TRAIN_DEVICE" --seed 42 --run_name visual_foothold_B_fresh \
  agent.device="$TRAIN_DEVICE" agent.resume=False
```

B 日志目录：`logs/instinct_rl/g1_parkour_visual_foothold/`。
不加 `--resume`、`--load_run` 或 `--warm_start`，不加载 model_24000。
默认继续使用 Mixed 的作者步行 AMP 数据；如需 GRAIL，A/B 都显式传入相同的
`--motion_root` 和 `--motion_selection`，核对清单哈希。

空闲资源允许时，另起 A 组，保持环境数、种子、迭代预算和运动数据一致：

```bash
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Visual-Foothold-Baseline-Amp-G1-v0 \
  --headless --num_envs 1024 --max_iterations 30000 \
  --device "$TRAIN_DEVICE" --seed 42 --run_name visual_foothold_A_fresh \
  agent.device="$TRAIN_DEVICE" agent.resume=False
```

A 日志目录：`logs/instinct_rl/g1_parkour_visual_baseline/`。
两组保持相同地形生成种子、原 Mixed 课程和控制配置；学习后的课程等级轨迹可能不同，
所以最终必须在相同固定几何上比较。B 增加了视觉几何处理、接触输入和结构化参考，
该对照衡量组合方法，不单独归因于 FK 或滤波。

## B 的确定性播放、MP4 和 CSV

下面使用训练地形恢复保存配置，直接加载冻结 actor，不启动 AMP 判别器或优化器。
替换 `RUN` 为实际 B 运行目录，checkpoint 必须存在。它不是原 terrain_eval 独立路线成功率测试。

```bash
RUN=/workspace/instinctlab/logs/instinct_rl/g1_parkour_visual_foothold/实际运行目录
python -u scripts/visual_foothold/play.py \
  --load_run "$RUN" --checkpoint model_30000.pt \
  --headless --device "$TRAIN_DEVICE" --num_envs 4 \
  --max_steps 1500 --seed 42 --video \
  --output_dir outputs/visual_foothold/B_seed42
```

生成 `visual_foothold.mp4`、`visual_trace.csv`、`summary.json` 和实际播放配置。
视频跟随 env0，CSV 记录所有环境的自动重置前状态，包括终止帧。
`status=complete` 仅表示完成指定播放步数，不能解释为路线全部成功。
视频帧在自动重置后渲染，个别终止时刻会与 CSV 的重置前姿态不同。
MP4 复用已有 FFmpeg 录制器，未自动安装依赖。当前独立多地形评估器有旧 actor 观测白名单，
不要直接用旧 eval_mixed_parkour.py 评估 B；后续固定路线适配应保留这个新观测接口。

## 本地 CPU 检查

```bash
python scripts/visual_foothold/test_core.py -v
```

需要已有 Torch。覆盖项目光轴/深度单位、未知区域与楼梯边界拒绝、斜坡、近远 curb 选择、
与原 NumPy FK 的独立对照、接触确认、摆动前目标锁定、支撑丢失以及环境重置隔离。
这些检查不启动 Isaac Sim，不证明 CUDA 性能、相机/管理器运行兼容性或训练效果。
