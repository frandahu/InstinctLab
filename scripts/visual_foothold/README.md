# 深度落脚参考：作者数据 + GRAIL 的单策略训练

冻结现有策略进行落脚误差、±2/±4 cm 参考扰动和 EWMA 补偿对照，见
[误差感知先导实验与 Docker 命令](ERROR_AWARE_README.md)。该入口不重新训练策略。

使用任务：`Instinct-Parkour-Visual-Foothold-Amp-G1-v0`。
一条训练命令只启动一个策略，不会自动启动 A/B 两组训练。
`Baseline` 任务仅保留为以后可选的对照入口，现在不需要运行。

实现独立保存在
`source/instinctlab/instinctlab/tasks/parkour/config/g1/visual_foothold/`。
现有 Mixed、foot_state_demo、Curb Crossing 任务不变。继续使用现有
`scripts/instinct_rl/train.py`，保留 cuDNN bootstrap、数据准备和最终 checkpoint 保存。

## 实现了什么

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
9. AMP 同时使用作者步行语料和已重定向的 G1 GRAIL 动作，初始片段抽样权重各占 50%。
   作者数据来自 `data/hiking_in_the_wild/parkour_motion_reference/` 的原始选择清单；
   GRAIL 来自 `data/grail_instinctlab/` 的 `curb`、`slope`、`stair_p1`、`stair_p2`。
   各来源内部保留原相对权重，GRAIL 默认逐片段等权。这是一个联合专家池、一个策略，
   不按地形切换，也不要求两类数据各贡献一半训练帧。50/50 是初始设置，尚未训练验证。
   原 `Instinct-Parkour-Mixed-Amp-G1-v0` 仍默认使用作者步行语料，避免改变原基线。

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

修复后的训练日志用 `VisualFoothold/raw/` 记录分子、分母和事件总量；误差均值不再填零。
同时记录 `touchdown_samples_available`，回放 JSON 在没有有效样本时输出 `null`（N/A）。
用聚合的 `raw/error_sum_m / raw/commanded_touchdowns` 计算误差，不能平均每回合误差。
完整修复说明、指标口径、验证门槛和历史 checkpoint 的服务器命令见
[Visual Foothold Chain 修复与验证](CHAIN_FIX.md)。以下首次训练示例仍使用默认 50/50 专家池；
历史 `visual_author_grail_fresh` 实际使用 75% GRAIL，恢复时必须按保存清单保持一致。
有效目标覆盖率长期接近零时不能把训练称为成功接入落脚学习，应先查深度覆盖与锚点有效性。
参数、运动数据清单和实现 SHA256 保存在运行的 `params/` 中，新增
`visual_foothold_runtime.json`。固定种子还不保证 CUDA 位级确定性。

## 执行：服务器原有 Isaac Sim 5.1 / Instinct-RL 环境

先将本次代码同步到服务器工作区；下面命令本身不会同步代码。
在包含这些改动的 `/workspace/instinctlab` 中执行。`cuda:0` 必须是实际空闲的逻辑 GPU。

```bash
cd /workspace/instinctlab
TRAIN_DEVICE=cuda:0

# 主任务的小规模启动检查：不是正式训练，也不恢复旧模型
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
  --device "$TRAIN_DEVICE" --seed 42 --run_name visual_author_grail_fresh \
  agent.device="$TRAIN_DEVICE" agent.resume=False
```

训练日志目录：`logs/instinct_rl/g1_parkour_visual_foothold/`。
不加 `--resume`、`--load_run` 或 `--warm_start`，不加载 model_24000。
默认同时加载作者数据和 GRAIL，二者都通过原有 AMP 判别器提供运动先验。
深度图负责落脚参考；没有按地形切换数据，也没有新增逐帧全身动作跟踪。

服务器需要已有转换结果 `data/grail_instinctlab/{curb,slope,stair_p1,stair_p2}/*_retargeted.npz`。
还需要原作者语料；它继续复用已有的固定哈希准备入口，已存在时不重复下载。
GRAIL 不会自动下载或重新转换；缺失任一必需子集时停止准备，不会只用作者数据继续。
不会删除、复制或改写两类源动作。只生成一个带 `motion_weights` 的联合清单，保存在
`outputs/visual_foothold/reference_manifests/`，然后交给原有动作加载器。
训练入口逐文件检查数据格式并记录哈希、实际清单和初始采样概率到
`params/motion_inventory.json`，启动时打印 `[VISUAL AMP]`。
本地作者数据为 1 个整合文件，GRAIL 为 1500 个文件，合计 1501 个文件。
抽样按来源归一化，作者占 50%，curb/slope 各约 16.67%，stair_p1/stair_p2 各约 8.33%。
文件数量不等于动作种类数量，抽样概率也不是训练帧比例。服务器实际数量以检查结果为准。

可先离线检查（不启动仿真）：

```bash
python scripts/instinct_rl/prepare_visual_motions.py --motion_root /workspace/instinctlab/data/grail_instinctlab
```

非默认 GRAIL 目录可通过 `--motion_root` 或 `INSTINCTLAB_GRAIL_MOTION_ROOT` 指定。
作者数据可通过 `--author_motion_root` 或 `INSTINCTLAB_AUTHOR_MOTION_ROOT` 指定，目录中
需有 `parkour_motion_without_run.yaml`。`--amp_grail_fraction` 可调整 GRAIL 初始抽样份额，
必须严格介于 0 与 1，保证两类来源都参与。
通用 `INSTINCTLAB_PARKOUR_MOTION_ROOT` / `INSTINCTLAB_PARKOUR_MOTION_SELECTION` 仍可覆盖默认值；
若之前设置过作者语料的环境变量，应先取消，或显式传入 GRAIL 的 `--motion_root`。
使用 GRAIL 自选清单时加 `--motion_selection`，它只改变 GRAIL 部分，作者数据仍参与。
YAML 权重键必须为 `motion_weights`。训练日志会打印 `sources={'author': 0.5, 'grail': 0.5}`，
`params/motion_inventory.json` 记录最终文件、来源、哈希及抽样权重。
多进程训练不分片专家池，避免唯一作者文件只分到一个进程；每进程都会加载联合专家池。

更换专家数据后从零训练，不恢复旧作者语料模型；已有恢复检查会拒绝数据哈希或权重不匹配。
此改动不会让旧 checkpoint 自动获得 GRAIL 经验。

如果以后确实要做对照，再手动使用 `Instinct-Parkour-Visual-Foothold-Baseline-Amp-G1-v0`。
该任务也使用相同的联合专家池，但没有新增足端参考观测和奖励；不会随主任务自动启动。
届时应保持相同数据、种子和训练预算，并在相同固定几何上比较。

## B 的复杂地形行走验证

使用原有 `scripts/foot_state_demo/run_demo.py`，显式指定
`--task Instinct-Parkour-Visual-Foothold-Amp-G1-v0`。它沿用固定路线、起终点、速度指令和足端诊断，
并保留 B 模型训练时的 actor 观测。下楼梯及混合路线命令见
[foot-state Demo](../foot_state_demo/README.md#visual-foothold-模型沿用同一个验证入口)。

## B 的训练地形诊断回放

以下 `play.py` 仅用于训练分布及参考干预诊断，会保留保存配置中的随机围墙和指令。
行走验证使用上一节的 foot-state Demo。下面恢复训练配置，直接加载冻结 actor，不启动 AMP 判别器或优化器。
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

联合专家池检查只需要已有 NumPy/PyYAML，不需要 Torch：

```bash
python scripts/visual_foothold/test_motion_pool.py -v
```

足端和感知算法检查：

```bash
python scripts/visual_foothold/test_core.py -v
```

需要已有 Torch。覆盖项目光轴/深度单位、未知区域与楼梯边界拒绝、斜坡、近远 curb 选择、
与原 NumPy FK 的独立对照、多候选共同可行性、实际摆脚走廊、历史覆盖与过期、接触确认、
确认期间锚点移动、跨步相结算、触地取消与反弹、摆动前目标锁定、支撑丢失以及环境重置隔离。
这些检查不启动 Isaac Sim，不证明 CUDA 性能、相机/管理器运行兼容性或训练效果。
