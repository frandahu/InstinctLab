# 未知楼梯验证

`eval_stairs.py` 加载冻结的 G1 Parkour checkpoint，只做推理，默认以 **headless** 模式运行，并在输出目录直接生成 **`staircase.mp4`**。服务器无需图形桌面，脚本自动启用离屏渲染。沿用训练保存的深度相机、本体观测历史、动作缩放和网络配置。验证场景使用直线楼梯，训练场景包含的金字塔楼梯仍留在训练配置中。

默认 8 条独立路线，每条都是 **上 8 级 → 顶部平台 → 下 8 级 → 地面终点**，顶部平台长 1.20 m，楼梯横向宽度 2 m。每组楼梯先采样一套基准尺寸：阶高 0.08–0.20 m、踏面深度 0.25–0.40 m。每段 8 级中随机选 **2 级**，改变阶高、踏面深度或两者，其余 6 级完全保持基准尺寸。这里的台阶“宽度”按行走方向的踏面深度理解，横向通道宽度保持一致。每条路线运行 5 次，共 40 次；重复试验保留同一几何，只对初始位置和朝向施加小扰动。

下楼段将上楼段的阶高变化重新分配到随机位置，使上升与下降总高度相等，最后回到地面；下楼的踏面变化另行采样。不是每一级都随机，也不会把上楼、下楼分到不同通道。

这里的“未知”是策略未得到楼梯真值、测试几何与训练的金字塔楼梯不同、随机尺寸由测试 seed 决定。默认阶高与训练区间有交集，不能把结果称为训练范围之外的尺寸外推。

## 服务器运行

当前训练在 GPU 1，下面让验证使用 GPU 0。先确认 GPU 0 有空余显存，并在仓库根目录、原训练 Python 环境中执行：

```bash
python scripts/instinct_rl/eval_stairs.py \
  --task Instinct-Parkour-Target-Amp-G1-v0 \
  --load_run /workspace/instinctlab/logs/instinct_rl/g1_parkour/20260930_050456 \
  --checkpoint model_2000.pt \
  --device cuda:0 \
  --headless \
  --num_envs 8 \
  --stair_mode up_down \
  --episodes_per_env 5 \
  --seed 42 \
  --output_dir outputs/stair_eval/model2000_up_down_seed42
```

仿真和策略推理都使用 `--device` 指定的卡，不需要额外传 `agent.device`。checkpoint 先读到 CPU，再将策略参数复制到验证设备；不恢复优化器或 AMP 判别器，避免原 checkpoint 中的 GPU 1 张量额外占用训练卡。若输出目录已存在，脚本会自动创建同级编号目录，例如 `model2000_up_down_seed42_001`、`_002`，保留旧文件。实际路径会在启动时打印，并写入 `cases.json` 的 `output_dir`；不传 `--output_dir` 时使用自动时间戳目录。此命令测试原来的 `model_2000.pt`。续训会创建新的日志目录；测试续训 checkpoint 时，把 `--load_run` 换成训练输出的实际新目录，`--checkpoint` 换成已完整保存的确切文件名。

服务器原训练 Python 环境需要 FFmpeg 编码依赖，缺失时安装一次：

```bash
python -m pip install "imageio[ffmpeg]"
```

先做短启动检查时，可以用 `--num_envs 2 --episodes_per_env 1 --max_steps 100`，会录制约 2 秒 MP4。100 步通常不足以走完整段楼梯；结果会记录为尚未完成，不能据此判断成功率。

默认已经启用少量异常台阶，`--irregular` 可显式启用，`--regular` 用于全等高等深的对照验证。`--irregular_fraction 0.20` 控制每段异常台阶比例，数量向上取整并限制为严格少于一半，默认 8 级中有 2 级；`--dimension_variation 0.25` 控制相对基准尺寸的最大变化量，实际在 12.5%–25% 内采样并裁剪到尺寸区间。两个尺寸区间不能同时固定，否则无法生成异常台阶。`--num_steps` 是每段级数，默认 `up_down` 共 16 级；异常台阶模式每段至少 3 级。`--landing_depth` 控制顶部平台长度，最小 0.8 m。

单独上楼/下楼仍可传 `--stair_mode up` / `--stair_mode down`，`mixed` 保留交替分配上楼、下楼通道的诊断模式。单环境支持 `up_down`、`up` 或 `down`。

只观察一条楼梯时，可以这样运行（默认已启用视频）：

```bash
python scripts/instinct_rl/eval_stairs.py \
  --load_run /workspace/instinctlab/logs/instinct_rl/g1_parkour/20260930_050456 \
  --checkpoint model_2000.pt \
  --device cuda:0 --headless \
  --num_envs 1 --stair_mode up_down --episodes_per_env 1 \
  --output_dir outputs/stair_eval/model2000_up_down_video
```

视频路径为 `outputs/stair_eval/model2000_up_down_video/staircase.mp4`，使用固定侧面视角，覆盖所选路线的上楼、顶部平台、下楼及终点平台，相机高度按楼梯最高点设置。默认 H.264 编码、1280×720，在当前 0.02 s 控制周期下每两步录一帧，播放速度为 25 FPS，与仿真时间一致。逐帧编码，不把整段 RGB 图像留在内存中；正常结束、中断或出错时都会关闭编码器并保存已录制片段，不会尝试打开播放器。

默认记录整个验证过程。`--video_length 2250` 限制只录制前 2250 个仿真步（当前约 45 秒），验证统计仍继续。`--video_env_id 1` 可观察第二条路线；`--video_stride` 控制录制间隔，`--video_width` / `--video_height` 控制分辨率。`--no_video` 可关闭 RGB 渲染和编码，仅输出统计。

策略的深度输入仍使用训练时的射线深度相机，64×36 是当前仓库默认值，实际取自保存的训练配置。MP4 使用独立的观察者相机，不改变策略输入。

## 判定与结果

- **顶部检查点**：`up_down` 必须先踏上顶部平台。两个脚踝都在平台内部，距顶部高度小于 0.20 m，根节点位于平台上方；至少一只脚的当前接触力超过 5 N，机身倾角小于 0.5 rad，根节点高于当地支撑面 0.5 m，才记录 `summit_reached=true`。每次重置清除该标记，空中经过平台不会通过检查。
- **成功**：通过顶部检查点之后，机器人靠近下楼后的地面终点（水平距离小于 0.35 m），两个脚踝均越过最后一级 0.1 m，并处于终点平台范围内、距离平台高度小于 0.20 m；至少一只脚的当前接触力超过 5 N，机身倾角小于 0.5 rad，根节点高于当地支撑面 0.5 m，连续保持 0.5 s。单独上楼/下楼模式不要求顶部检查点。
- **失败**：躯干接触、训练原有的姿态终止、根节点距当地地面低于 0.5 m、走出楼梯范围或状态非有限值。根节点高度使用当前位置的楼梯高度计算，避免高平台上的跌倒被绝对高度掩盖。
- **超时**：默认 45 s 内未满足成功条件，可用 `--episode_length_s` 调整。失败与成功同一步出现时，按失败记录。
- 测量在物理步之后、环境自动重置之前进行，因此失败时的坐标不会被初始坐标覆盖。

每次运行保存：

| 文件 | 内容 |
|---|---|
| `summary.json` | 总体、各路线类型成功率及失败原因、通过顶部检查点的试验数，是否完整运行 |
| `episodes.csv` | 每次试验的结果、耗时、顶部检查点状态、最大路径进度、侧向偏离和速度跟踪误差 |
| `trace.csv` | 每步位置、脚踝位置、接触力、姿态、速度指令、顶部检查点及终止标记 |
| `cases.json` | 完整楼梯尺寸、基准尺寸、每段异常台阶索引（从 0 起）、顶部平台范围、seed、checkpoint 路径和 SHA256、配置来源 |
| `eval_env.yaml` / `eval_agent.yaml` | 实际验证使用的配置 |
| `staircase.mp4` | 默认输出的离屏渲染视频，H.264 编码 |
| `startup_status.json` / `startup.log` | 启动与运行阶段、进程号、最后进入的步骤；提前退出时帮助定位 |
| `startup_error.txt` | Python 异常完整堆栈（发生异常时生成，在关闭仿真器前写入） |

`success_rate_completed` 只对已经结束的试验计算；`success_rate_planned` 用全部计划试验作分母。只在 `status=complete`、`uncompleted_episodes=0` 时报告完整测试成功率。中断或步数上限留下的试验不会被当作超时/成功。`summary.json` 的 `video` 字段记录 MP4 路径、实际帧数、FPS、观察通道和编码错误（如有）。

`max_progress_fraction` 是根节点的最大向前距离除以终点距离，不代表稳定踏上了对应比例的台阶。速度跟踪误差是在当前机身坐标系中实际水平速度与该步速度指令的欧氏距离。

比较 `model_2000` 与续训模型时，保持 seed、环境数量、楼梯参数、速度、时限、传感器和噪声设置相同。每条楼梯的重置扰动按各自 seed 与重置次数独立生成，CSV 记录实际初始扰动，便于核对。相机/观测噪声保留训练流程；不同策略导致重置时刻变化，因此不能假设噪声样本逐步完全相同。先比较成功率/终止原因，再在相同环境、相同试验编号的共同时间区间比较轨迹；不要直接比较长短不同的整段轨迹均值。使用多个 seed 增加楼梯实例，重复同一楼梯的 5 次试验不能视为 5 个不同楼梯。

默认读取 `--load_run/params/env.yaml` 和 `agent.yaml`，以恢复 checkpoint 对应的相机、动作、actor 观测及网络配置。缺失时脚本会报错；只有确认当前代码配置与训练一致时才用 `--use_current_cfg`。验证时直接构建冻结 actor，移除场景中的动作参考和两个 AMP 观测组，因此**不读取 GRAIL/AMASS 训练动作文件**，也不构建 WasabiPPO 判别器。`base_lin_vel` 只在 critic 观测中，不是 actor 输入。这里保留训练相机/观测噪声，关闭材质随机化和间歇推力，先测楼梯几何行走能力。

验证脚本使用独立的 YAML 读取器兼容保存配置中的 `!!python/object/apply:builtins.slice`（例如关节/连杆选择的 `slice(None)`），并保留 tuple 等配置值。遇到这类 `ConstructorError` 时，更新本仓库即可；无需编辑旧训练 YAML 或重新训练。该兼容处理不修改 PyYAML 的全局读取器。

恢复环境配置时，脚本递归初始化已有字段中默认 `None`、但保存值为数值或其他普通配置值的可选字段（包括 seed、地形尺度、相机噪声上下限等），再调用 Isaac Lab 的 `from_dict()`。这兼容训练初始化后写入的值与旧版严格类型检查之间的差异；未知字段与已有非空字段的类型错误仍由原配置恢复函数检查，不把嵌套配置对象替换为普通字典。随后验证用 `--seed` 设置测试随机种子，默认 42。

保存的 `/scene/terrain/terrain_generator` 不参与恢复，因为验证随后会明确替换为上楼—平台—下楼生成器。深度相机、观测、动作、机器人与物理参数继续读取保存配置。`cases.json` 的 `saved_config_restore` 记录初始化过的可选字段路径及跳过的训练地形路径，保存的原 YAML 不会被修改。

## 没有生成 MP4 时

如果报错是 `render_first_frame` 阶段的 `TypeError: Unable to write from unknown dtype, kind=f, size=0`，且堆栈经过 `SyntheticData.py` 的 `dep_attrib_data.set(dep_data)`，检查 NumPy 与 Isaac Sim 的二进制接口兼容性。Isaac Lab 2.3 / Isaac Sim 4.5–5.1 要求 NumPy `<2`；仅跑策略和射线深度相机可能没有触发这条 RGB 渲染路径。

评测脚本现在会在导入 AppLauncher/PyTorch 之前选择 NumPy：已有 1.x 时保持使用；检测到 2.x 时优先加载 Isaac Sim `pip_prebundle` 中适配当前 Python 的 1.26。如果未找到，自动用 `pip --no-deps --target` 将 `numpy==1.26.4` 缓存到 `outputs/stair_eval/runtime_deps/`，后续运行复用缓存。首次缓存需要网络；它不会修改训练环境的 site-packages，也不会把整个 Isaac Sim pip archive 加到 Python 搜索路径。启动日志打印实际 NumPy 版本和路径，`cases.json` 的 `numpy_runtime` 保存同样的信息。正常更新后直接使用原评测命令，无需在公共环境里降级 NumPy。

终端必须先出现 `[INFO] Off-screen MP4: ...`，才表示已成功编码第一帧。如果日志停在配置解析、环境创建或策略加载阶段，MP4 录制尚未启动。检查输出目录里的 `startup_status.json` 与 `startup_error.txt`，不要把初始化退出当成验证完成。阶段日志会立即刷新，Python 异常在仿真器关闭前保存和打印，避免关闭过程提前结束解释器而丢失原始报错。原生崩溃的 Python 栈写入 stderr；SIGKILL 无法由 Python 捕获，需要结合 shell 退出码定位。

在服务器的 Bash 中运行下面的诊断命令（新输出目录，保留完整 stdout/stderr）：

如果指定目录发生冲突，读取诊断文件时应改用终端 `[INFO] Results directory:` 打印的实际编号目录。

```bash
eval_dir="outputs/stair_eval/up_down_model6000_$(date +%Y%m%d_%H%M%S)"
python -u scripts/instinct_rl/eval_stairs.py \
  --load_run /workspace/instinctlab/logs/instinct_rl/g1_parkour/20261001_074348_from20260930_050456 \
  --checkpoint model_6000.pt \
  --device cuda:0 --headless \
  --num_envs 1 --episodes_per_env 1 --stair_mode up_down \
  --output_dir "$eval_dir" 2>&1 | tee "${eval_dir}_console.log"
eval_exit=${PIPESTATUS[0]}
echo "Python exit code: $eval_exit"
echo "Output directory: $eval_dir"
tail -n 60 "${eval_dir}_console.log"
cat "$eval_dir/startup_status.json"
if [ -f "$eval_dir/startup_error.txt" ]; then
  cat "$eval_dir/startup_error.txt"
fi
```

`startup_status.json` 的 `returned` 仅表示 Python 验证函数已返回；是否完成全部试验仍看 `summary.json`。如果原生进程被结束，状态可能保留在 `running` 和最后进入的阶段。不要为了跳过配置报错直接加 `--use_current_cfg`，它可能改变与 checkpoint 对应的观测或网络配置。

## 视频中机器人几乎不动时

先看 `trace.csv`：`episode_step`、身体/脚部位置、接触力是否变化，`command_x_m_s` 是否非零。真实状态基本不前进时，单凭视频无法区分训练结果与推理/执行配置问题；`status=complete` 表示试验结束，`timeout` 表示未在时限内完成，并不表示机器人走成功了。

用同一 checkpoint、速度、传感器、动作配置和路线长度做平地对照。`--terrain_mode flat` 将楼梯高度全部归零，每条路线用一整块平地网格；保持起点扰动种子和目标位置，关闭顶部检查点要求。它不加载训练动作数据、不修改权重。例子最多运行 1000 步（当前配置为 20 秒），提前走到终点则提前结束：

```bash
python -u scripts/instinct_rl/eval_stairs.py \
  --load_run /workspace/instinctlab/logs/instinct_rl/g1_parkour/20261001_074348_from20260930_050456 \
  --checkpoint model_6000.pt \
  --device cuda:0 --headless \
  --num_envs 1 --episodes_per_env 1 \
  --terrain_mode flat --max_steps 1000 \
  --output_dir outputs/stair_eval/flat_model6000
```

每 250 步打印 `[DIAG]`；同样的数值写入 `trace.csv` 的新增列：

| 列 | 含义 |
|---|---|
| `policy_action_rms` / `policy_action_delta_rms` | 本步网络输出及与上一步输出之差的均方根；动作单位沿用训练配置，未经动作缩放 |
| `observed_command_x_min_m_s` / `observed_command_x_max_m_s` | 实际 actor 输入中全部指令历史帧的前进速度范围，归一化器之前；最初几步历史中可能含零 |
| `depth_input_min` / `depth_input_max` / `depth_input_mean` | 实际 actor 深度输入经过相机处理后的范围/均值，不是原始距离，单位取决于训练配置 |
| `joint_velocity_rms_rad_s` / `applied_torque_rms_nm` | 全部关节的实际速度/施加力矩均方根，物理步后、自动重置前采样 |

平地能走、楼梯停步时，优先检查新楼梯场景的感知输入和策略泛化。平地仍停步时，先检查 actor 是否确实收到非零指令、深度输入是否合理、动作是否变化、关节是否响应；再与原训练地形的播放结果及训练速度跟踪指标对照。动作小、动作近似恒定或深度近似恒定都不能单独证明模型坏了；站立平衡也可能输出非零动作。若只有训练时带随机动作噪声才能移动，也不能据此认定确定性策略已经学会行走。

平地短测结束时若 `status=step_limit`，表示没有在短测预算内结束试验，不能当作完整的 45 秒超时结果或完整楼梯成功率。

### 自动读取训练指标

不必手动搜索训练终端或打开 TensorBoard 网页。在服务器仓库根目录运行：

```bash
python scripts/instinct_rl/inspect_parkour_training.py \
  --load_run /workspace/instinctlab/logs/instinct_rl/g1_parkour/20261001_074348_from20260930_050456 \
  --checkpoint model_6000.pt
```

脚本只读该目录的 `events.out.tfevents.*`、`params/env.yaml` 和指定 checkpoint。它不启动 Isaac Sim、不读取动作数据集、不初始化 CUDA；TensorBoard 使用自身的无 TensorFlow 兼容路径。输出 `[ACTION_STD]`、保存的奖励权重，以及速度跟踪、等待惩罚、脚部腾空、存活、动作噪声、回合长度等日志中已有指标。默认只显示 checkpoint 迭代之前每个指标的最后 3 个值；`--at_iteration` 可修改截止点，省略 checkpoint 时看最新指标。找不到日志或指标会明确报告，不能用保存的参数代替实际学习结果。

奖励日志已经包含权重，且 `/sum`、`/timestep`、`/max_episode_len_s` 使用不同时间归一化，不能直接当作速度或混合比较。读取脚本排除 `Train/time/...` 等以运行耗时为横轴的指标，避免将耗时错误标记为 checkpoint 迭代。缺少 TensorBoard 日志时，终端中打印的训练指标也可用于检查。

### 采样动作对照

验证默认使用 `act_inference` 的动作均值。`--sample` 改为从同一 checkpoint 的策略分布采样，沿用训练保存的动作标准差；相机、观测归一化、动作缩放及权重不变。启动时打印动作模式和 checkpoint 的标准差，`summary.json` 记录 `action_mode`，方便区分结果。

在前面的平地命令后加 `--sample`，并使用新输出目录（例如 `outputs/stair_eval/flat_model6000_sample`）。该选项只用于诊断训练采样与确定性验证的差异。若只有采样时移动，仍需检查脚步、速度跟踪和原训练地形，随机抖动、滑动或跌倒不代表学会行走；默认楼梯成功率继续使用确定性策略。

## 无仿真预览

可在没有 Isaac Sim 的电脑上导出测试楼梯清单，检查尺寸与 seed：

```bash
python scripts/instinct_rl/eval_stairs.py \
  --load_run unused --checkpoint model_2000.pt \
  --dry_run --stair_mode up_down \
  --output_dir outputs/stair_eval/case_preview
```

预览只生成 `cases.json`，不代表模型或 GPU 验证通过。
