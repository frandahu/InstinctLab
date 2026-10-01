# 未知楼梯验证

`eval_stairs.py` 加载冻结的 G1 Parkour checkpoint，只做推理，默认以 **headless** 模式运行，并在输出目录直接生成 **`staircase.mp4`**。服务器无需图形桌面，脚本自动启用离屏渲染。沿用训练保存的深度相机、本体观测历史、动作缩放和网络配置。验证场景使用直线楼梯，训练场景包含的金字塔楼梯仍留在训练配置中。

默认 8 条独立楼梯，4 条上楼、4 条下楼；每条有 8 级，阶高在 0.08–0.20 m、踏面深度在 0.25–0.40 m 内采样，宽度 2 m。每条楼梯运行 5 次，共 40 次。每条楼梯内部默认等高等深；`--irregular` 改为每一级独立采样。重复试验保留同一几何，只对初始位置和朝向施加小扰动。

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
  --episodes_per_env 5 \
  --seed 42 \
  --output_dir outputs/stair_eval/model2000_seed42
```

仿真和策略推理都使用 `--device` 指定的卡，不需要额外传 `agent.device`。checkpoint 先读到 CPU，再将策略参数复制到验证设备；不恢复优化器或 AMP 判别器，避免原 checkpoint 中的 GPU 1 张量额外占用训练卡。输出目录必须不存在，避免覆盖已有结果。此命令测试原来的 `model_2000.pt`。续训会创建新的日志目录；测试续训 checkpoint 时，把 `--load_run` 换成训练输出的实际新目录，`--checkpoint` 换成已完整保存的确切文件名。

服务器原训练 Python 环境需要 FFmpeg 编码依赖，缺失时安装一次：

```bash
python -m pip install "imageio[ffmpeg]"
```

先做短启动检查时，可以用 `--num_envs 2 --episodes_per_env 1 --max_steps 100`，会录制约 2 秒 MP4。100 步通常不足以走完整段楼梯；结果会记录为尚未完成，不能据此判断成功率。

对不等高楼梯验证，在上面的命令追加 `--irregular`，同时更换输出目录。单独上楼/下楼可传 `--stair_mode up` / `--stair_mode down`；单个环境必须选择其中一种。

只观察一条楼梯时，可以这样运行（默认已启用视频）：

```bash
python scripts/instinct_rl/eval_stairs.py \
  --load_run /workspace/instinctlab/logs/instinct_rl/g1_parkour/20260930_050456 \
  --checkpoint model_2000.pt \
  --device cuda:0 --headless \
  --num_envs 1 --stair_mode up --episodes_per_env 1 \
  --output_dir outputs/stair_eval/model2000_up_video
```

视频路径为 `outputs/stair_eval/model2000_up_video/staircase.mp4`，使用固定侧面视角，覆盖所选楼梯和终点平台。默认 H.264 编码、1280×720，在当前 0.02 s 控制周期下每两步录一帧，播放速度为 25 FPS，与仿真时间一致。逐帧编码，不把整段 RGB 图像留在内存中；正常结束、中断或出错时都会关闭编码器并保存已录制片段，不会尝试打开播放器。

默认记录整个验证过程。`--video_length 1250` 限制只录制前 1250 个仿真步（当前约 25 秒），验证统计仍继续。`--video_env_id 1` 可观察默认混合测试中的下楼通道；`--video_stride` 控制录制间隔，`--video_width` / `--video_height` 控制分辨率。`--no_video` 可关闭 RGB 渲染和编码，仅输出统计。

策略的深度输入仍使用训练时的射线深度相机，64×36 是当前仓库默认值，实际取自保存的训练配置。MP4 使用独立的观察者相机，不改变策略输入。

## 判定与结果

- **成功**：机器人靠近终点（水平距离小于 0.35 m），两个脚踝均越过最后一级 0.1 m，并处于平台范围内、距离平台高度小于 0.20 m；至少一只脚的当前接触力超过 5 N，机身倾角小于 0.5 rad，根节点高于当地支撑面 0.5 m，连续保持 0.5 s。
- **失败**：躯干接触、训练原有的姿态终止、根节点距当地地面低于 0.5 m、走出楼梯范围或状态非有限值。根节点高度使用当前位置的楼梯高度计算，避免高平台上的跌倒被绝对高度掩盖。
- **超时**：25 s 内未满足成功条件。失败与成功同一步出现时，按失败记录。
- 测量在物理步之后、环境自动重置之前进行，因此失败时的坐标不会被初始坐标覆盖。

每次运行保存：

| 文件 | 内容 |
|---|---|
| `summary.json` | 总体、上楼、下楼成功率及失败原因，是否完整运行 |
| `episodes.csv` | 每次试验的结果、耗时、最大路径进度、侧向偏离和速度跟踪误差 |
| `trace.csv` | 每步位置、脚踝位置、接触力、姿态、速度指令及终止标记 |
| `cases.json` | 完整楼梯尺寸、seed、checkpoint 路径和 SHA256、配置来源 |
| `eval_env.yaml` / `eval_agent.yaml` | 实际验证使用的配置 |
| `staircase.mp4` | 默认输出的离屏渲染视频，H.264 编码 |

`success_rate_completed` 只对已经结束的试验计算；`success_rate_planned` 用全部计划试验作分母。只在 `status=complete`、`uncompleted_episodes=0` 时报告完整测试成功率。中断或步数上限留下的试验不会被当作超时/成功。`summary.json` 的 `video` 字段记录 MP4 路径、实际帧数、FPS、观察通道和编码错误（如有）。

`max_progress_fraction` 是根节点的最大向前距离除以终点距离，不代表稳定踏上了对应比例的台阶。速度跟踪误差是在当前机身坐标系中实际水平速度与该步速度指令的欧氏距离。

比较 `model_2000` 与续训模型时，保持 seed、环境数量、楼梯参数、速度、时限、传感器和噪声设置相同。每条楼梯的重置扰动按各自 seed 与重置次数独立生成，CSV 记录实际初始扰动，便于核对。相机/观测噪声保留训练流程；不同策略导致重置时刻变化，因此不能假设噪声样本逐步完全相同。先比较成功率/终止原因，再在相同环境、相同试验编号的共同时间区间比较轨迹；不要直接比较长短不同的整段轨迹均值。使用多个 seed 增加楼梯实例，重复同一楼梯的 5 次试验不能视为 5 个不同楼梯。

默认读取 `--load_run/params/env.yaml` 和 `agent.yaml`。缺失时脚本会报错；只有确认当前代码配置与训练一致时才用 `--use_current_cfg`。AMP 参考传感器保留用于现有 WasabiPPO/runner 的构造，仍需能访问训练使用的 GRAIL 数据；参考动作不进入 actor 观测，也不用于验证成功判定。这里保留训练相机/观测噪声，关闭材质随机化和间歇推力，先测楼梯几何行走能力。

## 无仿真预览

可在没有 Isaac Sim 的电脑上导出测试楼梯清单，检查尺寸与 seed：

```bash
python scripts/instinct_rl/eval_stairs.py \
  --load_run unused --checkpoint model_2000.pt \
  --dry_run --irregular \
  --output_dir outputs/stair_eval/case_preview
```

预览只生成 `cases.json`，不代表模型或 GPU 验证通过。
