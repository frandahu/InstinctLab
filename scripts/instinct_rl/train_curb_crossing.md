# G1 GRAIL curb 跨步微调

任务：`Instinct-Parkour-Curb-Crossing-Amp-G1-v0`；独立日志目录：
`logs/instinct_rl/g1_curb_crossing`。现有 Mixed/Stairs 任务、数据与 checkpoint 保留。

目标是从一个平台跨步到下一个平台，短间隙内没有低处地面触地。
允许另一只脚继续支撑在前一平台；不要求双脚腾空。

## 数据准备

默认读取已转换的 `data/grail_instinctlab/curb/*_retargeted.npz`，沿用现有
torso-root、wxyz 和 G1 29 关节约定。不会下载数据、重新 retarget 或修改原始文件。

正常训练命令会自动做一次 CPU 筛选：用现有足端 FK 找低速、交替、抬高的支撑候选，
截取 0.40–0.70 m 支撑点间距的跨步窗口，排除窗口内低处支撑、明显侧向步及大高差。
生成的 NPZ、清单和原始文件哈希/帧区间保存在 `outputs/curb_crossing_references/`，后续验证后复用。
筛选使用的是支撑点间距，不是地形 gap。

**筛选结果是运动学候选，不是经过地形/接触验证的两平台跨越标签。** 原始 NPZ 不带平台几何与
接触标签，同一高平台上的大步仍可能进入候选；检查 `curation.json` 的源片段与视频，
可用人工审核清单覆盖默认筛选。实际跨越约束由下面的训练地形、触地状态和奖励提供。
没有合格候选会报错，不回退到作者步行数据或全部 curb 数据。

可先独立运行（无需 Isaac Sim/GPU）：

```bash
python scripts/instinct_rl/prepare_curb_crossing_motions.py \
  --source_root data/grail_instinctlab \
  --output_root outputs/curb_crossing_review
```

使用审核后的清单：训练命令加
`--motion_root /包含curb子目录的根目录 --motion_selection /清单绝对路径/crossing.yaml`。
YAML 使用 `selected_files` 和 `motion_weights`；仅允许 `curb/` 下的 G1 NPZ。
更改源数据或 URDF 后使用新的输出目录，防止悄悄覆盖既有参考。

## 训练目标与课程

- 8 个等级，短间隙从约 0.18 m 逐步扩到 0.30 m；平台高 0.12–0.20 m，长 0.80–1.10 m。
- 每条 curb 路线含 5 个平台、3 段短间隙和 1 段 0.85 m 长间隙，长间隙允许经平地。
  每四列保留一列纯平地，维持基础行走。
- 平台支撑须超过 20 N 向上接触力，满足旋转后足底包络和 0.025 m 边距，持续至少 0.06 s。
  从前一平台到相邻下一平台的干净跨越，每段每回合仅奖励一次，事件奖励为 2。
- 短间隙低处地面触地按每足每秒 -4 惩罚，并使该段本回合不能算干净跨越。
  普通平地、长间隙、平台前后上下台不受这个惩罚。
- 课程晋级要求所有短间隙干净跨越、走到终点并双足承重/直立/低速维持 0.5 s。
  路线通过但短间隙踩地不会晋级。
- AMP 系数保持 0.25，`feet_air_time=0.75`，其他 PPO、网络和基础奖励沿用 Mixed 配方。
  不增加 actor 观测或改变动作接口；特权几何只用于训练目标。

新课程使用水平步长工程上限 0.60 m：0.30 m gap 加足底裕度/步宽需约 0.572 m。
这不是动态可行性保证，也不是强行放大机器人的动作。可将
`env.scene.terrain.terrain_generator.gap_range=[0.18,0.26]`
作为初始更窄课程。扩跨度必须同时满足 `max_step_m` 的几何条件。

## 从 model_24000 权重开始

更换参考分布时使用 `--warm_start`，不使用 `--resume`。它严格加载兼容的 actor/value 和
normalizer 权重；AMP 判别器、优化器、回放缓冲区和迭代编号重新开始，记录源 checkpoint SHA256。
同一新任务后续断点续训才使用 `--resume`，并校验参考文件哈希及采样权重一致。

先执行 32 环境、5 次迭代启动检查；将 `DEVICE` 改成实际空闲设备：

```bash
RUN=/workspace/instinctlab/logs/instinct_rl/g1_parkour_mixed/20261008_063454_author_walk_air075_resume_gpu1_device-cuda:0_from20261008_033821
DEVICE=cuda:0
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Curb-Crossing-Amp-G1-v0 \
  --warm_start "$RUN/model_24000.pt" \
  --headless --device "$DEVICE" --num_envs 32 --seed 42 \
  --max_iterations 5 --run_name grail_curb_startup agent.device="$DEVICE"
```

确认 AMP/PPO 更新、有限值和最终 checkpoint 保存后，重新从该旧权重启动正式微调：

```bash
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Curb-Crossing-Amp-G1-v0 \
  --warm_start "$RUN/model_24000.pt" \
  --headless --device "$DEVICE" --num_envs 1024 --seed 42 \
  --max_iterations 5000 --run_name grail_curb_finetune agent.device="$DEVICE"
```

训练预算与奖励系数是待验证的实验起点，不保证 5000 次内学会 0.30 m 跨越。
观察 `Episode/Command/base_velocity/curb_clean_crossings`、
`curb_ground_touched_short_gaps`、`curb_route_success`（实际前缀取决于服务器 runner），
结合课程等级、速度误差、物理失败和基础平地表现选 checkpoint。
这些任务计数来自名义足底包络和接触启发式，不是 CoP、无滑移或稳定性证明。

## 固定条件评估

用已有足端 demo，指定新任务和新运行目录：

```bash
python -u scripts/foot_state_demo/run_demo.py \
  --task Instinct-Parkour-Curb-Crossing-Amp-G1-v0 \
  --load_run /workspace/instinctlab/logs/instinct_rl/g1_curb_crossing/实际运行目录 \
  --checkpoint model_5000.pt --device "$DEVICE" --headless --seed 42 \
  --terrain_mode curb --num_envs 4 --episodes_per_env 1 \
  --curb_count 5 --curb_height_range 0.12 0.20 --curb_depth_range 0.80 1.10 \
  --curb_gaps 0.30 0.85 0.26 1.00 --max_step_m 0.60 \
  --episode_length_s 120 --max_steps 6000 --video --video_view overview \
  --output_dir outputs/foot_state_demo/curb_crossing_seed42
```

`--max_step_m 0.60` 只让诊断参考与训练课程使用同一工程包络，不控制 actor。
评估删除训练专用奖励/课程/状态，使用原有独立触地记录；路线 success 与
`bridge_short_gap/matched` 分别报告，不能把路线成功等同于跨越成功。
比较旧/新模型时固定同一地形、seed、速度和参考参数；另测纯平地并使用多个 seed。

## 本地检查边界

`python -m unittest discover -s scripts/instinct_rl -p test_curb_crossing.py -v`
检查接触状态、奖励防重复、reset 隔离、课程和权重初始化。
CPU 检查与数据筛选不代表 Isaac Sim 5.1/GPU 启动或跨越效果已验证。
