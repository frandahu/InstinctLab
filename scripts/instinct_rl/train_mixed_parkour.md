# G1 深度相机多地形训练

任务：`Instinct-Parkour-Mixed-Amp-G1-v0`，日志目录：`logs/instinct_rl/g1_parkour_mixed`。
同一个深度相机 MoE 策略在多种地形上共同训练；不为平地、楼梯或斜坡分别训练策略。

## 配置依据与区别

环境直接继承上游 `G1ParkourEnvCfg`，算法直接继承 `G1ParkourPPORunnerCfg`。
使用原始奖励、速度指令、控制参数、传感器历史、随机化和速度跟踪地形课程，
不继承 `Stairs-v1` 的路线课程、奖励改写或噪声上限。
AMP 系数保持 **0.25**，不再乘 `step_dt`；初始动作标准差 1.0、学习率 0.001、
adaptive 调度、熵系数 0.006、每轮 24 步、默认 30000 次迭代均来自原版。
保留本分支每 1000 次保存、每 50 次日志和正常结束时显式保存最终模型的便利设置。

| 地形 | 原版比例 |
|---|---:|
| 低起伏粗糙地面 | 5% |
| 粗糙地面站立任务 | 5% |
| 间隙及其边缘 | 10% |
| 普通上行 / 下行楼梯 | 各 15% |
| 高台阶上行 / 下行 | 各 10% |
| 离散箱体 / 随机箱体边缘 | 各 10% |
| 斜坡 | 10% |

10 行、20 列，初始最大难度 5，随后按原版速度跟踪得分升降级。
各地形类型在训练中持续存在，不采用全体环境先只练平地、再全体只练楼梯的切换。
低起伏地面不等同于专门的纯平地步行训练；需要用平地控制验证检查基础步态。

**参考数据与上游不同**：上游使用外部 `parkour_motion_without_run.yaml`，本仓库没有提供该清单。
默认复用现有 G1 重定向数据中的 `curb`、`slope`、`stair_p1`、`stair_p2` 四个子集。
本地分别有 500、500、250、250 条；服务器以检查脚本实际输出为准。
按照现有加载器逐文件等权采样，并保留原版动作课程机制；不按地形指定专家。
这些目录名不能证明动作中包含足够的平地步行，也不能证明动作重定向质量。
若步态仍异常，应先查看参考片段及确定性策略，不能把格式检查当成学习成功。

## 先检查服务器数据

```bash
cd /workspace/instinctlab
python scripts/instinct_rl/check_parkour_motions.py --scan
python scripts/instinct_rl/check_parkour_motions.py \
  --motion_root /workspace/instinctlab/data/grail_instinctlab
```

第二条检查全部所选 NPZ 的文件、G1 URDF 关节映射、数组维度、有限数值和单位四元数。
任何默认子集缺失都会明确报错，不会悄悄退回只用楼梯数据。
正常训练也会执行该检查，并保存 `params/motion_inventory.json`，列出实际文件与初始采样概率。
`--scan` 只检查已知数据目录和清单位置，不会遍历服务器所有磁盘。

若已获得经过检查的原版或自选 G1 动作清单，在训练和检查命令中增加：

```bash
--motion_root /实际/G1重定向动作根目录 \
--motion_selection /实际/parkour_motion_without_run.yaml
```

清单格式示例（路径与数量必须改成真实数据）：

```yaml
selected_files:
  - walking/example_retargeted.npz
  - slope/example_retargeted.npz
  - stair_p1/example_retargeted.npz
motion_weights: [1.0, 1.0, 1.0]
```

实际加载器读取的是 `motion_weights`，不是 `weights`。
清单覆盖默认四子集过滤；不会自动下载数据或生成未经审查的步行动作。
也可用 `INSTINCTLAB_PARKOUR_MOTION_ROOT` / `INSTINCTLAB_PARKOUR_MOTION_SELECTION` 固定路径。

## 启动检查与正式训练

先用空闲 GPU 跑 5 次迭代，确认环境构建、PPO 更新和模型保存。它不能验证行走质量：

```bash
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Mixed-Amp-G1-v0 \
  --headless --num_envs 32 --max_iterations 5 \
  --device cuda:1 --seed 42 --run_name startup \
  agent.device=cuda:1
```

启动检查通过后重新从头正式训练：

```bash
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Mixed-Amp-G1-v0 \
  --headless --num_envs 1024 --max_iterations 30000 \
  --device cuda:1 --seed 42 --run_name fresh_mixed \
  agent.device=cuda:1
```

不加载先前失败模型。新目录独立保存策略、判别器、优化器和实际服务器实现记录
`params/parkour_runtime.json`；配置保存在 `params/env.yaml`、`params/agent.yaml`。
30000 是上游预算，不是成功保证。四子集动作缓存比以前只有楼梯更大；显存不足时
先确认空闲卡并降低并行环境数，以实际启动结果决定规模。

## 确定性验证与 MP4

先使用同一个 checkpoint 做纯平地控制验证，再测试未知楼梯。
下面的 `实际运行目录`、`model_实际迭代数.pt` 必须替换为实际文件：

```bash
python -u scripts/instinct_rl/eval_stairs.py \
  --task Instinct-Parkour-Mixed-Amp-G1-v0 \
  --load_run /workspace/instinctlab/logs/instinct_rl/g1_parkour_mixed/实际运行目录 \
  --checkpoint model_实际迭代数.pt \
  --device cuda:0 --headless --num_envs 1 --episodes_per_env 1 \
  --terrain_mode flat --seed 20261006 \
  --output_dir outputs/stair_eval/mixed_flat
```

平地起步稳定后，在同一命令中去掉 `--terrain_mode flat`，增加 `--stair_mode up_down`，
输出目录改为 `outputs/stair_eval/mixed_stairs`。默认输出 MP4 并采用动作均值，不加 `--sample`。
验证读取保存的传感器和策略配置，替换为独立验证地形，不加载 AMP 动作数据或启动训练 runner。
这两个测试覆盖平地和楼梯，不能据此声称所有坡道、箱体和间隙都已通过独立验证。

本地仅能执行 CPU 数据检查、回归测试和语法检查，尚未完成 Isaac Sim/GPU 训练或收敛验证。

参考：[上游环境](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py)、
[上游策略与算法参数](https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py)。
