# G1 深度相机多地形训练

任务：`Instinct-Parkour-Mixed-Amp-G1-v0`；日志：`logs/instinct_rl/g1_parkour_mixed`。
同一个步行策略同时学习低起伏地面、楼梯、台阶边缘、斜坡、箱体及间隙。

## 作者的步行参考已经接入

默认使用作者项目页 [Data & Model](https://project-instinct.github.io/hiking-in-the-wild/) 公开的
`parkour_motion_without_run_retargetted.npz` 和原始 `parkour_motion_without_run.yaml`。
实测为 **18982 帧、50 Hz、379.62 秒**（首末帧时间差），包含全部 29 个 G1 关节。
[论文 III-E](https://arxiv.org/html/2601.07718v1#S3.SS5) 说明步行参考由离线 MPC 合成步行和
NOKOV 人体动捕经 GMR 重定向组合而成；跑步使用另一套数据和独立策略。
公开 NPZ 没有逐帧来源标签，因此不额外猜测 MPC/动捕比例，也不自行拆分或修改动作。

这次补齐的是作者发布的、包含 MPC 步行的 **AMP 参考数据**。训练和验证仍由 RL 策略输出关节目标，
无需运行 ROS/OCS2，也没有在机器人动作后面串联在线 MPC 控制器。
之前默认使用的 GRAIL curb/slope/stair 数据不再自动参与本任务。
GRAIL 数据仍保留，可通过显式参数用于对照实验。

环境、奖励、速度指令、控制参数、传感器历史、随机化和地形课程直接继承 `G1ParkourEnvCfg`；
算法继承 `G1ParkourPPORunnerCfg`。AMP 系数为原版 **0.25**，不再额外乘 `step_dt`。
不继承 `Stairs-v1` 的奖励改写或噪声上限。

| 地形 | 原版比例 |
|---|---:|
| 低起伏粗糙地面 | 5% |
| 粗糙地面站立任务 | 5% |
| 间隙及其边缘 | 10% |
| 普通上行 / 下行楼梯 | 各 15% |
| 高台阶上行 / 下行 | 各 10% |
| 离散箱体 / 随机箱体边缘 | 各 10% |
| 斜坡 | 10% |

训练持续保留各种地形，不将所有环境先后切换成纯平地、纯楼梯。
低起伏地面不是专门的纯平地训练；基础步态须另用纯平地验证。

## 数据准备：首次自动，之后离线复用

直接运行下面的训练命令时，默认数据若缺失会自动下载作者的 11.9 MB ZIP。
脚本只提取原始 NPZ、YAML 和说明文件到 `data/hiking_in_the_wild/parkour_motion_reference/`，
验证固定 SHA256、G1 关节映射、数组形状、有限数值和单位四元数。
下载内容与 ZIP 提取内容均校验，不覆盖已经被修改的同名文件。
模型和部署说明不参与本次训练；二进制数据不提交到 Git。

也可以在启动 Isaac Sim 前单独准备并检查（不需要 GPU）：

```bash
cd /workspace/instinctlab
python scripts/instinct_rl/prepare_parkour_references.py
python scripts/instinct_rl/check_parkour_motions.py
```

服务器无法访问 Google Drive 时，在能访问的电脑从项目页下载 Data & Model ZIP，传到服务器：

```bash
python scripts/instinct_rl/prepare_parkour_references.py \
  --archive /实际路径/hiking-in-the-wild_DataModel.zip
```

安装成功后不再要求联网。如果上游数据包变更，校验会明确报错，避免悄悄换掉实验输入。
`provenance.json` 记录来源、数据校验值和时长；每次训练的 `params/motion_inventory.json`
记录实际清单、每个文件的 SHA256 和初始采样权重。
这些检查验证数据兼容性，不能代替仿真接触验证或策略收敛验证。

## 启动检查与正式训练

先用空闲 GPU 跑 5 次迭代，确认环境、AMP/PPO 更新及模型保存可用：

```bash
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Mixed-Amp-G1-v0 \
  --headless --num_envs 32 --max_iterations 5 \
  --device cuda:1 --seed 42 --run_name author_walk_startup \
  agent.device=cuda:1
```

应看到 `[REFERENCES] Verified 18982 frames, 379.62 s`，以及 `[MIXED]` 打印的作者数据路径和 YAML。
一个 NPZ 包含整套步行动作，`motions=1` 不表示只使用一种地形。
确认实际服务器的 `params/parkour_runtime.json` 中 AMP 系数为 0.25。
5 次迭代只验证启动，不能判断步态。

通过后重新从头正式训练，使用独立运行目录：

```bash
python -u scripts/instinct_rl/train.py \
  --task Instinct-Parkour-Mixed-Amp-G1-v0 \
  --headless --num_envs 1024 --max_iterations 30000 \
  --device cuda:1 --seed 42 --run_name author_walk_fresh \
  agent.device=cuda:1
```

这次不加 `--resume`，不加载之前失败模型的策略、判别器或优化器。
原版预算是 30000 次迭代，每 1000 次和正常结束时保存模型。
预算不是成功保证。建议训练期间定期用同一条件验证确定性策略的平地起步，再判断楼梯能力。

## 确定性验证与 MP4

先对新 checkpoint 做纯平地验证；下面的运行目录和模型文件须替换成实际保存的文件：

```bash
python -u scripts/instinct_rl/eval_stairs.py \
  --task Instinct-Parkour-Mixed-Amp-G1-v0 \
  --load_run /workspace/instinctlab/logs/instinct_rl/g1_parkour_mixed/实际运行目录 \
  --checkpoint model_实际迭代数.pt \
  --device cuda:0 --headless --num_envs 1 --episodes_per_env 1 \
  --terrain_mode flat --seed 20261006 \
  --output_dir outputs/stair_eval/author_walk_flat
```

平地起步稳定后，同一命令去掉 `--terrain_mode flat`、增加 `--stair_mode up_down`，
输出目录改为 `outputs/stair_eval/author_walk_stairs`。
默认输出 MP4 并使用动作均值，不加 `--sample`。
验证读取保存的传感器与策略配置，替换独立测试地形，不加载 AMP 动作数据或训练 runner。
平地和楼梯通过后，还需要独立测试斜坡、箱体边缘和间隙。

## 自选数据对照

自定义动作须显式指定根目录和清单；原版加载器使用 `motion_weights`，不是 `weights`：

```bash
--motion_root /实际/G1动作根目录 \
--motion_selection /实际/selection.yaml
```

也支持 `INSTINCTLAB_PARKOUR_MOTION_ROOT` / `INSTINCTLAB_PARKOUR_MOTION_SELECTION`。
只显式指定 GRAIL 根目录而没有清单时，使用 curb、slope、stair_p1、stair_p2 四个子集，缺失即报错。
旧的 `INSTINCTLAB_GRAIL_MOTION_ROOT` 不会改变本任务默认作者数据选择。
恢复训练时必须保持原运行的动作输入一致，不能把数据切换伪装成完全相同的断点续训。
代码会核对原运行的文件哈希与采样权重；旧运行未记录哈希或输入有变化时，拒绝按原样断点续训。

本地验证包括实际作者数据下载、解压、校验及 CPU 回归；尚未完成 Isaac Sim/GPU 训练和步态收敛验证。
