# G1 足端状态独立 Demo

所有新实现都在本目录。现有训练入口、任务文件、奖励、策略输入和已有评估器源码均不修改；不加载优化器或 AMP 数据集，不训练新模型。

本版本实现**足端状态获取与记录**，尚未生成期望落脚目标，也未接入落脚跟踪奖励或跨步固定参考系。

## 文件

| 文件 | 作用 |
|---|---|
| `foot_state.py` | 纯 NumPy 批量 FK、解析足端速度、一阶低通、每足四状态接触状态机 |
| `sim_snapshot.py` | 仿真专用重置前采样，继承原成功项并原样返回成功判定 |
| `demo_output.py` | 足端轨迹、触地／离地事件、统计和观察者视频面板 |
| `run_demo.py` | 独立仿真入口，复用已有冻结 actor 加载、配置恢复、地形与 MP4 工具 |
| `offline_demo.py` | 合成测量输入检查，不需要 Isaac Sim、PyTorch 或 checkpoint |
| `test_foot_state.py` | CPU 坐标、速度、事件和重置契约检查 |

## 服务器运行：现有模型 + 双机器人 + MP4

在原 Isaac Sim 5.1 / Instinct-RL 训练 Python 环境、仓库根目录执行。`RUN` 沿用之前的模型目录；若模型已移动，使用实际包含 checkpoint 与 `params/` 的目录。`EVAL_DEVICE` 根据当前空闲的逻辑 GPU 设置。

```bash
cd /workspace/instinctlab
RUN=/workspace/instinctlab/logs/instinct_rl/g1_parkour_mixed/20261008_063454_author_walk_air075_resume_gpu1_device-cuda:0_from20261008_033821
EVAL_DEVICE=cuda:0

python -u scripts/foot_state_demo/run_demo.py \
  --load_run "$RUN" --checkpoint model_24000.pt \
  --device "$EVAL_DEVICE" --headless \
  --num_envs 2 --episodes_per_env 1 --max_steps 1500 \
  --terrain_mode mixed --video_view lane --video_env_id 0 \
  --output_dir outputs/foot_state_demo/model24000_mixed_seed42
```

默认最多 1500 个控制步，当前配置下约 30 秒仿真；`step_limit` 表示演示达到步数上限，不能当成路线失败。达到所有计划回合后会提前正常结束。

- 首次只跑约 4 秒：改成 `--num_envs 1 --max_steps 200`，使用新的输出目录。
- 查看全部机器人：加 `--video_view overview`；单通道更适合看足端细节。
- 复用已生成的 hard 地形协议：加 `--difficulty hard`；同时显式给 `--num_envs 2` 保持 demo 规模。
- 只记录数据：加 `--no_video`。
- 完整单回合：设 `--max_steps 3000 --episode_length_s 60`。这不是训练命令。

MP4 沿用项目的 FFmpeg 编码器，文字面板使用 Pillow；缺少依赖会明确报错，不自动安装或升级依赖。需要时在原仿真环境安装 `imageio[ffmpeg]` 与 `Pillow`。

## 输出

- `foot_state.mp4`：机器人视频，加双足状态、机体坐标下的位置／速度及 XY 小图；左脚绿色、右脚青色。只修改输出 RGB，不向场景增加物体，不影响策略深度图。
- `foot_trace.csv`：每个控制步、每个有效环境、每只脚一行；包括终止前的末帧。
- `foot_events.csv`：确认的触地／离地事件、首次候选时间和确认延迟；触地另存首次候选时相对另一只脚的坐标及其参考有效性。
- `foot_summary.json`：运行状态、完成回合及终止原因、FK 位置误差、事件数、CPU FK＋观测器时间。
- `cases.json`、`eval_env.yaml`、`eval_agent.yaml`、`startup_status.json`、`startup.log`：配置、几何、checkpoint/URDF/config 哈希及启动诊断。

输出目录已存在时自动创建编号兄弟目录，不覆盖旧结果。生成物统一写入 Git 已忽略的 `outputs/foot_state_demo/`。

## 测量和坐标约定

1. 仿真原型输入是关节角、关节速度和**模拟接触力的世界向上分量**。绝对连杆位姿只用于 FK 真值对照，不输入接触观测器或 actor。接触力可用性不能直接假设到实机。
2. FK 从当前评估配置的带鞋 URDF 读取根到双足的祖先链，包含腰部，按名称映射关节。没有引入 InEKF 或新学习模型，也没有读取 MotionReference 管理器内部状态。
3. 当前 URDF 根为 `torso_link`。足底采用名义坐标：踝部连杆位置加 `--sole_offset 0.039 0 -0.058` 米，方向与该连杆一致。这来自现有鞋底几何的约定，**不是实测压力中心**；换鞋或模型时必须同步更改。
4. `sole_*_b_m` 是相对躯干的脚掌坐标；速度是该坐标的时间导数，**不是世界足速**。支撑脚速度不会被强制置零。
5. `sole_*_other_foot_m` 是相对**当前另一只脚**的坐标；即便另一脚悬空也有几何意义，但不能当成有效固定支撑参考。`reference_foot=-1/0/1` 分别表示无可信参考／左／右。
6. 触地坐标在接触候选首次出现时锁存。`candidate_reference_valid=0` 时，不得用于支撑参考下的落脚误差。即使为 1，也没有证明另一脚无滑移或滚动。

FK 默认使用理想仿真关节测量，因此误差反映模型、顺序和坐标一致性，不代表实机估计精度。`--encoder_noise_std 0.005` 可单独给观测器增加 0.005 rad 的独立角度噪声，不改变 actor 输入；使用独立随机数生成器。

## 状态机与初始参数

`air → contact_candidate → support → release_candidate → air`，每足独立；短暂接触可以退回 air，短暂失力可以返回 support。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--velocity_cutoff_hz` | 10 Hz | 仅平滑足端速度，位置保留 FK 原值 |
| `--force_on_n` / `--force_off_n` | 20 / 10 N | 向上力进入／退出阈值 |
| `--confirm_on_s` / `--confirm_off_s` | 0.02 / 0.02 s | 首次候选后的持续确认时间 |

这些是 demo 初值，未经实机标定。50 Hz 下默认需要相邻两帧确认，约 20 ms；首次候选本身仍受一个控制周期的采样误差影响。初始站立建立支撑状态，但不会伪造“迈步触地”事件。环境自动重置后只清除对应环境的状态与滤波器。

`contact_score_heuristic` 只是力阈值映射，不是概率。向上力有助于排除纯侧撞，但不能识别全部台阶立面摩擦、局部脚尖接触或滑移；`support` 表示力门控确认，不能作为稳定平衡保证。

## 本地离线检查

```bash
python scripts/foot_state_demo/offline_demo.py \
  --steps 300 --num_envs 2 --output_dir outputs/foot_state_demo/offline

python -m unittest discover -s scripts/foot_state_demo -p test_foot_state.py -v

python scripts/foot_state_demo/run_demo.py --dry_run \
  --load_run example --checkpoint model_24000.pt \
  --output_dir outputs/foot_state_demo/config_preview
```

离线入口使用当前真实 URDF，但关节运动和接触力是合成输入；输出 CSV、JSON 和 `offline_preview.png`，不模拟机器人、不生成机器人 MP4。没有独立仿真真值时，FK 误差标为 `null`，不制造零误差结果。

首次仿真重点查看：FK 对照误差、触地事件位置是否在候选时锁存、终止／重置是否串帧，以及 MP4 面板是否正常。CPU 耗时统计不含 GPU→CPU 拷贝、CSV 和渲染；不能据此宣称整个系统开销或优于 InEKF。
