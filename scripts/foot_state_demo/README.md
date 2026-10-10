# G1 足端状态独立 Demo

所有新实现都在本目录。现有训练入口、任务文件、奖励、策略输入和已有评估器源码均不修改；不加载优化器或 AMP 数据集，不训练新模型。

V2 实现**足端状态、冻结落脚参考、触地偏差与踩边几何诊断**。参考在摆动开始时生成，采用固定的通道／世界坐标，不随之后的支撑脚移动而改变。

当前 Mixed actor 仍只接收原有观测和速度命令，**不会接收或执行这些落脚参考**。这里的误差是实际落脚相对几何建议的偏差，不能报告为已下发指令的跟踪误差。下一阶段的参考条件控制／训练需单独接入。

## 文件

| 文件 | 作用 |
|---|---|
| `foot_state.py` | 纯 NumPy 批量 FK、解析足端速度、一阶低通、每足四状态接触状态机 |
| `sim_snapshot.py` | 仿真专用重置前采样，继承原成功项并原样返回成功判定 |
| `demo_output.py` | 足端轨迹、触地／离地事件、统计和观察者视频面板 |
| `run_demo.py` | 独立仿真入口，复用已有冻结 actor 加载、配置恢复、地形与 MP4 工具 |
| `offline_demo.py` | 合成测量输入检查，不需要 Isaac Sim、PyTorch 或 checkpoint |
| `test_foot_state.py` | CPU 坐标、速度、事件和重置契约检查 |
| `foothold_reference.py` | 独立连续 curb 布局、几何落脚参考、足底投影与踏面裕度 |
| `foothold_output.py` | 冻结目标、候选触地误差、踩边证据、路线规则和视频文字 |
| `test_foothold_reference.py` | CPU 近远 curb、不可达目标、脚跟跨边、候选锁存与参考失效检查 |

## 服务器运行：现有模型 + 双机器人 + MP4

在原 Isaac Sim 5.1 / Instinct-RL 训练 Python 环境、仓库根目录执行。`RUN` 沿用之前的模型目录；若模型已移动，使用实际包含 checkpoint 与 `params/` 的目录。`EVAL_DEVICE` 根据当前空闲的逻辑 GPU 设置。

```bash
cd /workspace/instinctlab
RUN=/workspace/instinctlab/logs/instinct_rl/g1_parkour_mixed/20261008_063454_author_walk_air075_resume_gpu1_device-cuda:0_from20261008_033821
EVAL_DEVICE=cuda:0

python -u scripts/foot_state_demo/run_demo.py \
  --load_run "$RUN" --checkpoint model_24000.pt \
  --device "$EVAL_DEVICE" --headless \
  --num_envs 2 --episodes_per_env 1 --max_steps 6000 --episode_length_s 120 \
  --terrain_mode mixed --video_view lane --video_env_id 0 \
  --output_dir outputs/foot_state_demo/model24000_mixed_seed42
```

默认最多 6000 个控制步，当前配置下约 120 秒仿真；`step_limit` 表示演示达到步数上限，不能当成路线失败。达到所有计划回合后会提前正常结束。默认 curb 数量为 5，近远间距交替出现。

- 首次只跑约 4 秒：改成 `--num_envs 1 --max_steps 200`，使用新的输出目录。
- 查看全部机器人：加 `--video_view overview`；单通道更适合看足端细节。
- 复用已生成的 hard 地形协议：加 `--difficulty hard`；同时显式给 `--num_envs 2` 保持 demo 规模。
- 只记录数据：加 `--no_video`。
- hard 混合路线更长，可设 `--max_steps 9000 --episode_length_s 180`。这些均是冻结策略评估命令。

MP4 沿用项目的 FFmpeg 编码器，文字面板使用 Pillow；缺少依赖会明确报错，不自动安装或升级依赖。需要时在原仿真环境安装 `imageio[ffmpeg]` 与 `Pillow`。

## 输出

- `foot_state.mp4`：机器人视频，加双足状态、机体坐标下的位置／速度及 XY 小图；左脚绿色、右脚青色。只修改输出 RGB，不向场景增加物体，不影响策略深度图。
- `foot_trace.csv`：每个控制步、每个有效环境、每只脚一行；包括终止前的末帧。
- `foot_events.csv`：确认的触地／离地事件、首次候选时间和确认延迟；触地另存首次候选时相对另一只脚的坐标及其参考有效性。
- `foot_summary.json`：运行状态、完成回合及终止原因、FK 位置误差、事件数、CPU FK＋观测器时间。
- `foothold_targets.csv`：首次生成的目标、生成时间、支撑脚、路线意图；不可达目标会给出原因。
- `foothold_trace.csv`：每个控制步的通道坐标、足底旋转矩阵、四角、踏面编号、边缘裕度及当前目标。
- `foothold_events.csv`：确认触地后写入**首次候选时刻**的实际位置、预先生成的目标和 XYZ 偏差。`global_step/time_s/episode_step` 对应候选帧，`confirmation_global_step/confirmation_time_s` 对应确认帧。
- `foothold_summary.json`、`foothold_report.md`：分阶段统计及最需关注的下楼梯触地时间／脚／裕度。`foot_summary.json` 同时包含该摘要。
- `cases.json`、`eval_env.yaml`、`eval_agent.yaml`、`startup_status.json`、`startup.log`：配置、几何、checkpoint/URDF/config 哈希及启动诊断。

输出目录已存在时自动创建编号兄弟目录，不覆盖旧结果。生成物统一写入 Git 已忽略的 `outputs/foot_state_demo/`。

## 测量和坐标约定

1. 足端观测器输入仍是关节角、关节速度和**模拟接触力的世界向上分量**，绝对位姿不输入观测器或 actor。独立参考／踩边诊断分支使用真实躯干世界位姿和生成地形几何，把 FK 足端注册到固定通道坐标；这属于仿真特权信息，不是已实现的实机定位。接触力可用性不能直接假设到实机。
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

## 连续 curb：近跨、远经平地

沿用上文的 `RUN` 与 `EVAL_DEVICE`，在服务器仓库根目录执行：

```bash
python -u scripts/foot_state_demo/run_demo.py \
  --load_run "$RUN" --checkpoint model_24000.pt \
  --device "$EVAL_DEVICE" --headless \
  --terrain_mode curb --num_envs 4 --episodes_per_env 1 \
  --curb_count 5 --curb_height_range 0.12 0.20 --curb_depth_range 0.80 1.10 \
  --curb_gaps 0.18 0.85 0.22 1.00 \
  --episode_length_s 120 --max_steps 6000 --video_view overview \
  --output_dir outputs/foot_state_demo/model24000_curb_reference_seed42
```

默认相邻平台间隙依次为 **0.18、0.85、0.22、1.00 m**，第五个平台后留 1 m 平地。增加平台数量时循环使用间距列表；每个机器人保留自己的 seeded 高度／深度，记录全部机器人，MP4 可选择单通道或总览。

- `--bridge_gap_m 0.30`：短间隙参考跳过低处平地，直接选择下一平台。靠近边缘前先在当前平台前进；最大水平步长（包含左右脚间距）0.55 m、高差 0.24 m，且足底包络保留 0.025 m 裕度。
- 较远间隙保留平地支撑面，参考先下平台、在平地走，再上下一平台。
- 几何检查不通过时记录 `no_reachable_safe_foothold`，不能用“近”作为保证可跨越的依据。这些上限是 Demo 工程参数，尚非动态可行性结论。
- `cases.json` 保存每段间隙的 `curb_reference_rules`，包含最小安全中心步长及跨越可达性。`curb_rule_counts` 根据确认触地事件检查相邻平台之间是否出现平地触地，展示冻结策略是否自然满足规则；几何上不可跨的短间隙记为 `bridge_unavailable/ungraded`。没有记录平地触地不等于独立证明腾空或无接触。
- 若需沿用评估器的随机间距，使用 `--curb_layout sampled --curb_gap_range 0.18 1.00`。默认交替布局只作用于本 Demo，训练地形和已有评估器不变。

## 下楼梯踩边诊断

```bash
python -u scripts/foot_state_demo/run_demo.py \
  --load_run "$RUN" --checkpoint model_24000.pt \
  --device "$EVAL_DEVICE" --headless \
  --terrain_mode stairs --stair_mode down --num_steps 8 \
  --num_envs 2 --episodes_per_env 1 --episode_length_s 90 --max_steps 4500 \
  --video_view lane --video_env_id 0 \
  --output_dir outputs/foot_state_demo/model24000_downstairs_reference_seed42
```

也可用 `--terrain_mode mixed` 在连续路线中同时记录上下楼梯、斜坡和五个 curb。

先看 `foothold_summary.json → by_phase → stairs_down`，再看 `foothold_report.md` 对应候选时间的视频。报告记录全部机器人；视频单通道只覆盖所选 `env_id`。

- **`edge_margin_m < 0`**：名义足底包络投影越出对应踏面，标记 `overhang`。
- **`0 ≤ edge_margin_m < 0.025`**：未越出，但边缘裕度偏小，标记 `near_edge`。
- `loaded_edge` 同时要求向上力达到接触阈值及足底接近候选踏面；不会把普通空中摆脚直接计为踩边。
- 判定考虑完整足底朝向及与踏面相交的区域。即使脚掌中心越过边界、脚跟仍接近上一级，也会记录上一级为候选支撑面；多个踏面同时接近时 `surface_ambiguous=1`，不能确定实际承力面。
- `phase/curb_id` 是脚掌中心所在区域，`surface_phase/surface_curb_id` 是候选承力踏面。路线统计使用后者，避免把“脚跟还在 curb 上、中心已越边”误计成平地触地；承力面不明确的过渡记为 `ungraded`。
- 合并共面的地形接缝，避免把最后一级平地和后续平地之间的网格接缝误当成台阶边缘。

足底采用当前圆柱碰撞鞋的**保守矩形包络**（长 0.186 m、宽 0.072 m），不是实际接触多边形／压力中心。越边是几何证据，不能单凭此认证脚尖／脚跟确实承力、打滑或失稳。换鞋和 `sole_offset` 后需要同步校核包络。

## 目标与误差的有效性

参考在释放支撑或空中阶段、另一脚确认支撑时生成，之后不根据实际落脚反向移动目标。没有可靠参考时不临时在触地点补造目标。

- `world_error_valid=1`：有提前生成的固定世界参考，允许用仿真特权定位计算偏差。
- `support_error_valid=1`：另要求从生成到候选触地持续有效支撑，且原支撑脚位移 ≤0.015 m、转角 ≤0.20 rad。目标和实际位置都投到**同一个冻结支撑坐标系**。
- 支撑失效／移动后，冻结支撑坐标系的误差留空并给出原因，世界真值偏差可保留。该检查依赖仿真位姿，不代表已有无滑移实机观测器。
- `target_surface_matched` 反映实际候选承力面与建议踏面是否一致。跨到别的踏面仍保留 XYZ 偏差，不能把该值单独解读为控制器跟踪精度。

新增参考选择和 CSV／面板耗时不计入原 `observer_cpu_ms`，原计时范围仍为 FK＋接触观测器。终止／重置前状态会保留在 CSV；该步 RGB 可能已显示自动重置后的机器人，须以 CSV 的终止帧为准。

## 本地离线检查

```bash
python scripts/foot_state_demo/offline_demo.py \
  --steps 300 --num_envs 2 --output_dir outputs/foot_state_demo/offline

python -m unittest discover -s scripts/foot_state_demo -p 'test_*.py' -v

python scripts/foot_state_demo/run_demo.py --dry_run \
  --load_run example --checkpoint model_24000.pt \
  --output_dir outputs/foot_state_demo/config_preview
```

离线入口使用当前真实 URDF，但关节运动和接触力是合成输入；输出 CSV、JSON 和 `offline_preview.png`，不模拟机器人、不生成机器人 MP4。没有独立仿真真值时，FK 误差标为 `null`，不制造零误差结果。

首次仿真重点查看：FK 对照误差、触地事件位置是否在候选时锁存、终止／重置是否串帧，以及 MP4 面板是否正常。CPU 耗时统计不含 GPU→CPU 拷贝、CSV 和渲染；不能据此宣称整个系统开销或优于 InEKF。
