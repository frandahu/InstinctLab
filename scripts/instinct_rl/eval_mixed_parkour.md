# G1 复杂地形验证与 MP4

入口：`scripts/instinct_rl/eval_mixed_parkour.py`，复用 `eval_stairs.py` 的冻结 actor、保存配置恢复、诊断与 MP4 编码。不修改训练配置，不恢复优化器或 AMP 判别器，不需要动作数据集。使用当前项目的 Isaac Sim 5.1 工作流。

## 路线

默认 `--terrain_mode mixed`：每条路线依次通过 **上楼 → 顶部平台 → 下楼 → 上坡 → 平台 → 下坡 → 路沿平台 → 地面终点**。每段楼梯默认 4 级，沿用已有稀疏异常台阶采样；阶高 0.08–0.20 m、踏面深度 0.25–0.40 m。坡角 5–20 度、每段坡水平长度 2 m，路沿高 0.08–0.20 m；各顶部平台默认长 1.2 m。

| `--terrain_mode` | 验证内容 |
|---|---|
| `mixed` | 每条通道都是完整楼梯、斜坡、路沿连续路线 |
| `slope` | 上坡、平台、下坡 |
| `curb` | 单个路沿平台，上沿后再下沿 |
| `stairs` | 原有上楼、平台、下楼协议 |
| `flat` | 原有平地对照 |
| `suite` | 按通道依次分配 flat、stairs、slope、curb、mixed，至少 5 个环境 |

`suite` 提供独立分地形结果，`mixed` 检查连续切换地形。相同 seed 和参数生成相同几何，增加环境数不改变已有通道的几何。测试几何不作为 actor 观测；坡角等范围可能与训练重叠，不能据此声称分布外泛化。

## 服务器命令

在容器内仓库根目录、训练所用 Python 环境执行。`--device` 是容器内的**逻辑** GPU 编号，先选择有足够空闲显存的设备；不要根据 Run 名称推断物理 GPU 映射。

```bash
cd /workspace/instinctlab
RUN=/workspace/instinctlab/logs/instinct_rl/g1_parkour_mixed/20261008_063454_author_walk_air075_resume_gpu1_device-cuda:0_from20261008_033821
EVAL_DEVICE=cuda:0  # 按当前空闲的逻辑 GPU 修改

# 最小启动/录像检查：仅约 4 秒仿真，不能据此判断路线成功率。
python scripts/instinct_rl/eval_mixed_parkour.py \
  --load_run "$RUN" --checkpoint model_22000.pt \
  --device "$EVAL_DEVICE" --headless \
  --num_envs 1 --episodes_per_env 1 --max_steps 200 \
  --output_dir outputs/terrain_eval/model22000_smoke

# 完整连续路线：4 种几何，每种重复 3 次，单回合最多 60 秒。
python scripts/instinct_rl/eval_mixed_parkour.py \
  --load_run "$RUN" --checkpoint model_22000.pt \
  --device "$EVAL_DEVICE" --headless \
  --num_envs 4 --episodes_per_env 3 --seed 42 \
  --output_dir outputs/terrain_eval/model22000_mixed_seed42

# 分地形对照：每种地形一个通道，各重复 3 次。
python scripts/instinct_rl/eval_mixed_parkour.py \
  --load_run "$RUN" --checkpoint model_22000.pt \
  --device "$EVAL_DEVICE" --headless \
  --terrain_mode suite --num_envs 5 --episodes_per_env 3 --seed 42 \
  --output_dir outputs/terrain_eval/model22000_suite_seed42
```

比较 `model_8000.pt` 时，只替换 checkpoint 和输出目录；保持 seed、地形参数、速度、环境数、回合数、噪声及动作模式一致。默认确定性均值动作，`--sample` 仅用于单独检查探索噪声。多个 seed 才能增加几何实例；同一路线重复 3 次不代表 3 种独立几何。

## 输出和判定

- `terrain_walk.mp4`：mixed/slope/curb/suite 默认输出，H.264、1280×720；在 0.02 s 控制周期下默认 25 FPS。stairs/flat 沿用 `staircase.mp4`。
- `summary.json`：总体和 `by_terrain` 分组成功率、失败原因、完成/计划试验数及视频编码状态。
- `episodes.csv`：每回合结果、速度误差、侧向偏离、通过/要求的检查点数量。
- `trace.csv`：物理步后且自动重置前的状态、局部地面高度、姿态、脚接触、指令、动作变化、关节速度、力矩及深度输入统计。
- `cases.json`：路线轮廓、碰撞网格顶点/面、检查点、seed、参数、checkpoint 和保存配置的 SHA256。
- `eval_env.yaml`、`eval_agent.yaml`、`startup_status.json`、`startup.log`，异常时另有 `startup_error.txt`。

连续路线必须在**每个**障碍顶部平台完成有支撑的检查点：双脚踝位于平台范围、距平台高度小于 0.20 m，至少一只脚接触力超过 5 N、机身倾角小于 0.5 rad、根节点距当地地面高于 0.5 m。然后到达最终目标，保持原评估器的稳定终点条件 0.5 秒才算成功；仅飞过检查点不计通过。失败优先于同一步成功。

高度终止按解析的当地支撑面计算，斜坡使用与碰撞网格一致的线性插值。躯干接触、姿态失败、走出通道、非有限状态和超时分别报告。

只在 `status=complete` 且 `uncompleted_episodes=0` 时解读完整成功率。短运行的未完成试验不会被算作成功或物理摔倒。先比较完成率和终止原因，再按 env_id/episode_index 的共同时间区间比较运动误差，避免长短回合均值偏差。这里不报告逐帧参考动作模仿精度。

视频默认固定侧视全景，观察 `--video_env_id 0` 通道；suite 的 0–4 通道依次是平地、楼梯、斜坡、路沿、混合路线。可用 `--video_env_id 4` 观察 suite 的连续路线。MP4 观察者相机不改变策略的深度输入。`--video_length 1500` 限制录像步数而继续统计，`--no_video` 仅输出指标，输出目录已存在时自动创建编号目录。

若提示缺少 FFmpeg，在原仿真 Python 环境安装 `imageio[ffmpeg]` 后重试；脚本不会自动安装或升级依赖。渲染、checkpoint 或配置不兼容时查看启动日志；不会静默改用当前训练配置。

## 本地 CPU 检查

```bash
python scripts/instinct_rl/eval_mixed_parkour.py \
  --dry_run --load_run example --checkpoint model_22000.pt \
  --terrain_mode suite --num_envs 5 --output_dir outputs/terrain_eval/preview
python -m unittest discover -s scripts/instinct_rl -p test_terrain_eval.py -v
python -m unittest discover -s scripts/instinct_rl -p test_stair_eval.py -v
```

dry run 不启动仿真、不加载 checkpoint，也不生成 MP4。CPU 测试不等于 Isaac Sim、RTX 渲染或真实策略验证；先运行上面的最小服务器录像检查。
