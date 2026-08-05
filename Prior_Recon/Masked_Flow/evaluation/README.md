# Masked-Flow motion generation 评测

这套评测把 ARDY/Kimodo 一类 motion generation 表格中的指标映射到
Prior-Recon 的 G1、双手末端轨迹条件场景。Joint、keyframe、trajectory 和 waypoint
指标衡量的是约束遵循度；只有在生成时实际施加了与 reference 相同的对应约束和采样
时刻，这些值才可解释为控制误差。所有逐轨迹误差先在一条轨迹内取均值，再以轨迹为
统计单位做数据集平均，避免长轨迹获得更高权重。
没有任何接地帧的轨迹无法定义 Skate，会以 `null` 保留在 per-clip 结果中；聚合报告同时
给出 `skate_valid_clips` 和 `skate_omitted_clips`，不会静默隐藏覆盖率。

## 指标定义

| 输出字段 | 单位/方向 | 本项目中的定义 |
|---|---:|---|
| `skate_m_s` | m/s ↓ | G1 左右脚 sole marker 在连续接地帧上的平均水平速度；接地由两帧 sole 高度均不超过 `floor + 0.05 m` 判定。 |
| `r_precision_top3_percent` | % ↑ | 仅在提供外部冻结 evaluator embedding 时输出的 Top-3 条件到运动检索准确率。 |
| `fid` | ↓ | 仅在提供外部冻结 evaluator embedding 时输出的生成/reference Fréchet distance。 |
| `joint_rot_deg` | deg ↓ | 受约束末端关节（本项目为左右手）与 reference 目标的平均四元数 SO(3) 测地角误差。 |
| `joint_pos_m` | m ↓ | 受约束末端关节（本项目为左右手）与 reference 目标的平均欧氏位置误差。 |
| `keyframe_body_m` | m ↓ | 按 `keyframe_stride` 采样的 full-body 约束帧上，29 个驱动 joint body 的平均 FK 位置误差。 |
| `traj_m` | m ↓ | warmup 后所有帧的根节点 XY 平面轨迹误差。 |
| `waypoint_m` | m ↓ | 按 `waypoint_stride` 采样的根节点 XY waypoint 误差。 |

`joint_rot_deg` 和 `joint_pos_m` 优先使用 reference `.npz` 中的
`keypoints: (T, 2, 7)` 作为目标；没有该字段时使用 reference qpos 的 FK 双手姿态。
raw `feat` 不保存初始 root XY，而配套 keypoints 是 world-frame；评测器会仅对这种输入
恢复两者之间丢失的共享 XY 平移。显式 qpos 输入不做初始位姿对齐，初始化误差仍会计入。

UMI EE-only bundle 的 `feat` 是全零占位符，不含身体或根节点真值。评测器会自动识别
这种输入：仍计算 Skate、手部 Joint rot./pos. 和 EE 检索 proxy，但将
`keyframe_body_m`、`traj_m`、`waypoint_m` 与内部身体 Fréchet 记为 `null`，并通过
`body_reference_valid_clips` / `body_reference_omitted_clips` 报告覆盖率。全零 `feat`
若不同时提供 `keypoints` 会被拒绝，避免把占位符当作有效 reference。

`warmup_frames` 被视为 history/conditioning 帧，会从 Skate、内部 proxy 特征和所有逐对
误差中一致排除。`keyframe_stride`、`waypoint_stride` 必须与生成约束的真实采样协议一致；
默认值只是可运行基线，不代表论文协议或所有 Masked-Flow 配置。

## 输入约定

`--generated` 和 `--reference` 都可以是单文件或目录。目录会递归扫描并按相对路径
去掉扩展名后配对，例如：

```text
generated/task_a/clip_001.npy
reference/task_a/clip_001.npz
```

支持的 motion artifact：

- `.npy`：`(T, 36)` G1 MuJoCo qpos；
- `.npz`：包含 `qpos`、`planned_qpos` 或 `prior_qpos` 中任一 `(T, 36)` 数组；
- reference `.npz` 也可只含预处理落盘的 raw `feat: (T, 70)`，评测器会按
  delta69/70 约定重建 qpos；
- reference `.npz` 可选 `keypoints: (T, 2, 7)` 和标量 `source_fps`。

任一 artifact 的 `source_fps` 存在时会作为该 clip 的有效帧率；两侧都带 stamp 时必须
一致，否则评测直接失败。两侧都没有时才使用命令行 FPS。每条结果会记录实际 FPS，
正式对比应保证整个 split 使用一致帧率。

Dataset 在窗口化阶段动态追加的 73D `abs_root_channels` 特征不是 raw artifact，不能从
上述接口加载；评测器会明确拒绝它，避免把第 69 维相对 XY 误解成绝对 yaw。

生成和 reference 必须具有相同 clip ID 和帧数。至少需要两条轨迹才能计算 FID；
R@3 在候选数不超过 3 时必然为 100%，正式实验建议使用完整 held-out split，至少
数百条轨迹。

## 运行

在 `UMI-Motion-Prior` 根目录执行：

```bash
python -m Prior_Recon.Masked_Flow.scripts.evaluate_motion_generation \
  --generated outputs/qpos \
  --reference data/delta_feat/val \
  --output outputs/motion_metrics.json \
  --fps 30 \
  --warmup-frames 2 \
  --keyframe-stride 16 \
  --waypoint-stride 16
```

命令行会打印聚合结果，并写入包含 `protocol`、`aggregate` 和 `per_clip` 的 JSON。
`embedding_backend` 会记录 R-precision/FID 使用的 evaluator，保证不同后端的结果
不会被误混。

## 接入统一 evaluator embedding

未提供外部 embedding 时，评测器仍会计算针对 UMI 双手条件与 G1 运动学的内部诊断，
但 JSON 字段明确命名为 `proxy_r_precision_top3_percent` 和
`proxy_frechet_distance`，且 `metric_scope=internal_proxy_not_paper_comparable`。
它们不能和使用 TMR/HumanML3D/Bones Rigplay evaluator 的论文数值直接横向比较。若已有
在同一 G1 数据集上独立训练并冻结的 text/condition-motion evaluator，可为每个 clip
导出一维 `.npy` embedding，并同时传入三个目录：

```bash
python -m Prior_Recon.Masked_Flow.scripts.evaluate_motion_generation \
  --generated outputs/qpos \
  --reference data/delta_feat/val \
  --condition-embeddings evaluator/condition \
  --generated-embeddings evaluator/generated \
  --reference-embeddings evaluator/reference \
  --output outputs/motion_metrics_external.json
```

embedding 必须是一维、有限、非零且三侧维度一致。三个目录必须复用 motion artifact
的相对 clip ID。此时 R@3 使用
condition/generated embedding，FID 使用 generated/reference embedding，报告中的
`embedding_backend` 为 `external_evaluator_embeddings`，标准字段名才会出现。
`metric_scope=external_evaluator_protocol_user_verified` 表示 checkpoint 与预处理一致性
由调用方保证，而不是评测器自动证明。跨模型对比时必须冻结同一 reference split、同一
evaluator checkpoint、同一采样数、相同 warmup 裁剪和本文件中的 protocol。
