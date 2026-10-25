# Go2 HIMLoco 独立环境

## 1. 环境边界

这套环境用于运行 `IsaacZH/himloco_lab` 的 Isaac Lab 端到端移植，不替换原来的
`/home/rum/IsaacLab`。两个 Python 环境和日志目录完全分开。

| 项目 | 固定位置或版本 |
|---|---|
| 官方 Isaac Lab 源码 | `/home/rum/IsaacLab_HIMLoco`，tag `v2.3.2`，commit `37ddf626871758333d6ed89cf64ad702aef127d0` |
| Python 环境 | `/home/rum/env_isaaclab_himloco`，Python 3.11 |
| Isaac Sim | `5.1.0.0` |
| Isaac Lab Python 包 | `isaaclab 0.54.2`、`isaaclab_rl 0.4.7` |
| 官方 RSL-RL 依赖 | `rsl-rl-lib 3.1.2` |
| 社区 HIMLoco 移植 | `/home/rum/himloco_lab`，commit `c3fbd0173f2a53a86e472c099f8e2e9744cf7639` |
| HIMLoco 官方算法对照 | `/home/rum/HIMLoco_official`，commit `ef289acaa62795009363b7b819c9186690630441` |
| Go2 本地资产 | `/home/rum/himloco_assets/unitree_ros`，commit `daadf41ee9afce8f90fdc09a98506012691fa122` |

重要边界：社区移植内置自己的旧式 `HIMOnPolicyRunner/HIMPPO`，并不是
`rsl-rl-lib 3.1.2` 或 5.0.1 的原生 HIM 实现。这里降低版本是为了减少 Isaac Lab
环境 API 差异，并保留作者已经打通的训练路径，不代表它已变成 HIMLoco 官方复刻。

## 2. 当前网络接口

- 单帧 Actor 观测：45 维（命令、IMU、12 关节角、12 关节速度、上一动作）。
- 历史：当前帧加5帧过去状态，共6帧，`45 × 6 = 270`，frame-major、最新帧在前。
- HIM estimator：`270 -> 128 -> 64 -> 19`，输出估计速度3维和latent 16维。
- Actor：`45 + 3 + 16 = 64 -> 512 -> 256 -> 128 -> 12`。
- Critic：238维，包含训练期特权线速度、外力和高度扫描；这些不进入部署端 Actor。

本地修复了社区移植的 episode 历史串线：初始化和每次环境 reset 后，会把新 episode
首帧复制到全部6个历史槽，不再把上一 episode 的末帧作为 successor 正样本。

## 3. 已完成验证

- `source/himloco_lab/test/test_history_buffer.py`：2项纯 Torch 测试通过。
- 2环境×2轮训练 smoke 通过，PPO、速度估计损失和prototype swap loss均完成更新。
- 修复历史 reset 后，再次完成2环境×1轮训练 smoke。
- smoke checkpoint 仅用于基础设施验证，按本协议视为 `eligible=false`，不得用于能力比较或正式续训。

## 4. 使用方法

查看任务：

```bash
cd /home/rum/himloco_lab
./run_himloco.sh scripts/list_envs.py --headless
```

再次运行最小训练 smoke：

```bash
cd /home/rum/himloco_lab
./run_himloco.sh scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go2-Velocity \
  --num_envs 2 \
  --max_iterations 2 \
  --device cuda:0 \
  --headless \
  --seed 42 \
  --run_name manual_smoke_only
```

未来正式训练不能直接复用上述参数。必须先冻结环境/reward/domain-randomization配置、
补齐多seed基线和独立评估门，再决定环境数、训练轮数和晋级规则。

## 5. 已知限制

- `model_11900.pt` 是原225/300 MLP网络，与新HIM的270/238/64拓扑不兼容，不能resume。
- 当前社区配置与原项目的地形、reward、命令分布和Critic都不同；从零训练所得结果不能
  直接归因于“加入HIM编码器”，必须建立相同环境下的无HIM对照组。
- Go2 URDF导入有头部visual引用警告，但物理环境、12关节和训练均成功创建；正式可视化
  前需要单独检查模型外观，不能把visual警告等同于物理碰撞失败。
- `pip check`会报告Isaac Sim 5.1的FastAPI与Isaac Lab的Starlette pin冲突。这一冲突在原
  环境也存在，真实headless启动与训练已通过；不要为消除提示擅自升级/降级Starlette。
- 社区仓库根许可证信息与部分文件头不完全一致。当前只做本机研究；公开发布或商用前需
  单独核对代码来源和许可证。
