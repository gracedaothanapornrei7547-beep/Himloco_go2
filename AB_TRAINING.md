# Go2 普通 history-MLP 与 HIM 公平 A/B

## 锁定变量

- 同一个 Gym task：`Unitree-Go2-Velocity`
- 同一套地形、奖励、事件随机化、速度课程、动作和 PD
- Actor 部署信号均为 45 维单帧本体观测的 6 帧历史，`270` 维
- Critic 均为同一个单帧 `238` 维特权观测
- 同一个 runner、rollout storage、PPO、seed、环境数和 transitions 预算
- 相同的 Critic、动作标准差及 Actor 共有后三层初始化

唯一算法变量：

- `history_mlp_270`：完整 270 维历史直接进入 `[512, 256, 128]` MLP
- `him_release_encoder_270_to_19`：HIM source encoder 将 270 维历史编码为估计速度 3 维和 latent 16 维，再与当前 45 维拼成 64 维进入相同尺寸 Actor；训练时增加官方 release 风格 velocity MSE 与 prototype swap loss

这不是原工程 `model_11900.pt` 的继续训练。两个分支都在隔离的 IsaacLab 2.3.2 环境从零训练，原 225 维模型保持不可变。

## 正式协议

- seeds：`42 123 2026`
- 每组：`512 env × 100 step × 15000 iteration = 768,000,000 transitions`
- 六个训练作业串行，顺序交错，避免同一组永远受先跑/后跑的机器热状态影响
- headless、GPU 0；控制器不自动换奖励、学习率或环境参数
- 每个 run 的 `params/ab_manifest.json` 记录共同身份和初始化哈希
- 每个 checkpoint 内嵌 `ab_metadata`
- 配对验证工具会拒绝环境身份、shape、seed、预算或初始化不一致的结果

## 状态与日志

正式控制器：

```bash
/home/rum/env_isaaclab_himloco/bin/python \
  /home/rum/himloco_lab/scripts/himloco_rsl_rl/run_ab_training.py \
  --seeds 42 123 2026 \
  --num-envs 512 \
  --iterations 15000 \
  --output-dir /home/rum/himloco_lab/logs/himloco_rsl_rl/ab_formal_v1
```

查看当前状态：

```bash
systemctl --user status go2-him-ab-v1.service --no-pager -l
journalctl --user -u go2-him-ab-v1.service -f -o cat
watch -n 2 nvidia-smi
```

控制器状态：

```bash
python -m json.tool \
  /home/rum/himloco_lab/logs/himloco_rsl_rl/ab_formal_v1/state.json
```

当前子任务的完整 Isaac 日志在同目录的 `seed<seed>_<variant>.log`。

## 结论边界

训练 reward 和 HIM loss 只能说明优化过程，不能直接说明 HIM 更会盲走。六组训练完成后，必须用同一组 held-out seeds、固定命令、平地/坡地/方块/上下楼、外力推扰进行独立评估，比较速度误差、存活率、姿态、机身接触、关节速度、冲击、均值与 seed 方差。独立评估完成前不得宣布优胜模型。
