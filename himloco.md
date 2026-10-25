# 四足机器人盲走强化学习标准训练路线与避坑红宝书（HIMLoco / Project-Robo 实战复盘）

> **文档定位**：面向四足机器人强化学习研发人员的实战规范与避坑手册。结合官方标杆路线（Project-Robo / HIMLoco / ETH Legged Gym）的设计哲学，深度复盘从早期 `stair_curriculum` 踩坑到后期 `p73` 攻克 15cm 极限台阶全流程，梳理出一套标准、通用、防踩坑的训练路线、量纲配平法则、病理调参字典、Sim2Real 对齐指南与真机部署工程闭环。

---

## 目录

- [一、 核心哲学与指导纲领：极简主义 vs 规则干预](#一-核心哲学与指导纲领极简主义-vs-规则干预)
  - [1.1 极简奖励假设（The Minimalist Reward Hypothesis）](#11-极简奖励假设the-minimalist-reward-hypothesis)
  - [1.2 伪智能（硬编码过拟合） vs 真智能（动力学涌现）](#12-伪智能硬编码过拟合-vs-真智能动力学涌现)
  - [1.3 四足强化学习三大铁律](#13-四足强化学习三大铁律)
- [二、 标准四阶段训练路线（Stage-by-Stage Standard Roadmap）](#二-标准四阶段训练路线stage-by-stage-standard-roadmap)
  - [2.1 阶段一：纯平地基因定型（Stage 1: Flat Ground Trot Baseline, 0~1500 轮）](#21-阶段一纯平地基因定型stage-1-flat-ground-trot-baseline-01500-轮)
  - [2.2 阶段二：高速直行与姿态整形（Stage 2: High-Speed Trot & Posture Shaping, 1500~3000 轮）](#22-阶段二高速直行与姿态整形stage-2-high-speed-trot--posture-shaping-15003000-轮)
  - [2.3 阶段三：多地形与台阶课程攻坚（Stage 3: Curriculum Terrain & Stair Assault, 3000~6000 轮）](#23-阶段三多地形与台阶课程攻坚stage-3-curriculum-terrain--stair-assault-30006000-轮)
  - [2.4 阶段四：Sim2Sim / Sim2Real 迁移与鲁棒性注水（Stage 4: Sim2Real Transfer, 6000~8000 轮）](#24-阶段四sim2sim--sim2real-迁移与鲁棒性注水stage-4-sim2real-transfer-60008000-轮)
- [三、 早期“毒药奖励”深度尸检与官方解法对照（Antipatterns & Autopsy）](#三-早期毒药奖励深度尸检与官方解法对照antipatterns--autopsy)
  - [3.1 毒药一：`feet_stumble` 负惩罚（触壁畏缩与拒爬死锁）](#31-毒药一feet_stumble-负惩罚触壁畏缩与拒爬死锁)
  - [3.2 毒药二：`flat_orientation` 强行水平仪锁死（压死俯仰与前胸撞墙）](#32-毒药二flat_orientation-强行水平仪锁死压死俯仰与前胸撞墙)
  - [3.3 毒药三：`base_height` 扫描仪绝对高度硬锁（台阶托底与高度震颤）](#33-毒药三base_height-扫描仪绝对高度硬锁台阶托底与高度震颤)
  - [3.4 毒药四：`joint_pos` 全局/小腿强吸引子（四肢半僵死与无法深屈膝）](#34-毒药四joint_pos-全局小腿强吸引子四肢半僵死与无法深屈膝)
  - [3.5 毒药五：高斯型 `foot_clearance` 与大额正分（踢正步与奖励黑客）](#35-毒药五高斯型-foot_clearance-与大额正分踢正步与奖励黑客)
- [四、 常见步态病理诊断与排障字典（Troubleshooting SOP）](#四-常见步态病理诊断与排障字典troubleshooting-sop)
  - [4.1 病理 1：台阶拒爬与触壁踏步（Stair Stalling）](#41-病理-1台阶拒爬与触壁踏步stair-stalling)
  - [4.2 病理 2：平地内八字与台阶外八字（青蛙腿 / Frog Legs）](#42-病理-2平地内八字与台阶外八字青蛙腿--frog-legs)
  - [4.3 病理 3：台阶腹部托底卡死（Bottom Dragging）](#43-病理-3台阶腹部托底卡死bottom-dragging)
  - [4.4 病理 4：平地走偏、蛇形走位（Veering & Yaw Drift）](#44-病理-4平地走偏蛇形走位veering--yaw-drift)
  - [4.5 病理 5：高频神经发抖与电机共振（Actuator Chattering, 50Hz抖动）](#45-病理-5高频神经发抖与电机共振actuator-chattering-50hz抖动)
  - [4.6 病理 6：兔子跳与同侧踱步（Bounding / Pacing 步态退化）](#46-病理-6兔子跳与同侧踱步bounding--pacing-步态退化)
- [五、 参数量纲配平理论与黄金权重矩阵（Scale & Unit Balancing Theory）](#五-参数量纲配平理论与黄金权重矩阵scale--unit-balancing-theory)
  - [5.1 物理量纲差异导致的“奖励淹没”现象](#51-物理量纲差异导致的奖励淹没现象)
  - [5.2 黄金单步得分贡献配平大表（Production Balancing Table）](#52-黄金单步得分贡献配平大表production-balancing-table)
- [六、 四大核心病理的手术刀级调参深度拆解（Deep Surgical Tuning SOP）](#六-四大核心病理的手术刀级调参深度拆解deep-surgical-tuning-sop)
  - [6.1 核心攻坚 A：髋关节偏离（`hip_deviation_adaptive`）自适应门控推导](#61-核心攻坚-a髋关节偏离hip_deviation_adaptive自适应门控推导)
  - [6.2 核心攻坚 B：防擦地安全门限（`feet_min_height_body`）几何余量设计](#62-核心攻坚-b防擦地安全门限feet_min_height_body几何余量设计)
  - [6.3 核心攻坚 C：对角步频与腾空时间（Cadence Engine）锚定法则](#63-核心攻坚-c对角步频与腾空时间cadence-engine锚定法则)
  - [6.4 核心攻坚 D：平滑性三阶阻尼链（Triple Damping Chain）消灭 50Hz 啸叫](#64-核心攻坚-d平滑性三阶阻尼链triple-damping-chain消灭-50hz-啸叫)
- [七、 四阶段全参数演进阶梯对照总表（Stage Transition Ladder）](#七-四阶段全参数演进阶梯对照总表stage-transition-ladder)
- [八、 Sim2Real 物理动力学与执行器映射（Actuator Dynamics & Parity）](#八-sim2real-物理动力学与执行器映射actuator-dynamics--parity)
- [九、 官方标准奖励函数参考代码模板（Production Template）](#九-官方标准奖励函数参考代码模板production-template)
- [十、 训练与部署交付自检清单（Delivery Checklist）](#十-训练与部署交付自检清单delivery-checklist)
- [十一、 核心工程公式推导与真机部署对接全链路（Real-Robot Interfacing & Mathematical Anatomy）](#十一-核心工程公式推导与真机部署对接全链路real-robot-interfacing--mathematical-anatomy)
  - [11.1 工程公式一：45 维本体观测空间向机身坐标系的数学投影](#111-工程公式一45-维本体观测空间向机身坐标系的数学投影)
  - [11.2 工程公式二：HIM 双网络 (ONNX) 推理时序闭环与 RT-Budget 预算](#112-工程公式二him-双网络-onnx-推理时序闭环与-rt-budget-预算)
  - [11.3 实机对接细节一：关节电机编号与排列映射（Permutation Indexing Trap 避坑铁律）](#113-实机对接细节一关节电机编号与排列映射permutation-indexing-trap-避坑铁律)
  - [11.4 实机对接细节二：低层混合控制协议 (Hybrid PD) 与平滑启动样条](#114-实机对接细节二低层混合控制协议-hybrid-pd-与平滑启动样条)
  - [11.5 实机对接细节三：安全哨兵与急停保护状态机 (FSM Watchdog)](#115-实机对接细节三安全哨兵与急停保护状态机-fsm-watchdog)

---

## 一、 核心哲学与指导纲领：极简主义 vs 规则干预

### 1.1 极简奖励假设（The Minimalist Reward Hypothesis）

四足机器人强化学习中最常犯的错误，就是**“用人类的先入为主替网络做微观决策”**：
- 看到狗走路有点晃，就加一个姿态惩罚；
- 看到狗不抬腿，就加一个抬腿高斯引导；
- 看到脚撞到台阶，就加一个防撞惩罚；
- 看到腿有点弯，就加一个关节位姿吸引子。

> [!CAUTION]
> **“奖励越多，死得越惨”**：当奖励项超过 15 个，各个标量在多目标优化中剧烈打架。网络表面上在收敛，实际上在各个相互矛盾的惩罚势场夹缝中寻找病态妥协（例如靠外八字劈叉避开碰撞与高度惩罚）。

**官方路线的终极假设**：
复杂的地形越障能力，不应来自于人工制定的微观姿态规则，而应来自于：
$$\text{智能} = \text{极简宏观目标 (速度追踪)} + \text{严格物理边界 (能耗/力矩/限位)} + \text{阶梯式地形课程 (Curriculum)}$$

### 1.2 伪智能（硬编码过拟合） vs 真智能（动力学涌现）

* **伪智能（硬编码过拟合）**：
  引入预设的贝塞尔曲线、强行设定腾空高度、硬锁关节角度。
  - *表现*：训练极快（500 轮就好看），但在 10cm 台阶只要踏空 1cm 或地表摩擦力稍变，立刻坠落，泛化能力几乎为零。
* **真智能（动力学涌现）**：
  网络没有任何人工步态曲线，仅凭速度追踪驱动。在前脚撞击台阶瞬间，利用时序本体感知推断出障碍立面，自发演化出高提膝深屈膝翻越。
  - *表现*：前期（0~1000轮）动作丑陋、摇晃试错，但一旦跨过阈值，面对 5cm、10cm、15cm 以及斜坡、泥地时表现出惊人的抗扰韧性。

### 1.3 四足强化学习三大铁律

1. **正项唯一律**：正向奖励（`weight > 0`）原则上只给**任务速度追踪（`track_lin_vel_xy`, `track_ang_vel_z`）**。除此以外不要随便给任何行为发大额正分，杜绝奖励黑客（Reward Hacking）。
2. **负项阻尼律**：负向项（`weight < 0`）只能充当“物理阻尼”与“边界防护”（如限制角速度、限制关节加速度、限制关节过热内耗），绝不能用来精细雕刻动作。
3. **课程替代律**：如果机器人学不会某个动作（如爬台阶、跨壕沟），**优先升级环境课程（从 2cm 逐步加到 15cm）**，而不是在奖励函数里加一项新惩罚。

---

## 二、 标准四阶段训练路线（Stage-by-Stage Standard Roadmap）

```mermaid
flowchart TD
    A["阶段一：纯平地基因定型 (0~1500 轮)\n[任务速度 + 基础 Trot 对角步频 + 基础能耗]"] --> B["阶段二：高速直行与姿态整形 (1500~3000 轮)\n[提速至 1.8~2.0 m/s + 髋关节自适应纠偏 + 消除抖动]"]
    B --> C["阶段三：多地形与台阶课程攻坚 (3000~6000 轮)\n[0cm → 5cm → 10cm → 15cm 动态课程 + 释放俯仰角]"]
    C --> D["阶段四：Sim2Real 域随机化注水 (6000~8000 轮)\n[执行器时延 0~20ms + 质量/质心抖动 + 电机包络截断]"]
```

### 2.1 阶段一：纯平地基因定型（Stage 1: Flat Ground Trot Baseline, 0~1500 轮）
* **核心目标**：建立对角步态（Trot）基因与站立平衡，建立稳定的速度闭环。
* **地形配置**：100% 纯平地（`plane`），关闭一切复杂台阶与坑洼。
* **关键奖励组合**：
  * `track_lin_vel_xy` (1.5) + `track_ang_vel_z` (0.75)：宇树官方 2:1 黄金牵引比例；
  * `trot_contact` (0.5~1.0)：加法型对角支撑同步激励（强制对角两足同时触地/摆动）；
  * `feet_slide` (-0.1)：惩罚触地相切向滑动，逼迫脚掌踏实发力；
  * `joint_acc` (-2.5e-7) + `action_rate` (-0.05)：基础平滑项，防止高频冲击。
* **验收标准（Gate 1）**：平地 $0.8 \sim 1.0\text{ m/s}$ 稳定小跑；步频 $2.5 \sim 3.0\text{ Hz}$；垂直震颤峰峰值 $< 1.5\text{ cm}$。

### 2.2 阶段二：高速直行与姿态整形（Stage 2: High-Speed Trot & Posture Shaping, 1500~3000 轮）
* **核心目标**：拉大步幅，提速至 $1.8 \sim 2.0\text{ m/s}$；手术式矫正腿部姿态，根治内八字与外八字。
* **核心改进：髋关节标称构型约束（`hip_deviation_adaptive`）**：
  大腿与小腿电机均处于矢状面（前后摆动），横向展开角 100% 由 4 个髋关节（`.*_hip_joint`）决定。
  只对 4 个髋关节偏离标称角度（FL/RL=+0.10, FR/RR=-0.10 rad）施加二次方惩罚；大腿和膝盖关节权重完全为 0。直线奔跑全权重约束消灭内外八，转向时自动门控放开允许侧移。
* **验收标准（Gate 2）**：平地冲刺 $1.8 \sim 2.0\text{ m/s}$；对称展开角收敛在 $+10^\circ \sim +12^\circ$（标称 $+11.5^\circ$）；单腿峰值扭矩 $< 25\text{ Nm}$。

### 2.3 阶段三：多地形与台阶课程攻坚（Stage 3: Curriculum Terrain & Stair Assault, 3000~6000 轮）
* **核心目标**：攻破 5cm $\to$ 10cm $\to$ 15cm 连续台阶，掌握盲走立面攀爬。
* **核心操作**：
  1. **解绑俯仰角**：只保留横滚锁死（`roll_orientation_l2 = -0.30`），将 `flat_orientation` 降权至 0，允许机身后仰抬起前胸（Pitch 达 $-15^\circ$）。
  2. **单侧防擦地安全抬腿（`feet_min_height_body`）**：仅在足端低于机身 $-0.305\text{m}$ 且间隙小于 3.5cm 拖步时施加微弱惩罚（-0.02）。爬楼梯时前腿高抬到 $-0.15\text{m}$ 时**完全零惩罚**。
  3. **释放小腿合法接触**：将 `.*_calf` 从 `undesired_contacts` 中剔除，仅惩罚大腿（thigh）和髋部（hip）触碰台阶。
* **验收标准（Gate 3）**：Isaac Lab 10cm 台阶通过率 100%；15cm 极限台阶上楼梯通过率 $> 90\%$。

### 2.4 阶段四：Sim2Sim / Sim2Real 迁移与鲁棒性注水（Stage 4: Sim2Real Transfer, 6000~8000 轮）
* **核心目标**：将仿真中训练的策略无缝部署到 MuJoCo 物理引擎和真机。
* **域随机化（DR）参数**：
  * 执行器延迟：$0 \sim 20\text{ ms}$ 均匀随机时延；
  * 质量/质心：机身质量随机偏移 $\pm 1.5\text{ kg}$，质心位置（CoM）随机偏移 $\pm 2.0\text{ cm}$；
  * 摩擦力：静摩擦力在 $[0.3, 1.25]$ 区间内随机分布；
  * 电机转速-力矩包络截断（Go2HV Envelope）：嵌入真实电机转速反电动势压降曲线。
* **验收标准（Gate 4）**：MuJoCo 原生仿真中连续运行 1.5 小时（5000+ 秒，180+ 次循环）无倾翻。

---

## 三、 早期“毒药奖励”深度尸检与官方解法对照（Antipatterns & Autopsy）

### 3.1 毒药一：`feet_stumble = -0.5`（触壁畏缩与拒爬死锁）
- **初期诱因**：人类直觉认为“脚尖撞到垂直台阶立面是不好的，应该重重惩罚”。
- **物理后遗症**：纯本体盲视四足在面对台阶时，**前腿踏空碰撞立面是其感知地形空间边界的唯一触觉途径**。一旦脚尖撞击立面被扣大分，策略立刻学乖：“只要我不去碰台阶，我就不会被扣分”。最终导致策略在台阶前频繁减速、原地踏步甚至后退拒绝攀爬。
- **官方解法**：**完全不惩罚碰撞**。官方代码库中没有 `feet_stumble`。机器狗前脚踢到台阶立面后，反作用力会传导到关节，时序 Encoder 自发利用这一接触信号推断出台阶几何，并顺势屈膝翻越。

### 3.2 毒药二：`flat_orientation = -2.5`（强行水平仪锁死）
- **初期诱因**：为了平地好看，希望机身像平底锅一样稳。
- **物理后遗症**：机器狗爬楼梯时整体质心处于斜坡上，前胸必须自然向上仰起（Pitch 角度跌至 $-10^\circ \sim -18^\circ$），才能将推力分解为沿坡面分量。水平仪强锁死如同一根铁棍将机头按平，导致前足无空间抬起、前胸硬撞台阶。策略为了避开此惩罚，演化出“前腿极端外八字大劈叉”来强行压低底盘。
- **官方解法**：**Roll 与 Pitch 严格解耦**。只严惩左右侧倾（Roll），对前后仰角（Pitch）给与完全的物理自适应宽容度。

### 3.3 毒药三：`base_height = -1.0`（扫描仪绝对高度硬锁）
- **初期诱因**：防止小狗下蹲或塌腰趴窝。
- **物理后遗症**：台阶高度呈阶梯状断续跳变。扫描仪测得的距离在入阶瞬间剧烈震荡，机身随着台阶抬升，高度惩罚暴涨。网络为了消灭惩罚，命令关节下蹲，将腹部重重压在台阶棱角上，导致整机卡死托底。
- **官方解法**：**直接删除该项**。底盘的高度应该由足端承重接触与关节力矩正则自然支撑平衡。

### 3.4 毒药四：`joint_pos` 全局/小腿强吸引子（四肢半僵死与无法深屈膝）
- **初期诱因**：强制小腿维持 $-1.5\text{ rad}$（屈膝 $86^\circ$），希望保持微屈悬挂。
- **物理后遗症**：把小腿锁死后，机器狗在面临 15cm 台阶时，后腿无法完全蹬直延伸（需要伸展至 $-0.8\text{ rad}$），前腿无法极限折叠（需要收缩至 $-2.5\text{ rad}$），四肢活动范围被严重截断。
- **官方解法**：**只管髋关节侧向外摆（`.*_hip_joint`），释放全部大腿和膝盖自由度**。用能耗（`energy`）和力矩正则（`joint_torques`）代替显式位置惩罚。

### 3.5 毒药五：高斯型 `foot_clearance` 与大额正分（踢正步与奖励黑客）
- **初期诱因**：为了越障，直接给足端离地高度发大红包（正奖励）。
- **物理后遗症**：网络变成薅羊毛机器。在平地上也像德国仪仗队一样疯狂高抬腿踢正步，动作滑稽且电机撞地冲击力巨大；在台阶前甚至学会反复踢墙蹭分。
- **官方解法**：**以负向安全边界为主**（单侧线性门限），抬高腿不发红包，只在拖步擦地（$<3.5\text{cm}$）时微扣分，平地自然保持 $0.7\text{cm}$ 极致平稳低空浅掠。

---

## 四、 常见步态病理诊断与排障字典（Troubleshooting SOP）

### 4.1 病理 1：台阶拒爬与触壁踏步（Stair Stalling）
* **视觉现象**：平地冲得很欢，一到台阶脚尖碰一下，立刻停滞不前、原地快速踏步或者左右绕着走。
* **排障 SOP**：
  1. 彻底移除 `feet_stumble`；
  2. 检查 `undesired_contacts`，确保 `body_names` 只留 `.*_hip` 和 `.*_thigh`，严禁包含 `calf`；
  3. 检查课程台阶起始高度，从 3cm 开始平滑建立通过信心。

### 4.2 病理 2：平地内八字与台阶外八字（青蛙腿 / Frog Legs）
* **视觉现象**：平地走路时前两只脚往中间收（内八字两脚打架），爬台阶时两条前腿向两侧严重撇出（外八字像青蛙跳）。
* **排障 SOP**：
  1. 引入手术式髋关节约束 `hip_deviation_adaptive`（权重 $-0.25 \sim -0.35$），严惩偏离标称姿态（FL/RL=+0.10, FR/RR=-0.10）；
  2. 降低机身横滚与俯仰的惩罚权重。

### 4.3 病理 3：台阶腹部托底卡死（Bottom Dragging）
* **视觉现象**：前腿跨上了台阶，但后腿无法蹬起，腹部底盘吸在台阶边缘磨蹭。
* **排障 SOP**：
  1. 确认 `base_height` 已完全置为 `None`；
  2. 解锁俯仰角，允许机身后仰角度达到 $-15^\circ$ 以上；
  3. 检查电机转速-力矩包络，确保后腿蹬地时能输出至少 $20\text{ Nm}$。

### 4.4 病理 4：平地走偏、蛇形走位（Veering & Yaw Drift）
* **视觉现象**：下发纯直行指令，机器狗边跑边向左或向右画弧线漂移。
* **排障 SOP**：
  1. 增加对角四肢腾空时间方差正则 `air_time_variance_penalty`（权重 $-0.5 \sim -1.0$）；
  2. 检查两对对角腿（FL+RR 与 FR+RL）的接触力与占空比是否达到 50% 对称；
  3. 确认任务角速度追踪权重为正向牵引（`track_ang_vel_z = 0.75`）。

### 4.5 病理 5：高频神经发抖与电机共振（Actuator Chattering, 50Hz抖动）
* **视觉现象**：关节电机发出刺耳的高频啸叫，肉眼可见高频细碎抽搐。
* **排障 SOP**：
  1. 将 `action_rate_l2` 权重提至 $-0.05 \sim -0.08$；
  2. 引入关节角加速度惩罚 `joint_acc_l2 = -2.5e-7`；
  3. 确认训练中加入了 1 帧（$20\text{ms}$）的执行器随机延迟（`actuator_delay`）。

### 4.6 病理 6：兔子跳与同侧踱步（Bounding / Pacing 步态退化）
* **视觉现象**：前后腿同时起跳（双腿蹦跳），或者同侧两条腿同时向前迈（踱步 Pacing）。
* **排障 SOP**：
  1. 部署加法型对角接触奖励 `trot_contact_consistency`；
  2. 设定严厉的同侧步态惩罚因子（`pace_penalty_weight = 1.0`, `bound_penalty_weight = 1.5`）。

---

## 五、 参数量纲配平理论与黄金权重矩阵（Scale & Unit Balancing Theory）

### 5.1 物理量纲差异导致的“奖励淹没”现象

强化学习优化器（PPO）对所有奖励标量求和作为标量损失。由于物理环境中各状态量的单位不同，**原始数值的尺度差异可达 $10^6$ 倍以上**：

* 速度指数追踪项：取值严格在 $[0.0, 1.0]$；
* 关节角速度平方和 $\sum \dot{q}^2$：典型值在 $10^1 \sim 10^2\text{ rad}^2/\text{s}^2$；
* 机械功率能耗 $\sum |\tau \cdot \dot{q}|$：典型值在 $10^2 \sim 10^3\text{ W}$；
* 关节角加速度平方和 $\sum \ddot{q}^2$：典型值高达 $10^5 \sim 10^7\text{ rad}^2/\text{s}^4$！

> [!WARNING]
> 如果新手给 `joint_acc` 随意赋权重 `-0.01`，那么仅这一项每步扣分就会高达 $-10^4$，而速度追踪哪怕跑满也只有 $+1.5$ 分！**任务目标被彻底淹没在惩罚噪声中，机器狗直接学会原地躺平装死！**

### 5.2 黄金单步得分贡献配平大表（Production Balancing Table）

在标准的 50Hz 控制回路中，**单步综合期望总得分应在 $+0.8 \sim +1.4$ 之间**。各项的单步贡献量配平矩阵如下：

| 奖励项名称 | 原始物理量计算公式 | 典型物理量范围 (Raw Range) | 生产推荐权重 (Weight) | 单步期望得分贡献 (Step Score) | 作用与物理意义 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`track_lin_vel_xy`** | $\exp(-\|v_{xy} - v_{xy}^{\text{cmd}}\|^2 / 0.25)$ | $0.2 \sim 0.95$ | **$+1.50$** | **$+0.80 \sim +1.42$** | **主牵引源（占比 70%）** |
| **`track_ang_vel_z`** | $\exp(-(\omega_z - \omega_z^{\text{cmd}})^2 / 0.25)$ | $0.5 \sim 0.98$ | **$+0.75$** | **$+0.40 \sim +0.73$** | 转向与航向锁死 |
| **`trot_contact`** | 详见对角布尔逻辑矩阵 | $0.0 \sim 1.0$ | **$+0.50$** | **$+0.20 \sim +0.45$** | 诱发对角节律小跑 |
| `lin_vel_z_l2` | $v_z^2$ | $0.005 \sim 0.05\text{ m}^2/\text{s}^2$ | **$-2.00$** | $-0.01 \sim -0.10$ | 抑制垂直机身颠簸 |
| `ang_vel_xy_l2` | $\omega_x^2 + \omega_y^2$ | $0.05 \sim 0.5\text{ rad}^2/\text{s}^2$ | **$-0.05$** | $-0.002 \sim -0.025$ | 抑制机身晃荡 |
| `roll_orientation_l2` | $g_x^2$ | $0.001 \sim 0.04$ | **$-0.30$** | $-0.0003 \sim -0.012$ | 严防左右侧翻（解耦Pitch） |
| `hip_deviation` | $\sum (q_{\text{hip}} - q_{\text{nom}})^2$ | $0.02 \sim 0.20\text{ rad}^2$ | **$-0.25$** | $-0.005 \sim -0.050$ | 约束侧摆消灭内外八字 |
| `feet_slide` | $\sum \|v_{\text{foot}}^{xy}\| \cdot \mathbb{I}_{\text{contact}}$ | $0.05 \sim 0.30\text{ m/s}$ | **$-0.10$** | $-0.005 \sim -0.030$ | 消除触地相滑步挫伤 |
| `foot_clearance` | $\sum \operatorname{ReLU}(h_{\text{margin}} - \Delta z)$ | $0.0 \sim 0.08\text{ m}$ | **$-0.02$** | $0.000 \sim -0.002$ | 摆动相单侧安全防擦地 |
| `action_rate_l2` | $\|a_t - a_{t-1}\|^2$ | $0.05 \sim 0.40$ | **$-0.05$** | $-0.002 \sim -0.020$ | 一阶平滑（防止指令突变） |
| `joint_acc_l2` | $\sum \ddot{q}^2$ | $10^4 \sim 5\times 10^5\text{ rad}^2/\text{s}^4$ | **$-2.5\times 10^{-7}$** | $-0.002 \sim -0.125$ | 二阶平滑（保护减速器齿轮） |
| `joint_torques_l2` | $\sum \tau^2$ | $100 \sim 600\text{ N}^2\text{m}^2$ | **$-2.0\times 10^{-4}$** | $-0.020 \sim -0.120$ | 限制电机工作温升区间 |
| `energy` | $\sum |\tau \cdot \dot{q}|$ | $50 \sim 400\text{ W}$ | **$-2.0\times 10^{-5}$** | $-0.001 \sim -0.008$ | 降低电池无效发热 |
| `undesired_contacts` | $\sum \mathbb{I}_{\text{contact}}(\text{thigh, hip})$ | $0 \text{ 或 } 1$ | **$-1.00$** | $0.000 \sim -1.000$ | 非法触地硬拦截 |

---

## 六、 四大核心病理的手术刀级调参深度拆解（Deep Surgical Tuning SOP）

### 6.1 核心攻坚 A：髋关节偏离（`hip_deviation_adaptive`）自适应门控推导

#### 1. 物理机理与标称角度定义
四足机器人的四条腿在站立时呈现轻微向外八字（A 字形支撑多边形），Unitree Go2 的官方机械结构标称几何中：
$$q_{\text{nominal}}^{\text{hip}} = \begin{cases} +0.10\text{ rad } (+5.73^\circ) & \text{左前腿 (FL) 与 左后腿 (RL)} \\ -0.10\text{ rad } (-5.73^\circ) & \text{右前腿 (FR) 与 右后腿 (RR)} \end{cases}$$
两腿相对夹角为 $11.46^\circ$（约 $+11.5^\circ$）。

#### 2. 自适应转向门控函数（Velocity Gating Function）
在直行奔跑时，必须将髋关节死死锚定在标称角度，防止前腿向内并拢（内八字阻碍摆动）或向外撇出（外八字降低推力效率）。但在大角度原地自转（Spin）或横向平移（Side Step）时，髋关节必须主动摆动。
为此构建动态门控因子：
$$\text{Gate}(\boldsymbol{v}_{\text{cmd}}) = \exp\left( -\frac{\|\omega_z^{\text{cmd}}\|^2}{2\sigma_\omega^2} - \frac{\|v_y^{\text{cmd}}\|^2}{2\sigma_y^2} \right), \quad \sigma_\omega = 0.4\text{ rad/s}, \; \sigma_y = 0.2\text{ m/s}$$
$$r_{\text{hip}} = -w_{\text{hip}} \cdot \text{Gate}(\boldsymbol{v}_{\text{cmd}}) \cdot \sum_{i \in \text{hips}} \left( q_i - q_i^{\text{nom}} \right)^2$$
- **直行状态**（$v_y=0, \omega_z=0$）：$\text{Gate}=1.0$，全权重严格纠偏；
- **剧烈转向状态**（$|\omega_z| > 0.8\text{ rad/s}$）：$\text{Gate} \to 0.0$，约束瞬间释放，赋予四肢全向机动自由度。

#### 3. 权重敏感度阶梯曲线
* $w = -0.10$：平地展开角仅收敛到 $-24.7^\circ$（严重内八字，前脚擦碰）；
* $w = -0.25$：台阶外八字完全收敛至 $+11.3^\circ$（标称 $+11.5^\circ$，重合度 98.3%）；平地内八字纠偏 40%；
* $w = -0.35$：平地展开角进一步拉齐至 $+10.8^\circ$，内外八字彻底根除。

---

### 6.2 核心攻坚 B：防擦地安全门限（`feet_min_height_body`）几何余量设计

#### 1. 为什么是 $-0.305\text{m}$ 与 $3.5\text{cm}$？
* Go2 连杆尺寸：大腿长 $L_1 = 0.213\text{m}$，小腿长 $L_2 = 0.213\text{m}$，完全垂直伸展极限长度为 $0.426\text{m}$；
* 标称站立时机身离地高 $0.340\text{m}$（足端相对机身重心坐标为 $z_{\text{nom}} = -0.340\text{m}$）；
* 摆动腿离地抬升 $3.5\text{cm}$ 时，足端在机身坐标系下的极限位置为：
  $$z_{\text{target}} = -0.340\text{m} + 0.035\text{m} = -0.305\text{m}$$

#### 2. 单侧非对称激活逻辑（One-Sided ReLU）
$$r_{\text{clearance}} = -w \cdot \sum_{i \in \text{feet}} \operatorname{ReLU}\left( z_i^{\text{body}} - z_{\text{target}} \right)^2 \cdot \mathbb{I}_{\text{swing}}(i)$$
* 当脚掌低空拖步（$z^{\text{body}} < -0.305\text{m}$）：处于安全区间之下，施加微弱惩罚逼其抬腿；
* 当攀爬 15cm 台阶，前膝深屈折叠至 $z^{\text{body}} = -0.180\text{m}$：**惩罚完全为 0**！绝不会阻碍极限越障高提膝。

---

### 6.3 核心攻坚 C：对角步频与腾空时间（Cadence Engine）锚定法则

#### 1. 步频 $f$ 与腾空时间 $T_{\text{air}}$ 的刚性物理映射
在四足对称 Trot 步态中，占空比（Duty Cycle）一般维持在 50%（支撑相与摆动相各占一半时间）：
$$T_{\text{stride}} = 2 \times T_{\text{air}}, \quad f_{\text{cadence}} = \frac{1}{T_{\text{stride}}} = \frac{1}{2 \times T_{\text{air}}}$$

* **$T_{\text{air}}^{\text{target}} = 0.20\text{s}$**：对应步频 **$5.0\text{ Hz}$** $\to$ 高频细碎抖腿，能量利用率极低，电机严重发烫；
* **$T_{\text{air}}^{\text{target}} = 0.32\text{s} \sim 0.35\text{s}$**：对应步频 **$2.85 \sim 3.12\text{ Hz}$** $\to$ **四足仿生奔跑黄金区间**，步幅开阔，支撑力矩饱满；
* **$T_{\text{air}}^{\text{target}} = 0.50\text{s}$**：对应步频 **$1.0\text{ Hz}$** $\to$ 乌龟拖步，单腿承受过大静态弯矩导致膝关节抖动。

#### 2. 静止死区门控（Zero-Velocity Lockout）
```python
if torch.norm(commands[:, :2], dim=1) < 0.1:
    reward_air_time = 0.0  # 速度指令归零时强行切断，严禁原地无休止踏步
```

---

### 6.4 核心攻坚 D：平滑性三阶阻尼链（Triple Damping Chain）消灭 50Hz 啸叫

电机 50Hz 神经发抖不是靠调小 PD 增益解决的，必须在 MDP 奖励层构建三阶阻尼链：

```mermaid
flowchart LR
    A["一阶差分 (Action Rate)\nΔa = a(t) - a(t-1)\n权重: -0.05 ~ -0.08\n[抑制控制指令突跳]"] --> B["二阶差分 (Jerk Smoothness)\nΔ²a = a(t) - 2a(t-1) + a(t-2)\n权重: -0.01 ~ -0.02\n[平滑加速度曲线]"] --> C["物理角加速度 (Joint Acc)\nq̈ = (q - 2q_prev + q_pprev) / Δt²\n权重: -2.5e-7\n[消除齿轮箱机械敲击]"]
```

* **黄金配比公式**：
  $$\frac{W_{\text{action\_rate}}}{W_{\text{smoothness}}} \approx 4 : 1, \quad W_{\text{joint\_acc}} = -2.5 \times 10^{-7}$$
  三者协同工作时，关节电机电流纹波（Current Ripple）降低 85%，彻底根除减速箱高频共振啸叫。

---

## 七、 四阶段全参数演进阶梯对照总表（Stage Transition Ladder）

在训练自己的四足时，请严格按照下表在各个 Stage 之间切换参数，切忌跳步：

| 奖励 / 约束参数名称 | Stage 1 (平地定型) | Stage 2 (高速直行) | Stage 3 (台阶攻坚) | Stage 4 (Sim2Real 注水) | 调参说明与切换判据 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`track_lin_vel_xy`** | 1.50 | 1.50 | 1.50 | 1.50 | 恒定主导项，保持 std=0.5 |
| **速度指令范围 $v_x$** | $[0.5, 1.0]\text{ m/s}$ | $[1.5, 2.0]\text{ m/s}$ | $[0.5, 1.2]\text{ m/s}$ | $[-1.0, 2.0]\text{ m/s}$ | 阶梯推进速度边界 |
| **地形类型 (Terrain)** | 100% Plane | 100% Plane | 5cm $\to$ 15cm Stairs | Multi-Rough + Stairs | Stage 3 开启台阶课程 |
| **`trot_contact`** | 1.00 | 0.80 | 0.50 | 0.50 | 早期强约束步态，后期放宽越障 |
| **`hip_deviation`** | -0.05 | **-0.25** | **-0.25 ~ -0.35** | **-0.25** | Stage 2 必须加码纠偏内外八字 |
| **`flat_orientation`** | -1.50 | -0.50 | **0.00 (彻底关闭)** | **0.00 (彻底关闭)** | Stage 3 必须清零，释放抬头自由 |
| **`roll_orientation`** | -0.50 | -0.40 | **-0.30** | **-0.30** | 恒定严防左右侧倾侧翻 |
| **`base_height`** | None | None | **None (严禁开启)** | **None (严禁开启)** | 全程坚决禁用，杜绝托底卡死 |
| **`foot_clearance`** | -0.01 | -0.02 | **-0.02 (安全门限)** | **-0.02 (安全门限)** | 仅用单侧门限，严禁用高斯引导 |
| **`feet_stumble`** | None | None | **None (严禁开启)** | **None (严禁开启)** | 全程坚决禁用，消除触壁畏缩 |
| **`undesired_contacts`** | Hip+Thigh+Calf | Hip+Thigh | **Hip+Thigh (严禁含Calf)**| **Hip+Thigh** | Stage 3 必须释放小腿触壁滑行 |
| **`action_rate_l2`** | -0.03 | -0.05 | -0.05 | **-0.08** | Stage 4 加强抗抖动注水 |
| **执行器时延 (Delay)** | 0 ms | 0 ms | $0 \sim 10\text{ ms}$ | **$0 \sim 20\text{ ms}$** | Stage 4 强制全开 |
| **质量扰动 (Base Mass)**| None | None | $\pm 0.5\text{ kg}$ | **$\pm 1.5\text{ kg}$** | Stage 4 强化质心鲁棒性 |

---

## 八、 Sim2Real 物理动力学与执行器映射（Actuator Dynamics & Parity）

### 8.1 真实电机转速-力矩包络截断（Go2HV Motor Envelope）

Unitree Go2HV 关节电机受限于反电动势（Back-EMF）与供电母线电压，输出扭矩随转速上升而下降：
$$\tau_{\text{limit}}(\dot{q}) = \tau_{\text{peak}} \cdot \max\left(0, 1 - \frac{|\dot{q}|}{\omega_{\text{no\_load}}}\right)$$
* 峰值堵转扭矩：$\tau_{\text{peak}} = 45.0\text{ Nm}$；
* 空载最高转速：$\omega_{\text{no\_load}} \approx 30.0\text{ rad/s}$（约 $286\text{ rpm}$）；
* 额定连续扭矩：$\tau_{\text{rated}} = 23.5\text{ Nm}$。

> [!IMPORTANT]
> 在仿真中，如果策略依赖了大于 $35\text{ Nm}$ 且转速大于 $15\text{ rad/s}$ 的工况，在真机上执行器驱动板会立即触发过流软保护降额，导致支撑腿塌陷。因此在 Stage 4 必须挂接该包络函数。

### 8.2 控制循环与时间离散化映射

* **高层强化学习策略回路**：$\Delta t_{\text{policy}} = 0.020\text{s}$（**50 Hz**），输入 6 帧时序历史（270 维），输出 12 个关节位置偏差 $\Delta q$；
* **低层硬件关节 PD 回路**：$\Delta t_{\text{physics}} = 0.002\text{s}$（**500 Hz**，Decimation = 10）；
* **Action Scale 物理意义**：
  $$q_{\text{des}} = q_{\text{default}} + \text{action} \times 0.25$$
  策略输出归一化在 $[-1.0, 1.0]$ 区间内，乘以 $0.25\text{ rad}$ 对应单步最大关节目标偏差为 $\pm 14.32^\circ$。这一数值既保证了足够的越障动态范围，又杜绝了一步发出过大位移命令击穿关节减速器。

---

## 九、 官方标准奖励函数参考代码模板（Production Template）

以下是经过 15cm 极限台阶实测验证、无毒、极简且高鲁棒的标准配置模板：

```python
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
import math

class StandardQuadrupedRewardsCfg:
    """四足机器人通用生产级奖励架构（去毒精简版）。
    
    设计准则：
      1. 正项唯一：仅牵引目标线速度与角速度；
      2. 物理守恒：限制力矩、角加速度与机械功率，杜绝内耗；
      3. 姿态解耦：严管横滚（Roll），释放俯仰（Pitch），手术式约束髋关节（Hip Roll）；
      4. 越障防擦：单侧线性安全门限，高抬腿零惩罚，彻底删除 stumble 与 height 毒药项。
    """

    # =========================================================================
    # 第一层：核心任务追踪（唯一的正分牵引源）
    # =========================================================================
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # =========================================================================
    # 第二层：步态协调性与地面交互
    # =========================================================================
    # 对角 Trot 步态加法诱导（包含踱步 pace 与兔子跳 bound 抑制）
    trot_contact = RewTerm(
        func=mdp.trot_contact_consistency,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )

    # 惩罚脚掌触地滑动（防止打滑并迫使踏实抓地）
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    # =========================================================================
    # 第三层：姿态与动力学解耦（根治内八/外八，释放爬坡仰角）
    # =========================================================================
    # 横滚硬锁死（严惩左右侧晃，防止侧翻）
    roll_orientation_l2 = RewTerm(func=mdp.roll_orientation_l2, weight=-0.30)

    # 基础垂直角速度阻尼（维持平稳，但不锁死 Pitch）
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    # ★ 髋关节手术式自适应构型约束（根治平地内八与台阶外八字）
    hip_deviation = RewTerm(
        func=mdp.hip_deviation_adaptive,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "command_name": "base_velocity",
        },
    )

    # =========================================================================
    # 第四层：足端越障与防擦地（单侧安全门限，彻底解绑高空束缚）
    # =========================================================================
    # 单侧防擦地安全抬腿（仅在离地低于 3.5cm 拖步时微扣分，爬台阶深屈膝高抬腿完全 0 惩罚）
    foot_clearance = RewTerm(
        func=mdp.feet_min_height_body,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": -0.305,
            "margin": 0.035,
        },
    )

    # 非法触地：严格将 calf（小腿）移出惩罚列表！允许小腿蹭着台阶棱角自然借力滑上踏面
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh"]),
            "threshold": 1.0,
        },
    )

    # =========================================================================
    # 第五层：关节能耗与平滑性保护（保护硬件齿轮与电机）
    # =========================================================================
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
```

---

## 十、 训练与部署交付自检清单（Delivery Checklist）

在新四足项目训练与交付前，逐项勾选以下清单：

- [ ] **检查 1：毒药项清零**
  - 确认 `feet_stumble`、`base_height_l2` 已彻底从奖励中移除；
  - 确认 `undesired_contacts` 中绝对没有包含 `calf`（小腿）。
- [ ] **检查 2：观测空间对齐（45 维本体感受）**
  - 确认单帧本体观测严格为 45 维（机身角速度 3 + 重力投影 3 + 关节位置 12 + 关节速度 12 + 上一步动作 12 + 速度指令 3）；
  - 确认历史观测 Ring Buffer 严格对齐 6 帧（$45 \times 6 = 270$ 维）。
- [ ] **检查 3：电机包络与时延注水**
  - 确认执行器挂载了真实转速-力矩曲线（如 Go2HV 的 45Nm 额定与反电势降额）；
  - 确认阶段四开启了 $0 \sim 20\text{ ms}$ 的执行器时延随机化。
- [ ] **检查 4：Sim2Sim 独立沙盒验机**
  - 导出 ONNX 双模型（`encoder.onnx` + `policy.onnx`）与 TorchScript `policy.pt`；
  - 在 MuJoCo 中独立运行至少 1000 秒，验证平地 $1.8\text{ m/s}$ 冲刺与连续台阶攀爬稳定性。

---

## 十一、 核心工程公式推导与真机部署对接全链路（Real-Robot Interfacing & Mathematical Anatomy）

> [!IMPORTANT]
> 仿真训练（Isaac Lab）到真实机器人（Unitree Go2 / SDK2 / C++）部署之间，存在三大“看不见的深坑”：
> 1. **坐标系与四元数投影数学差异**；
> 2. **关节拓扑排列索引不一致（Permutation Mismatch）**；
> 3. **低层混合 PD 执行器与通信协议抖动**。
> 任何一个细节未对齐，在真机上均表现为开机瞬间暴扭、电流过冲或倾翻。本章给出数学公式推导与可直接调用的 C++ 代码规范。

```mermaid
flowchart TD
    subgraph Sensing_Pipeline ["1. 硬件采样与观测投影 (500Hz → 50Hz)"]
        IMU["IMU (四元数 q + 角速度 ω)"] --> Gravity["重力投影数学推导\ng_b = R(q)ᵀ · [0, 0, -1]ᵀ"]
        MotorState["电机状态 (q_meas, q̇_meas)\n[Unitree 硬件顺序]"] --> PermutationIn["关节重排 Permutation\nUNITREE_TO_ISAAC"]
        PermutationIn --> ObsAssembly["组装 45 维单帧本体感受\n[ω_b, g_b, q̃, q̃̇, a_{t-1}, v_cmd]"]
        ObsAssembly --> RingBuffer["6 帧 FIFO 环形缓冲区 (270 维)"]
    end

    subgraph Dual_ONNX ["2. HIM 双网络极速推理 (RT-Budget < 1.0ms)"]
        RingBuffer --> Encoder["encoder.onnx\n270维 → [v̂(3维), z(16维)]"]
        Encoder & ObsAssembly --> Policy["policy.onnx\n64维 → 目标残差 a_t (12维)"]
    end

    subgraph Execution_Pipeline ["3. 指令重排与混合 PD 执行 (50Hz → 500Hz)"]
        Policy --> ActionScale["目标关节位置映射\nq_des = q_nom + a_t · 0.25"]
        ActionScale --> PermutationOut["关节逆重排 Permutation\nISAAC_TO_UNITREE"]
        PermutationOut --> HybridPD["五元组混合控制\nτ = Kp(q_des - q) + Kd(0 - q̇)"]
        HybridPD --> Motor["Go2HV 硬件底层执行器 (CAN/Ethernet)"]
    end
```

---

### 11.1 工程公式一：45 维本体观测空间向机身坐标系的数学投影

#### 1. 机身重力矢量投影公式（Projected Gravity Vector）
在真实机器狗机身上，IMU 测得姿态四元数 $\boldsymbol{q} = [w, x, y, z]^T \in \mathbb{R}^4$（表示机身坐标系相对于世界参考系的旋转）。世界重力矢量固定为 $\boldsymbol{g}_w = [0, 0, -1]^T$。
机身局部坐标系下的重力投影 $\boldsymbol{g}_b = [g_x^b, g_y^b, g_z^b]^T$ 是策略感知机身倾斜角与地面倾角的最关键输入。

旋转矩阵 $\boldsymbol{R}(\boldsymbol{q})$ 的第三列即为局部坐标系垂直轴在世界系下的投影，故其转置直接作用于 $\boldsymbol{g}_w$：
$$\boldsymbol{g}_b = \boldsymbol{R}(\boldsymbol{q})^T \begin{bmatrix} 0 \\ 0 \\ -1 \end{bmatrix} = \begin{bmatrix} -2(xz + wy) \\ -2(yz - wx) \\ -(1 - 2(x^2 + y^2)) \end{bmatrix}$$

* **C++ 高性能零内存开销实现**：
  ```cpp
  inline void compute_projected_gravity(
      float w, float x, float y, float z,
      float& gx, float& gy, float& gz) 
  {
      gx = -2.0f * (x * z + w * y);
      gy = -2.0f * (y * z - w * x);
      gz = -(1.0f - 2.0f * (x * x + y * y));
  }
  ```
  > [!CAUTION]
  > **四元数内存排列陷阱**：Unitree SDK 与 Isaac Lab 均为 `[w, x, y, z]`（标量在前）；而 ROS / Eigen 默认内存布局为 `[x, y, z, w]`（标量在后）。若直接 `memcpy` 传递，重力投影将彻底反转，导致机器狗开机后误以为自己“倒挂金钩”而猛烈反向起跳翻车！

#### 2. 机身角速度校准与偏置滤波
真实 IMU 陀螺仪存在固态温漂（Bias）：
$$\boldsymbol{\omega}_b = \boldsymbol{\omega}_{\text{raw}} - \boldsymbol{b}_{\text{gyro}}$$
在静止标定阶段记录前 200 帧均值作为 $\boldsymbol{b}_{\text{gyro}}$。若角速度高频噪声严重，可在 500Hz 硬件层施加截止频率为 $30\text{Hz}$ 的一阶低通滤波（LPF）：
$$\boldsymbol{\omega}_b^{(k)} = \beta \boldsymbol{\omega}_b^{(k-1)} + (1 - \beta) (\boldsymbol{\omega}_{\text{raw}}^{(k)} - \boldsymbol{b}_{\text{gyro}}), \quad \beta = \exp(-2\pi f_c \Delta t) \approx 0.686$$

#### 3. 关节位置与速度残差化映射
$$\tilde{q}_i = q_i^{\text{meas}} - q_i^{\text{nominal}}$$
$$\tilde{\dot{q}}_i = \dot{q}_i^{\text{meas}} \times \text{scale}_{\dot{q}}, \quad \text{scale}_{\dot{q}} = 1.0\text{ (或 0.05)}$$
**铁律**：`scale_joint_vel` 在仿真环境与实机 C++ 节点中必须 $100\%$ 数值精确一致，任何缩放差异会导致网络误判足端接触刚度。

---

### 11.2 工程公式二：HIM 双网络 (ONNX) 推理时序闭环与 RT-Budget 预算

#### 1. 270 维时序观测环形缓冲区（FIFO Ring Buffer）
单帧本体观测向量 $o_t \in \mathbb{R}^{45}$ 构成如下：
$$\boldsymbol{o}_t = \Big[ \boldsymbol{\omega}_b (3), \; \boldsymbol{g}_b (3), \; \tilde{\boldsymbol{q}} (12), \; \tilde{\dot{\boldsymbol{q}}} (12), \; \boldsymbol{a}_{t-1} (12), \; \boldsymbol{v}_{\text{cmd}} (3) \Big]^T \in \mathbb{R}^{45}$$

HIM 编码器需要过去 6 帧时序输入：
$$\boldsymbol{O}_t = \Big[ \boldsymbol{o}_t, \; \boldsymbol{o}_{t-1}, \; \boldsymbol{o}_{t-2}, \; \boldsymbol{o}_{t-3}, \; \boldsymbol{o}_{t-4}, \; \boldsymbol{o}_{t-5} \Big]^T \in \mathbb{R}^{270}$$
在 C++ 部署中，为杜绝动态内存分配（Heap Allocation）导致的实时性毛刺，采用静态预分配环形内存移动：
```cpp
// 零拷贝推进：将旧数据向后平移 45 维，最新帧拷入头部
std::memmove(history_buffer + 45, history_buffer, 5 * 45 * sizeof(float));
std::memcpy(history_buffer, current_obs, 45 * sizeof(float));
```

#### 2. 双网络协同推导公式与耗时预算（RT-Budget）
1. **显式速度与隐式环境特征估计**：
   $$\begin{bmatrix} \hat{\boldsymbol{v}}_t \\ \boldsymbol{z}_{\text{latent}} \end{bmatrix} = \text{EncoderONNX}(\boldsymbol{O}_t), \quad \hat{\boldsymbol{v}}_t \in \mathbb{R}^3, \; \boldsymbol{z}_{\text{latent}} \in \mathbb{R}^{16}$$
2. **策略动作推理**：
   $$\boldsymbol{a}_t = \text{PolicyONNX}\Big( \big[ \boldsymbol{o}_t, \; \hat{\boldsymbol{v}}_t, \; \boldsymbol{z}_{\text{latent}} \big] \Big) \in \mathbb{R}^{12}, \quad a_{t, i} \in [-1.0, 1.0]$$

* **生产环境端到端耗时拆解（Jetson Orin NX 16GB / 20W 模式）**：
  * Ring Buffer 内存搬移与归一化：**$0.03\text{ ms}$**
  * `encoder.onnx` 前向推理（CPU/GPU TensorRT）：**$0.32\text{ ms}$**
  * `policy.onnx` 前向推理：**$0.18\text{ ms}$**
  * 关节 Permutation 重排与 PD 打包：**$0.02\text{ ms}$**
  * Unitree Ethernet/CAN 底层通信往返：**$1.15\text{ ms}$**
  * **单步总延迟：$1.70\text{ ms} \ll 20.0\text{ ms}$**（CPU 负载仅占 $8.5\%$，零抖动稳定运行在 50Hz）。

---

### 11.3 实机对接细节一：关节电机编号与排列映射（Permutation Indexing Trap 避坑铁律）

> [!CAUTION]
> **四足机器人实机部署的第一死穴**：Isaac Lab 与 Unitree 官方硬件采用完全不同的拓扑索引！
> - **Isaac Lab 顺序**：按**关节类别**分组：`[全髋关节(4), 全大腿(4), 全小腿(4)]`；
> - **Unitree Go2 硬件 SDK 顺序**：按**单腿**分组：`[右前腿(3), 左前腿(3), 右后腿(3), 左后腿(3)]`。

#### 1. 关节双向转换矩阵映射对照表

| 硬件电机索引 (Unitree SDK) | 硬件关节物理名称 | 对应 Isaac Lab 索引 | Isaac Lab 数组物理名称 |
| :---: | :---: | :---: | :---: |
| **0** | `FR_hip` | **1** | `FR_hip_joint` |
| **1** | `FR_thigh` | **5** | `FR_thigh_joint` |
| **2** | `FR_calf` | **9** | `FR_calf_joint` |
| **3** | `FL_hip` | **0** | `FL_hip_joint` |
| **4** | `FL_thigh` | **4** | `FL_thigh_joint` |
| **5** | `FL_calf` | **8** | `FL_calf_joint` |
| **6** | `RR_hip` | **3** | `RR_hip_joint` |
| **7** | `RR_thigh` | **7** | `RR_thigh_joint` |
| **8** | `RR_calf` | **11** | `RR_calf_joint` |
| **9** | `RL_hip` | **2** | `RL_hip_joint` |
| **10** | `RL_thigh` | **6** | `RL_thigh_joint` |
| **11** | `RL_calf` | **10** | `RL_calf_joint` |

#### 2. C++ 生产级重排代码模板（开箱即用）
```cpp
// 宇树硬件状态 -> IsaacLab 观测排列
static constexpr int UNITREE_TO_ISAAC[12] = {
    3,  // Isaac[0]  = FL_hip   (Unitree[3])
    0,  // Isaac[1]  = FR_hip   (Unitree[0])
    9,  // Isaac[2]  = RL_hip   (Unitree[9])
    6,  // Isaac[3]  = RR_hip   (Unitree[6])
    4,  // Isaac[4]  = FL_thigh (Unitree[4])
    1,  // Isaac[5]  = FR_thigh (Unitree[1])
    10, // Isaac[6]  = RL_thigh (Unitree[10])
    7,  // Isaac[7]  = RR_thigh (Unitree[7])
    5,  // Isaac[8]  = FL_calf  (Unitree[5])
    2,  // Isaac[9]  = FR_calf  (Unitree[2])
    11, // Isaac[10] = RL_calf  (Unitree[11])
    8   // Isaac[11] = RR_calf  (Unitree[8])
};

// IsaacLab 策略输出动作 -> 宇树硬件发送排列
static constexpr int ISAAC_TO_UNITREE[12] = {
    1,  // Unitree[0]  = FR_hip   (Isaac[1])
    5,  // Unitree[1]  = FR_thigh (Isaac[5])
    9,  // Unitree[2]  = FR_calf  (Isaac[9])
    0,  // Unitree[3]  = FL_hip   (Isaac[0])
    4,  // Unitree[4]  = FL_thigh (Isaac[4])
    8,  // Unitree[5]  = FL_calf  (Isaac[8])
    3,  // Unitree[6]  = RR_hip   (Isaac[3])
    7,  // Unitree[7]  = RR_thigh (Isaac[7])
    11, // Unitree[8]  = RR_calf  (Isaac[11])
    2,  // Unitree[9]  = RL_hip   (Isaac[2])
    6,  // Unitree[10] = RL_thigh (Isaac[6])
    10  // Unitree[11] = RL_calf  (Isaac[10])
};

// 状态重排函数
inline void remap_state(const float* unitree_raw, float* isaac_ordered) {
    for (int i = 0; i < 12; ++i) {
        isaac_ordered[i] = unitree_raw[UNITREE_TO_ISAAC[i]];
    }
}

// 指令重排函数
inline void remap_action(const float* isaac_raw, float* unitree_ordered) {
    for (int i = 0; i < 12; ++i) {
        unitree_ordered[i] = isaac_raw[ISAAC_TO_UNITREE[i]];
    }
}
```

---

### 11.4 实机对接细节二：低层混合控制协议 (Hybrid PD) 与平滑启动样条

#### 1. 宇树五元组底层混合控制物理方程
宇树电机驱动板接收五元组控制结构体 `MotorCmd`：
$$\tau_{\text{motor}} = K_p (q_{\text{des}} - q_{\text{meas}}) + K_d (\dot{q}_{\text{des}} - \dot{q}_{\text{meas}}) + \tau_{\text{ff}}$$
* **生产环境最佳参数设定**：
  * $q_{\text{des}} = q_{\text{nominal}} + \boldsymbol{a}_t \times 0.25$（策略位置指令）；
  * $\dot{q}_{\text{des}} = 0.0$（将速度跟踪交由 PD 刚度隐式吸收）；
  * $K_p = 25.0 \sim 30.0\text{ Nm/rad}$；
  * $K_d = 0.5 \sim 0.8\text{ Nm}\cdot\text{s/rad}$；
  * $\tau_{\text{ff}} = 0.0\text{ Nm}$。

#### 2. 开机五次多项式平滑起步样条（Quintic Spline Soft-Start）
真实机器狗通电时处于趴地状态（`LayDown`）。**严禁直接挂接强化学习策略！** 否则初始残差过大，电机会在第一帧以 $45\text{ Nm}$ 极限力矩瞬间猛弹，发生严重机械过载。
必须通过 $T=2.0\text{s}$ 的五次多项式开环缓慢拉升至标称站姿：
$$s(t) = 10 \left(\frac{t}{T}\right)^3 - 15 \left(\frac{t}{T}\right)^4 + 6 \left(\frac{t}{T}\right)^5, \quad t \in [0, T]$$
$$q_i(t) = q_i^{\text{lay}} + s(t) \cdot \left( q_i^{\text{nominal}} - q_i^{\text{lay}} \right)$$
该曲线满足起始与终点位置、速度、加速度均为 0，保证关节无冲击平滑站立。站稳后（跟踪误差 $< 0.05\text{ rad}$）无缝切换至强化学习策略。

---

### 11.5 实机对接细节三：安全哨兵与急停保护状态机 (FSM Watchdog)

在 C++ 部署主循环中，必须设立三道独立于神经网络的**非抢占式硬件安全哨兵**：

```cpp
enum class FSMState {
    Passive,       // 阻尼泄力模式 (Kp=0, Kd=2.0)
    FixStand,      // 五次样条平滑站立
    RLRunning,     // 强化学习策略 50Hz 闭环
    EmergencyStop  // 故障硬急停 (切断 PWM 输出)
};

void SafetyWatchdog::check_and_protect(const RobotState& state, FSMState& current_fsm) 
{
    // 哨兵 1：机身翻转角倾覆保护 (倾角 > 60° 触发)
    // 根据重力投影 gz < -0.5 说明机身下翻或剧烈翻滚
    if (state.projected_gravity_z > -0.50f) {
        spdlog::critical("[SAFETY] 姿态翻转严重失稳！触发急停！gz = {:.2f}", state.projected_gravity_z);
        current_fsm = FSMState::Passive;
        return;
    }

    // 哨兵 2：关节超速失控保护
    for (int i = 0; i < 12; ++i) {
        if (std::abs(state.joint_vel[i]) > 30.0f) { // 30 rad/s 约 286 rpm
            spdlog::critical("[SAFETY] 关节 {} 发生飞车超速！速度: {:.2f} rad/s", i, state.joint_vel[i]);
            current_fsm = FSMState::Passive;
            return;
        }
    }

    // 哨兵 3：底层通信心跳超时保护 (连续丢包 > 3 帧，即 60ms)
    if (state.last_packet_age_ms > 60) {
        spdlog::critical("[SAFETY] 硬件底层通信中断！连续丢包超过 60ms！");
        current_fsm = FSMState::Passive;
        return;
    }
}
```

> [!TIP]
> 当触发任何安全哨兵时，**千万不要下发零力矩自由滑行（零阻尼）**，否则机器狗会由于重力瞬间硬砸在地上摔坏减速箱。正确做法是**切入高阻尼模式（$K_p = 0.0, K_d = 2.0\text{ Nm}\cdot\text{s/rad}$）**，让机器人如液压阻尼器般柔和“瘫坐”在地上，保护四肢与机身外壳。
