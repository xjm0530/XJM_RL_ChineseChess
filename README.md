# 基于模仿学习与PPO的中国象棋AI设计与实现

演示视频补充在data/videos下

## 项目简介

本项目设计并实现了一个基于**模仿学习（Imitation Learning）**和**近端策略优化（Proximal Policy Optimization, PPO）**的中国象棋人工智能系统。项目采用四阶段递进策略：首先通过Pikafish象棋引擎自对弈生成8万局专家对局数据，使用监督学习训练ResNet10策略网络；然后基于模仿学习预训练模型，使用PPO强化学习算法进行策略微调；最后将训练好的神经网络与蒙特卡洛树搜索（MCTS）结合，在推理阶段进一步提升决策质量。

## 目录结构

```
XJM_ChineseChess/
├── config.yaml                  # 项目配置文件
├── requirements.txt             # Python依赖包列表
├── PROJECT_README.md            # 项目说明文档
├── data/
│   ├── games/                   # 对局数据（.npz格式）
│   │   ├── games_round_*.npz        # 普通模式对局数据
│   │   └── games_high_quality_*.npz # 高质量模式对局数据
│   ├── jsons/                   # 训练日志与评估结果（JSON格式）
│   │   ├── imitation_training_log.json   # 模仿学习训练日志
│   │   ├── rl_training_log.json          # 强化学习训练日志
│   │   └── evaluation_result.json        # 模型评估结果
│   ├── figures/                 # 训练曲线与评估图表（PNG/PDF格式）
│   │   ├── imitation_training_curves.png  # 模仿学习损失与准确率曲线
│   │   ├── imitation_stage_comparison.png # 三阶段性能对比
│   │   ├── rl_training_curves.png          # PPO训练综合曲线
│   │   ├── rl_reward_curve.png            # PPO奖励变化曲线
│   │   ├── evaluation_bar_chart.png        # 模型ELO等级分对比
│   │   ├── battle_results_heatmap.png      # 模型对战热力图
│   │   └── model_comparison_radar.png      # 模型能力雷达图
│   └── models/                  # 模型权重保存目录
│       ├── resnet10_best.pth         # 模仿学习最佳模型
│       └── ppo_xiangqi_best.zip      # PPO最佳模型
├── engines/                     # 象棋引擎目录
│   ├── pikafish-*.exe               # Pikafish引擎各版本
│   └── pikafish.nnue                # NNUE评估网络
├── scripts/                     # 训练与评估脚本
│   ├── train.py                     # 统一训练入口脚本
│   └── evaluate_models.py           # 模型综合评估脚本
└── src/                         # 源代码
    ├── model/
    │   └── resnet.py                # ResNet骨干网络、ResNet10、PolicyValueNet
    ├── trainer/
    │   ├── game_generator.py        # 对局数据生成器
    │   ├── imitation_trainer.py     # 模仿学习训练器
    │   ├── rl_trainer.py            # PPO强化学习训练器
    │   └── mcts_evaluator.py        # MCTS评估器
    └── utils/
        ├── common.py                # 通用工具函数
        └── data_utils.py            # 数据加载与处理工具
```

## 技术架构

### 1. 核心模型

本项目基于深度残差网络（ResNet）构建策略网络，所有模型定义在 [src/model/resnet.py](file:///d:/作业及项目/大三下/强化学习/强化学习大作业/XJM_ChineseChess/src/model/resnet.py) 中。

#### ResNetBackbone
共享的残差骨干网络，包含4个残差层，通道数从64逐层递增到512。

#### ResNet10
- **输入**：14通道10×9的棋盘状态张量
- **结构**：ResNetBackbone + 自适应平均池化 + 全连接层
- **输出**：8100维动作概率分布（90起始位置 × 90目标位置）
- **用途**：模仿学习策略网络、PPO的特征提取器

#### PolicyValueNet
- **结构**：ResNetBackbone + 策略头 + 价值头
- **策略头**：1×1卷积 → 上采样 → 扁平化 → 全连接，输出8100维动作概率
- **价值头**：1×1卷积 → 池化 → 扁平化 → MLP → Tanh，输出标量价值（-1~1）
- **用途**：MCTS搜索的策略价值网络

### 2. 棋盘编码

采用14通道二值平面表示棋盘状态，定义在 [src/utils/common.py](file:///d:/作业及项目/大三下/强化学习/强化学习大作业/XJM_ChineseChess/src/utils/common.py) 中：

| 通道 | 红方棋子 | 通道 | 黑方棋子 |
|------|---------|------|---------|
| 0 | 帅 (K) | 7 | 将 (k) |
| 1 | 仕 (A) | 8 | 士 (a) |
| 2 | 相 (B) | 9 | 象 (b) |
| 3 | 马 (N) | 10 | 马 (n) |
| 4 | 车 (R) | 11 | 车 (r) |
| 5 | 炮 (C) | 12 | 炮 (c) |
| 6 | 兵 (P) | 13 | 卒 (p) |

**视角统一**：始终从当前走棋方的视角编码棋盘，黑方走棋时自动翻转棋盘和棋子通道。

### 3. 动作空间

采用 90×90 = 8100 的扁平动作空间表示：
- `action = from_square × 90 + to_square`
- `from_square / to_square = row × 9 + col`
- 非法动作由动作包装器自动回退到第一个合法动作

### 4. 训练流水线

#### 阶段一：对局数据生成

[src/trainer/game_generator.py](file:///d:/作业及项目/大三下/强化学习/强化学习大作业/XJM_ChineseChess/src/trainer/game_generator.py)

使用Pikafish UCI象棋引擎进行自对弈，生成带标签的监督学习数据：

- **普通模式**：12轮，每步思考时间短，用于快速积累大量数据
- **高质量模式**：4轮，每步思考时间长，用于精细调优
- **多进程加速**：支持并行生成，提高数据产出效率
- **数据格式**：`(board, action)` 对，board为14×10×9浮点张量，action为0~8099整数
- **存储格式**：压缩npz文件，支持内存映射读取

#### 阶段二：模仿学习

[src/trainer/imitation_trainer.py](file:///d:/作业及项目/大三下/强化学习/强化学习大作业/XJM_ChineseChess/src/trainer/imitation_trainer.py)

三阶段渐进式训练策略：

| 阶段 | 数据来源 | 学习率 | Epochs |
|------|---------|--------|--------|
| 1 | 普通对局前5轮 | 0.001 | 80 |
| 2 | 普通对局后6轮 | 0.0005 | 80 |
| 3 | 高质量对局 | 0.0002 | 80 |

- 每epoch采样30万训练样本、5万验证样本
- 学习率调度：每20 epoch衰减0.5倍
- 自动保存验证准确率最高的模型
- 训练日志以JSON格式持久化

#### 阶段三：PPO强化学习

[src/trainer/rl_trainer.py](file:///d:/作业及项目/大三下/强化学习/强化学习大作业/XJM_ChineseChess/src/trainer/rl_trainer.py)

基于Stable Baselines3的PPO实现：

- **特征提取器**：ResNet10骨干网络（加载模仿学习预训练权重）
- **策略网络**：两层256单元MLP
- **价值网络**：两层256单元MLP
- **关键超参数**：
  - 总步数：2,000,000
  - 学习率：3e-4
  - n_steps：2048
  - batch_size：64
  - n_epochs：10
  - gamma：0.99
  - gae_lambda：0.95
  - ent_coef：0.01

### 5. 环境包装器

三个Gym环境包装器位于 [src/trainer/rl_trainer.py](file:///d:/作业及项目/大三下/强化学习/强化学习大作业/XJM_ChineseChess/src/trainer/rl_trainer.py)：

#### XiangqiObsWrapper
将gym-xiangqi环境的(10,9)整数状态转换为14通道(14,10,9)浮点张量，自动处理视角翻转。

#### XiangqiActionWrapper
将8100动作空间映射到环境原生的129600动作空间，包含：
- 动作坐标转换与视角翻转
- 合法性校验
- 非法动作自动回退到第一个合法动作

#### XiangqiRewardWrapper
自定义奖励函数，包含以下组成部分：

| 奖励项 | 说明 | 配置值 |
|--------|------|--------|
| 吃子奖励 | 吃掉对方棋子的价值奖励 | 兵1.0，过河兵2.0，士2.0，象2.0，马4.0，炮4.5，车9.0，将1000.0 |
| 重复惩罚 | 局面重复3次以上的惩罚 | -50.0 |
| 步数惩罚 | 每走一步的惩罚，鼓励快速获胜 | 前50步-0.1，50步后-0.5 |

### 6. MCTS评估器

[src/trainer/mcts_evaluator.py](file:///d:/作业及项目/大三下/强化学习/强化学习大作业/XJM_ChineseChess/src/trainer/mcts_evaluator.py)

实现了蒙特卡洛树搜索（MCTS），可与策略价值网络结合使用：

- **PUCT选择公式**：平衡探索与利用
- **支持两种模型**：自动识别ResNet10（纯策略）或PolicyValueNet（策略+价值）
- **默认配置**：c_puct=1.0，模拟次数200次
- 可用于对弈评估或生成更高质量的训练数据

### 7. 棋力测评模块

[scripts/evaluate_models.py](file:///d:/作业及项目/大三下/强化学习/强化学习大作业/XJM_ChineseChess/scripts/evaluate_models.py)

实现了系统化的棋力评估，包含三个核心功能：

#### ELO等级分计算

- **EloCalculator**类封装标准ELO更新逻辑
- 参数配置：K=24，初始分=1200
- 期望得分公式：$E_A = 1/(1+10^{(R_B-R_A)/400})$
- 分数更新公式：$R_A' = R_A + K \cdot (S_A - E_A)$

#### 自适应对手匹配

8个不同难度的对手池，从完全随机到深度搜索：

| 对手 | ELO分 | 策略 | 对应人类水平 |
|------|-------|------|-------------|
| Random | 100 | 随机走法 | 无棋力 |
| Greedy | 250 | 贪心吃子 | 初学者 |
| Level 1 | 400 | 简单搜索 | 业余初级 |
| Level 2 | 600 | 中等搜索 | 业余中级 |
| Level 3 | 800 | 较深搜索 | 业余高级 |
| Level 4 | 1000 | 深度搜索 | 业余高手 |
| Level 5 | 1200 | 强力搜索 | 地方棋院等级 |
| Level 6 | 1400 | 专家搜索 | 接近职业水平 |

测评机制：每轮50局，胜率>70%升级，<20%降级，共20轮

#### 模型间对战

- 支持纯策略（IL）、强化学习（RL）、RL+MCTS三种推理模式
- 两模型交替执红执黑，消除先手优势
- 每组对战200局，统计胜负分布和非法走法率
- 评估结果保存至`data/jsons/evaluation_result.json`
- 训练曲线与评估图表保存至`data/figures/`

## 使用方法

### 环境配置

**Python 环境**：Anaconda 虚拟环境  
**环境路径**：`D:\Anacoda\envs\xjm\python.exe`  
**Python 版本**：3.8+

**GPU 环境**：
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- CUDA 版本：12.1
- cuDNN 版本：8.8.1

安装依赖：
```bash
pip install -r requirements.txt
```

安装 CUDA 版本的 PyTorch（如需 GPU 加速）：
```bash
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 配置文件

所有超参数和路径配置在 [config.yaml](file:///d:/作业及项目/大三下/强化学习/强化学习大作业/XJM_ChineseChess/config.yaml) 中：

- `game`：对局生成配置（轮次、时间、进程数等）
- `imitation`：模仿学习训练配置
- `rl`：PPO强化学习配置
- `mcts`：MCTS搜索配置
- `paths`：数据、模型、引擎路径
- `rewards`：奖励函数权重

### 训练流程

#### 1. 生成对局数据
```bash
python scripts/train.py --stage generate
```

使用Pikafish引擎自对弈生成训练数据，数据保存在 `data/games/` 目录。

#### 2. 模仿学习训练
```bash
python scripts/train.py --stage imitation
```

三阶段渐进式训练ResNet10策略网络，模型保存在 `data/models/resnet10_best.pth`。

#### 3. PPO强化学习微调
```bash
python scripts/train.py --stage rl
```

基于模仿学习预训练模型进行PPO强化学习微调。

#### 4. 模型评估
```bash
python scripts/train.py --stage evaluate
# 或直接运行评估脚本
python scripts/evaluate_models.py
```

评估指标包括：
- **ELO等级分**：通过自适应对手匹配计算，综合衡量模型棋力
- **非法动作率**：模型输出非法动作的比例
- **模型间胜率**：IL vs RL、IL vs RL+MCTS、RL vs RL+MCTS三组200局对战

评估结果保存在 `data/jsons/evaluation_result.json`，可视化图表保存在 `data/figures/`。

### 直接运行各模块

**数据生成**：
```bash
python -m src.trainer.game_generator
```

**模仿学习**：
```bash
python -m src.trainer.imitation_trainer
```

**强化学习**：
```bash
python -m src.trainer.rl_trainer --mode train
```

**时间估算**：
```bash
python -m src.trainer.rl_trainer --mode estimate
```

**MCTS评估**：
```bash
python -m src.trainer.mcts_evaluator
```

## 工具函数库

### common.py
[src/utils/common.py](file:///d:/作业及项目/大三下/强化学习/强化学习大作业/XJM_ChineseChess/src/utils/common.py)

包含通用工具函数：
- `load_config` / `save_config`：YAML配置读写
- `get_device`：获取可用计算设备
- `set_seed`：设置随机种子
- `ensure_dir`：确保目录存在
- `save_model` / `load_model`：模型保存加载（自动兼容新旧格式）
- `encode_board` / `encode_board_from_state` / `encode_board_from_char`：多种输入格式的棋盘编码
- `flip_perspective`：视角翻转
- `flip_action`：动作坐标翻转
- `piece_id_to_symbol`：棋子ID到符号的映射

### data_utils.py
[src/utils/data_utils.py](file:///d:/作业及项目/大三下/强化学习/强化学习大作业/XJM_ChineseChess/src/utils/data_utils.py)

包含数据加载工具：
- `RoundDataset`：单轮对局数据集，内存映射读取，自动过滤无效样本
- `FastImitationDataset`：跨文件数据集，支持索引子集
- `create_train_val_datasets`：快速创建训练验证集

## 依赖环境

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | 2.3.0 (CUDA 12.1) | 深度学习框架 |
| numpy | 1.24.4 | 数值计算 |
| gym | 0.26.2 | 强化学习环境框架 |
| gym-xiangqi | 0.0.3 | 中国象棋 Gym 环境 |
| stable-baselines3 | 2.4.1 | PPO 强化学习算法 |
| pandas | 2.0.3 | 数据处理 |
| pyyaml | 6.0.2 | YAML 配置解析 |
| tqdm | 4.66.1 | 进度条显示 |

## 设计特点

1. **四阶段训练**：数据生成 → 模仿学习 → PPO强化学习 → MCTS推理增强，渐进式提升棋力
2. **分阶段数据策略**：前6万场0.05s快速积累 + 后2万场0.2s高质量对局，共8万场16轮
3. **共享骨干网络**：ResNetBackbone在模仿学习、PPO和MCTS间复用，减少代码冗余
4. **统一编码规范**：14通道one-hot棋盘编码 + 视角统一 + 8100维动作空间，各模块一致
5. **灵活的奖励设计**：吃子奖励 + 重复惩罚 + 步数惩罚，可通过配置文件调整
6. **内存高效数据加载**：npz内存映射读取，处理百万级样本
7. **MCTS推理增强**：PUCT选择 + 神经网络先验，200次模拟提升决策质量
8. **系统化棋力测评**：ELO等级分 + 8级自适应对手池 + 模型间对战，全面量化棋力
