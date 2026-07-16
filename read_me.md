# Mini-GPT: Character-Level Language Model

这是一个基于 PyTorch 实现的字符级自回归语言模型（Character-Level Autoregressive Language Model）。本项目完整记录了从零手写 Transformer 底层数学逻辑，到运用工业级加速技术进行极致性能优化的全过程。不仅是一个高效的文本生成引擎，也是一个深入理解大语言模型（LLM）底层架构的绝佳实践库。

## 💡 核心演进与特性 (Key Features)

本项目在工程实现上分为**原理验证**与**性能压榨**两个阶段：

1. **原生基础实现 (Foundational Version)**：
   - 彻底摒弃 `nn.Transformer` 等高级封装，从最基础的线性层 (Linear Layer) 开始，纯手工实现了查询/键/值 (QKV) 矩阵切分、多头自注意力机制 (Multi-Head Self-Attention) 以及带有下三角掩码的注意力权重计算。
   - 代码结构高度契合底层数学公式，适合用作 Transformer 架构的源码级学习与剖析。

2. **工业级性能优化 (High-Performance Version)**：
   - 为解决原生 `for` 循环与显存读写带来的性能瓶颈，引入了现代深度学习框架的极致加速黑科技：
   - **FlashAttention** (`F.scaled_dot_product_attention`)：利用底层 C++ 融合算子实现内存高效的因果注意力计算。
   - **自动混合精度 (AMP)**：支持 `bfloat16` 混合精度训练，大幅降低显存占用并提升张量吞吐量。
   - **零拷贝数据加载 (Zero-Overhead DataLoader)**：采用 GPU 显存内 Tensor 切片切分 Batch，消除 CPU-GPU 通信瓶颈。
   - **硬件加速兼容**：启用 TF32 计算模式，并配置了 Fused Adam 优化器以加速参数更新。

3. **完整的工程化支持**：
   - **智能检查点机制 (Checkpointing)**：支持训练状态的实时保存与断点续训，并自动追踪记录具有最低 Loss 的最佳模型 (`best_model.pt`)。
   - **随机采样生成 (Sampling Generation)**：推理阶段采用多项式采样 (Multinomial Sampling) 替代贪心解码，显著提升文本生成的多样性。

## 📁 目录结构 (Project Structure)

```text
.
├── data/
│   ├── shakespeare.txt        # 原始训练语料集
│   └── encoded.txt            # 序列化后的 Token 数据
├── models/                    # 模型检查点保存目录 (.pt)
├── src/
│   ├── remap.py               # 分词器：构建字符级词表并序列化文本
│   ├── get_data.py            # 数据处理：基础数据加载模块
│   ├── save_model.py          # 状态管理：模型保存、加载与最优权重追踪
│   ├── Bigram.py              # 基线模型：二元语法模型对照
│   ├── transformer.py         # 📘 原生基础版：展示大模型底层的纯粹数学逻辑与数据流向
│   └── transformer_speed.py   # 🚀 极限加速版：面向实际训练的高性能重构版本
└── README.md
```

## 🛠️ 依赖环境 (Requirements)

- Python 3.8+
- PyTorch 2.0+ (以支持 FlashAttention)
- NVIDIA GPU (建议，以开启完整的混合精度与算子融合加速)

## 🚀 快速开始 (Quick Start)

### 1. 数据预处理
将训练语料放置于 `data/` 目录下（默认使用莎士比亚剧本集）。运行分词脚本构建词表并完成序列化：
```bash
python src/remap.py
```

### 2. 模型训练
**建议使用优化后的 `transformer_speed.py` 进行训练**以获得最佳性能。训练过程会自动检测可用硬件（CUDA/CPU），并在 `models/` 目录下定期更新检查点。
```bash
python src/transformer_speed.py
```
*注：中断训练后再次运行该脚本，系统会自动读取最新的检查点并无缝恢复训练进度。若想研究底层实现原理，可阅读或运行 `transformer.py`。*

### 3. 文本生成 (推理)
在 `transformer_speed.py` 训练阶段完成后（或手动中止训练循环），程序会自动进入推理交互模式。输入提示词后，模型将逐字符生成延续文本：
```text
📝 Type your prompt!
[Input your starting text here]
```

## ⚙️ 模型配置参考 (Architecture Details)

当前 `transformer_speed.py` 中的默认配置（约 ~10M 参数量，适用于基础显卡训练验证）：
- `vocab_size`: 取决于语料字符集大小
- `embed_size`: 384
- `network_depth`: 6 层 Transformer Block
- `head_num`: 6 个注意力头
- `seq_size`: 256 (上下文窗口大小)
- `batch_size`: 64
- `Optimizer`: Adam (lr=3e-4, fused=True)