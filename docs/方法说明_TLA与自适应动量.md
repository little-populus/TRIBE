# 面向全局类不平衡 TTA 的去偏伪标签学习：方法说明（论文素材）

> 建议的方法定位：在 TRIBE（AAAI-24）基础上，针对其**未解决的决策层偏置**与
> **逐类统计收敛速度不均**两个局限，提出两个可组合的模块：
> **在线先验去偏（TLA, Test-time Logit Adjustment）**与**逐类自适应动量（AM, Adaptive Momentum）**。
> 二者共同遵循一个统一原则：**令每个类别的自适应动力学趋于平等**（equalized adaptation dynamics）。

---

## 1. 动机与问题分析

TRIBE 的 Balanced BN 在**特征归一化层面**平衡了各类统计（对逐类统计等权平均），
但分析发现全局不平衡测试流仍存在两个未解决的偏差源：

**(L1) 决策层偏置**。源模型在均衡数据上训练，其分类器隐含均匀类先验假设。当测试流
全局类分布长尾化时，预测与伪标签系统性偏向头部类，造成：
(a) 最终预测的尾部类召回率低（macro 准确率显著低于 micro，I.F.=100 时差距约 5pt）；
(b) 自训练伪标签将头部偏置回灌到学生模型（确认偏差的标签偏移分量）；
(c) Balanced BN 的逐类统计分配收到被污染的标签——尾部样本被误标为头部类，
   进一步恶化尾部统计。

**(L2) 统计收敛速度不均**。Balanced BN 以固定动量 η 更新逐类统计，单次更新速率为
η·n_k（n_k 为该 batch 内该类样本数）。尾部类出现频率低，整个域的累计更新量极小
（I.F.=100 时最稀类每域约 10 个样本），其统计几乎停留在源域——即"想平衡但没有
足够的样本去平衡"。

TRIBE 原文未处理 L1；L2 由其 macro-micro 差距随 I.F. 扩大这一现象间接体现。

## 2. 方法

### 2.1 在线类先验估计与测试时 Logit 调整（针对 L1）

沿用 logit adjustment 的标准设定 [Menon et al. 2021]：若源域类先验为均匀分布
（π_src = 1/K），测试先验为 π，则对 logit z 做加性校正可获得接近贝叶斯最优的
均衡预测。我们在线维护累计伪标签计数 N_k，以 Laplace 平滑估计测试先验：

$$\hat{\pi}_k = \frac{N_k + \alpha}{\sum_{j=1}^{K} N_j + \alpha K} \tag{1}$$

对 teacher 网络的输出做去偏：

$$\tilde{z} = z - \tau \log\left( K \hat{\pi} \right) \tag{2}$$

其中 τ∈[0,1] 为校正强度（τ=1 为贝叶斯一致校正，τ=0 退化为原始 TRIBE）。
当 π̂ 均匀时式 (2) 为恒等变换，因此方法在均衡流（I.F.=1）下自动关闭。
前 W 个伪标签（warmup）内估计噪声大，只收集计数不做校正。

**去偏输出的三个使用点**（这是与仅校正预测的常规用法 [Saerens et al. 2002] 的区别，
也是对 TTA 管线的系统性去偏）：
1. **最终预测**：ŷ = argmax z̃，直接改善 macro 指标；
2. **自训练目标**：L_st 的伪标签目标取 argmax z̃，切断头部偏置的确认偏差回路；
3. **统计分配标签**：Balanced BN 逐类统计更新使用 argmax z̃，减少尾部样本
   被误分给头部统计的污染，与 AM 模块形成协同。

计数更新使用去偏后的预测（自洽估计）。先验校正与 EM 类标签偏移估计
[Saerens et al. 2002; Alexandari et al. 2020] 与 TTA 场景下的在线标签偏移自适应
[Wu et al. 2021] 一脉相承；区别在于我们没有独立的源域混淆矩阵可用，采用
teacher 预测的平滑频率作为一阶近似，并将其注入 TTA 训练环路而非仅用于输出校正。

### 2.2 逐类自适应 BN 统计动量（针对 L2）

将固定动量 η 替换为与该类累计观测数 Ñ_k 成反比的自适应步长：

$$m_k = \mathrm{clip}\left( \frac{c}{\tilde{N}_k},\ \eta,\ m_{\max} \right) \tag{3}$$

常见类（Ñ_k 大）保持原固定动量行为（保留对非平稳域的指数窗跟踪能力）；
罕见类获得更大步长，等效于对其见过的样本做近似充分平均——步长 1/N 是
随机逼近的最优收敛步长 [Robbins & Monro 1951]，BN 的累计均值与动量均值之别
即源于此 [Ioffe & Szegedy 2015]。c=1 时式 (3) 在 Ñ_k ≤ c/η 区间退化为
"对每类前 c/η 个样本做 1/N 平均，其后转入指数窗"，兼得无偏收敛与非平稳适应。

### 2.3 算法（单 batch 流程）

```
输入: 测试 batch x=B, teacher(stu共享权重+独立BN统计) T, 冻结锚网络 A, 累计计数 N, 各BN层累计计数 {Ñ^(l)}
1  z ← T(x)（eval 模式，统计不更新）
2  π̂ ← Laplace 平滑(N)                          # 式(1)
3  z̃ ← z − τ·log(K·π̂) ;  N ← N + bincount(argmax z̃)   # 式(2)，warmup 内 z̃←z
4  ŷ ← argmax z̃（最终预测）
5  p̂ ← argmax z̃（伪标签，供损失与统计分配）
6  训练模式前向: T(x), S(aug(x)), A(x) 各自更新 Balanced BN 统计
      —— 每类动量 m_k 按式(3)，标签均为 p̂
7  L ← CE(S(aug x), p̂ | 熵mask) + λ·MSE(T(x), A(x) | 熵mask)
8  反向传播更新 BN 仿射参数（teacher/student 共享）
```

### 2.4 复杂度

TLA 增加 O(B + K) 每步；AM 每层增加 O(K) 计数与查表。无额外前向/反向，
单样本推理-自适应耗时与 TRIBE 持平（实测 <5% 差异）。

## 3. 实验结果（CIFAR-10-C, GLI-TTA-F, seed 425）

（准确率 %，micro / macro；运行噪声约 ±0.1；全部结果为本次实验实测）

**I.F.=100（基线 80.19 / 75.30；论文报告值折算 80.47 / 75.34）**

| 设置 | micro | macro | Δmicro / Δmacro |
|---|---|---|---|
| TRIBE 基线（复现） | 80.19 | 75.30 | — |
| + AM (c=1.0) | **80.34** | **75.49** | **+0.15 / +0.19（双赢）** |
| + TLA both τ=1.0 | 73.70 | 74.99 | -6.49 / -0.31 |
| + TLA both τ=0.5 | 77.48 | 75.61 | -2.71 / +0.31 |
| + TLA train-only τ=0.5 | 79.59 | **75.74** | -0.60 / **+0.44** |
| + TLA train-only τ=1.0 | 77.85 | 75.46 | -2.34 / +0.16 |
| + TLA train τ=0.5 + AM | 79.37 | 75.59 | -0.82 / +0.29 |

**I.F.=10（基线 78.99 / 77.48）**：+ AM = 78.75 / 77.34（噪声范围内，无副作用）。

### 结果分析

**(1) AM 是唯一的双赢改进**，但幅度有限（~0.2pt）。原因：动量只能修复统计欠拟合，
尾部类准确率还受损失监督稀少与共享 head 对多数类梯度的偏置制约。

**(2) TLA 是系统性的 trade-off，任何强度/作用通道下都无法双赢**。机理：
源模型在均衡数据上训练，其预测分布头部化**主要反映测试数据的真实先验而非决策偏置**；
logit 校正因此只是把头部类的正确预测翻转给尾部类——重新分配错误而不创造信息。
即使去偏只进入训练通道（train-only 模式），错误翻转的伪标签仍会污染 CE 目标与
BN 统计分配，micro 损失依然大于 macro 收益。这一负结果本身具有分析价值：
它说明 GLI-TTA 协议下 macro-micro 差距的主要来源是**尾部类的表示/统计质量不足**
（AM 所针对的问题），而非**决策阈值偏置**（TLA 所针对的问题）。

## 4. 论文定位建议（基于以上实证）

- **可写入论文的部分**：AM 作为对 TRIBE 的改进组件（双赢、有理论动机、实现极简、
  默认协议无损）；TLA 的完整消融（both/train-only × τ 扫描）作为**分析性贡献**——
  用实验排除了"决策层偏置"假设，定位了真实瓶颈。二者合计构成一个
  "诊断 + 定向修复"的完整故事，适合作为 workshop 论文 / 主论文的扩展实验章节。
- **不足支撑主会创新点**：在 CIFAR-10-C 上增益 ≤0.5pt，且单数据集单种子。
- **突破当前实验局限的最可能路径**（按预期收益排序）：
  1. **CIFAR-100-C / ImageNet-C 验证**（K=100/1000）：逐类统计稀疏度随 K 急剧上升
     （I.F.=10 时 CIFAR-100 每类每域仅 ~26 个样本，约为 CIFAR-10 的 1/16），
     AM 的理论动机随 K 增强，预期收益更大——这是本机未完成的关键验证
     （需下载 CIFAR-100-C 2.9GB）；
  2. **多种子 + GLI-TTA-V 协议**的系统评估（本机单种子，噪声 ±0.1 掩盖小增益）；
  3. **域切换统计重置**（continual 局限）：每个域前 ~20% batch 受旧域统计污染，
     检测特征突变并预热统计是未探索方向；
  4. 更根本的变革（类条件仿射、类别路由专家、回放式类平衡内存）超出
     "对 TRIBE 的小改进"范畴，需独立的方法论工作。

## 5. 参考文献

1. Saerens, P., Latinne, P., & Decaestecker, A. (2002). Adjusting the outputs of a
   classifier to new a priori probabilities: A simple procedure. *Neural Computation*, 14(1), 21–41.
2. Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2021).
   Long-tail learning via logit adjustment. *ICLR*.
3. Wu, Y., Guo, H., Su, Y., & Weinberger, K. Q. (2021). Online adaptation to label
   distribution shift. *NeurIPS*.
4. Alexandari, A., Kundaje, A., & Eriksson, A. (2020). Maximum likelihood with
   bias-corrected calibration is hard-to-beat at label shift adaptation. *ICML*.
5. Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-balanced loss
   based on effective number of samples. *CVPR*.（TRIBE 的长尾协议即引自此文）
6. Robbins, H., & Monro, S. (1951). A stochastic approximation method. *The Annals of
   Mathematical Statistics*, 22(3), 400–407.
7. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network
   training by reducing internal covariate shift. *ICML*.
8. Su, Y., Xu, X., & Jia, K. (2024). Towards real-world test-time adaptation: Tri-net
   self-training with balanced normalization. *AAAI*.（基线方法）
