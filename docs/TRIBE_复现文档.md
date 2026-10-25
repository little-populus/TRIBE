# TRIBE（AAAI-24）实验复现文档

## 从 GitHub 下载仓库到完整执行 CIFAR-10-C 实验（GPU 实现）

> 项目：Towards Real-World Test-Time Adaptation: Tri-Net Self-Training with Balanced Normalization
> 论文：https://arxiv.org/abs/2309.14949（AAAI-24 接收）
> 源码仓库：https://github.com/Gorilla-Lab-SCUT/TRIBE.git

本文档完整记录了从零开始复现 TRIBE 实验的全过程，包括：从 GitHub 下载源码、创建虚拟环境、安装依赖、配置 GPU 计算环境、下载数据集与预训练权重，以及最终在 CIFAR-10-C 数据集上运行完整的 GLI-TTA 实验并得到结果。

---

## 一、项目背景

TRIBE（Tri-Net Self-Training with Balanced Normalization）是一个面向真实世界测试时自适应（Test-Time Adaptation, TTA）的方法，发表于 AAAI-2024。其核心贡献包括：

1. **Balanced BatchNorm 模块**：缓解对局部和全局类别不平衡测试数据的估计偏差。普通 BN 模块在批次数据量小或批次数据存在类别偏差时容易产生估计偏差。
2. **Tri-Net 架构**：降低过度适应当地分布的风险，提供稳定鲁棒的 TTA 流程。
3. 结合上述两个模块，TRIBE 可以处理不同类型的测试数据流（i.i.d. 测试流、局部和全局类别不平衡测试流），并稳定安全地适应目标域。

### 实验目标

本实验采用 GLI-TTA（Global and Local Class Imbalanced Test-Time Adaptation）协议，在 CIFAR-10-C 数据集上评估 TRIBE 方法的性能。CIFAR-10-C 包含 15 种常见图像损坏类型（高斯噪声、模糊、天气影响等），每种损坏有 5 个严重程度。本实验使用严重程度 5 的数据。

---

## 二、环境准备

### 2.1 从 GitHub 下载源码

```bash
git clone https://github.com/Gorilla-Lab-SCUT/TRIBE.git
cd TRIBE
```

### 2.2 系统环境说明

本复现实验的运行环境为 **Windows 系统**，配备 **NVIDIA GeForce RTX 5070 Ti GPU**（16GB 显存，Blackwell 架构）。

- 操作系统：Windows
- Python 版本：3.9.13
- 计算设备：NVIDIA RTX 5070 Ti（GPU）
- 显卡驱动：CUDA UMD 13.5

> **重要提示**：RTX 5070 Ti 属于 Blackwell 架构（sm_120），需要 **CUDA 12.8 及以上**版本的 PyTorch 才能正常支持。

### 2.3 创建虚拟环境

原项目使用 conda 创建环境。本复现实验使用 Python 自带的 `venv` 在项目目录下创建名为 `exp` 的虚拟环境：

```powershell
# Windows PowerShell
cd C:\path\to\TRIBE
python -m venv exp

# 激活虚拟环境
.\exp\Scripts\activate
```

> 说明：原 README 使用 `conda create -n tribe python=3.9.0`，这里改用 `venv` 创建 `exp` 环境，效果等价，均用于隔离项目依赖。

---

## 三、安装依赖

### 3.1 安装 requirements.txt 中的基础依赖

```powershell
pip install prettytable==3.4.1 yacs==0.1.8 argparse==1.4.0 tqdm==4.56.2 setproctitle==1.3.2
```

### 3.2 安装 GPU 版 PyTorch 与 torchvision

本机 GPU 为 RTX 5070 Ti（Blackwell 架构），需要 CUDA 12.8 及以上。从 PyTorch 官方 CUDA 12.8 索引下载：

```powershell
# 设置代理（如需要）
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
$env:HTTP_PROXY = "http://127.0.0.1:7897"

# 安装 CUDA 12.8 版 torch 和 torchvision
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

安装结果：
- `torch 2.8.0+cu128`（CUDA 12.8）
- `torchvision 0.23.0+cu128`

> **版本选择说明**：
> - RTX 5070 Ti 为 Blackwell 架构（sm_120），需要 CUDA 12.8+；
> - Python 3.9（cp39）限制了可用版本——cu132 索引不含 cp39 版本（仅 cp310+），故选择 cu128；
> - `torch 2.8.0+cu128` 是同时满足「Blackwell 支持 + Python 3.9」的最优选择。

torch 安装时会自动依赖 nvidia CUDA 运行库（nvidia-cublas-cu12、nvidia-cudnn-cu12、nvidia-cuda-runtime-cu12 等），这些会一并安装。

### 3.3 安装其他依赖

```powershell
# 科学计算与机器学习
pip install scikit-learn scipy

# robustbench 运行时依赖（注意 timm 必须为 0.6.x 版本）
pip install timm==0.6.13 requests pandas
pip install geotorch torchdiffeq
```

> **重要提示**：`timm` 必须安装 0.6.x 版本。若安装 1.0.x 版本，导入 robustbench 时会报错：
> `AttributeError: module 'timm.models.xcit' has no attribute 'XCiT'`。

### 3.4 安装 robustbench

robustbench 是项目自带的模型库子目录，用于加载预训练模型。以开发模式安装（`--no-deps` 跳过其 setup.py 中声明的不兼容版本约束）：

```powershell
cd robustbench
pip install --no-deps -e .
cd ..
```

### 3.5 验证 GPU 环境

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期输出：

```
2.8.0+cu128
True
NVIDIA GeForce RTX 5070 Ti
```

---

## 四、下载数据集与预训练权重

### 4.1 网络代理说明

CIFAR-10-C 数据集托管在 Zenodo，预训练权重托管在 Google Drive，二者在国内直连速度极慢。本实验使用本地 Clash 代理（端口 7897）加速下载。

### 4.2 下载 CIFAR-10-C 数据集（约 2.72 GB）

CIFAR-10-C 的 Zenodo 记录 ID 为 `2535967`，文件为 `CIFAR-10-C.tar`（MD5：`56bf5dcef84df0e2308c6dcbcbbd8499`）。

使用 curl 通过代理下载（`-C -` 支持断点续传）：

```powershell
mkdir datasets
cd datasets
curl -x http://127.0.0.1:7897 -L -C - -o CIFAR-10-C.tar `
  "https://zenodo.org/api/records/2535967/files/CIFAR-10-C.tar/content"
```

下载完成后校验 MD5 并解压：

```powershell
# 校验 MD5（应输出 56bf5dcef84df0e2308c6dcbcbbd8499）
certutil -hashfile CIFAR-10-C.tar MD5

# 解压
tar -xf CIFAR-10-C.tar
```

解压后的目录结构（含 15 种损坏的 `.npy` 文件 + `labels.npy`）：

```
datasets/CIFAR-10-C/
├── brightness.npy        # 亮度
├── contrast.npy          # 对比度
├── defocus_blur.npy      # 散焦模糊
├── elastic_transform.npy # 弹性变换
├── fog.npy               # 雾
├── frost.npy             # 霜
├── gaussian_blur.npy     # 高斯模糊
├── gaussian_noise.npy    # 高斯噪声
├── glass_blur.npy        # 玻璃模糊
├── impulse_noise.npy     # 脉冲噪声
├── jpeg_compression.npy  # JPEG 压缩
├── labels.npy            # 标签
├── motion_blur.npy       # 运动模糊
├── pixelate.npy          # 像素化
├── saturate.npy          # 饱和
├── shot_noise.npy        # 散粒噪声
├── snow.npy              # 雪
├── spatter.npy           # 溅射
└── zoom_blur.npy         # 缩放模糊
```

### 4.3 下载预训练模型权重（Standard / ResNet-50）

TRIBE 实验使用 `Standard`（ResNet-50）架构，权重从 Google Drive 下载，gdrive_id 为 `1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC`。

**关键问题**：robustbench 自带的 `download_gdrive` 使用旧域名 `docs.google.com/uc`，对大于一定体积的文件会返回 "Virus scan warning" 页面（HTML），而非真实权重文件，导致 `torch.load` 失败。

**解决方案**：改用 Google Drive 的新下载域名 `drive.usercontent.google.com/download`，并带上 `confirm=t` 和从警告页表单提取的 `uuid` 参数：

```powershell
mkdir ckpt\cifar10\corruptions

# 第一步：访问获取 uuid（从返回的 HTML 中提取 uuid 和 confirm 值）
curl -s -L -x http://127.0.0.1:7897 `
  "https://docs.google.com/uc?export=download&id=1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC" `
  -o gdrive_page.html

# 从页面中提取 uuid（示例值，实际每次可能不同）
# uuid="015d91ca-c1b7-4b49-832a-8bd41dbb5645", confirm="t"

# 第二步：使用新域名 + confirm + uuid 下载真实文件
curl -s -L -x http://127.0.0.1:7897 `
  "https://drive.usercontent.google.com/download?id=1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC&export=download&confirm=t&uuid=015d91ca-c1b7-4b49-832a-8bd41dbb5645" `
  -o ckpt\cifar10\corruptions\Standard.pt
```

下载完成后，`Standard.pt` 文件大小约 278 MB（291,987,963 字节），模型参数量 36.48 M。

### 4.4 修复 torch 2.8 的 torch.load 兼容性问题

PyTorch 2.6+ 将 `torch.load` 的默认 `weights_only` 参数从 `False` 改为 `True`，导致加载旧版 checkpoint 时抛出 `UnpicklingError`。

需要在 `robustbench/robustbench/utils.py` 中所有 `torch.load` 调用处添加 `weights_only=False`：

```python
# 修改前
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
# 修改后
checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
```

共需修改 8 处 `torch.load` 调用。

---

## 五、运行实验

### 5.1 实验配置说明

- **数据集配置**（`configs/dataset/cifar10.yaml`）：15 种损坏类型，严重程度 5，批大小 64
- **适配器配置**（`configs/adapter/TRIBE.yaml`）：TRIBE 算法，超参数 ETA=0.01、GAMMA=0.0、H0=0.05、LAMBDA=0.5
- **协议配置**（`configs/protocol/gli_tta.yaml`）：GLI-TTA 协议，全局不平衡因子 IMB_FACTOR=10，类别比例 constant，局部不平衡因子 GAMMA=0.1

### 5.2 运行命令

```powershell
# Windows PowerShell
cd C:\path\to\TRIBE

# 运行 TRIBE 在 CIFAR-10-C 上的 GLI-TTA 实验
.\exp\Scripts\python.exe GLI_TTA.py `
  -acfg configs\adapter\TRIBE.yaml `
  -dcfg configs\dataset\cifar10.yaml `
  -pcfg configs\protocol\gli_tta.yaml `
  OUTPUT_DIR TRIBE\cifar10_gpu
```

### 5.3 实验规模与耗时

- 15 种损坏类型 × 10,000 样本 = 150,000 个测试样本
- 在 GPU（RTX 5070 Ti）上运行，约 15 分钟完成
- 单样本平均适配时间约 0.015 秒

---

## 六、实验结果

### 6.1 各损坏类型的准确率

| 损坏类型 | 准确率 (%) | 错误率 (%) |
|---------|-----------|-----------|
| brightness（亮度） | 90.94 | 9.06 |
| fog（雾） | 88.15 | 11.85 |
| contrast（对比度） | 87.24 | 12.76 |
| zoom_blur（缩放模糊） | 86.14 | 13.86 |
| frost（霜） | 83.59 | 16.41 |
| motion_blur（运动模糊） | 82.52 | 17.48 |
| snow（雪） | 81.95 | 18.05 |
| defocus_blur（散焦模糊） | 81.95 | 18.05 |
| elastic_transform（弹性变换） | 78.23 | 21.77 |
| pixelate（像素化） | 76.08 | 23.92 |
| gaussian_noise（高斯噪声） | 75.32 | 24.68 |
| shot_noise（散粒噪声） | 74.02 | 25.98 |
| jpeg_compression（JPEG 压缩） | 73.80 | 26.20 |
| glass_blur（玻璃模糊） | 65.62 | 34.38 |
| impulse_noise（脉冲噪声） | 59.28 | 40.72 |
| **平均** | **78.99** | **21.01** |

### 6.2 总体结果

- **平均准确率：78.99%**
- **平均错误率：21.01%**
- **平均适配时间：0.0147 秒/样本**

### 6.3 结果说明

- 实验结果与论文报告的水平一致（TRIBE 在 CIFAR-10-C 严重程度 5 上的平均准确率约 78-80%）。
- 亮度（brightness）、雾（fog）、对比度（contrast）等轻度损坏的准确率最高（87-91%），脉冲噪声（impulse_noise）的准确率最低（59%），符合 TTA 方法在强噪声下的普遍表现。

---

## 七、常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `ModuleNotFoundError: No module named 'torch'` | 未安装 PyTorch | 按 3.2 节安装 GPU 版 torch |
| `timm.models.xcit has no attribute 'XCiT'` | timm 版本过高（1.0.x） | 降级到 `timm==0.6.13` |
| `ModuleNotFoundError: No module named 'autoattack'` | autoattack 未发布到 PyPI | 将 eval.py 中 autoattack 改为可选导入 |
| `UnpicklingError: invalid load key, '<'` | 下载的是 Google Drive 警告页 HTML | 按 4.3 节改用新域名 + confirm + uuid 下载 |
| `Weights only load failed` | torch 2.6+ 默认 weights_only=True | 加 `weights_only=False` 参数 |
| `no kernel image available for execution on the device` | torch 的 CUDA 版本过低，不支持 Blackwell | 使用 CUDA 12.8+ 的 torch |
| CUDA 不可用（`torch.cuda.is_available()=False`） | 装了 CPU 版 torch | 卸载后改装 `--index-url .../cu128` 版本 |
| 数据集下载中断 | 网络不稳定 | 用 curl `-C -` 断点续传 |

---

## 八、致谢

本项目基于以下开源项目：
- [TRIBE](https://github.com/Gorilla-Lab-SCUT/TRIBE)（AAAI-24）
- [RoTTA](https://github.com/BIT-DA/RoTTA)
- [robustbench](https://github.com/RobustBench/robustbench)

## 引用

```bibtex
@article{Su_Xu_Jia_2024,
  title={Towards Real-World Test-Time Adaptation: Tri-net Self-Training with Balanced Normalization},
  author={Su, Yongyi and Xu, Xun and Jia, Kui},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  DOI={10.1609/aaai.v38i13.29435},
  pages={15126-15135},
  volume={38}, number={13},
  month={Mar.}, year={2024},
}
```
