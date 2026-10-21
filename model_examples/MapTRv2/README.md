# MapTRv2 应用说明

本文说明如何在产品镜像中，基于官方 MapTRv2 仓库应用 TurboPhysAI 的 HCU 优化。镜像获取和容器创建方法参见[安装指南](../../docs/zh/get_started/installation.md#使用产品镜像)。

## 1. 模型简介

[MapTRv2](https://github.com/hustvl/MapTR/tree/maptrv2) 是面向自动驾驶在线矢量化地图构建的模型，通过多摄像头图像学习鸟瞰视角（BEV）特征，预测车道分隔线、人行横道、道路边界等地图元素。本文使用 ResNet-50 Backbone、BEV pooling 和 nuScenes 24 epoch 配置：

```text
projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py
```

## 2. 优化接入基线

TurboPhysAI 的 MapTRv2 优化基于官方 `hustvl/MapTR` 仓库的 `maptrv2` 分支，锁定 commit：

```text
e03f097abef19e1ba3fed5f471a8d80fbfa0a064
```

## 3. 准备模型源码

首次准备模型仓库时执行：

```bash
cd /workspace
mkdir -p model
git clone -b maptrv2 https://github.com/hustvl/MapTR.git model/MapTrv2
cd /workspace/model/MapTrv2
git checkout --detach e03f097abef19e1ba3fed5f471a8d80fbfa0a064
mkdir -p data ckpts work_dirs

git rev-parse HEAD
git status --short
```

生成 OptimizationConfig 时模型工作区必须干净。不要在该基线仓库内克隆 TurboPhysAI，也不要为接入优化直接修改模型源码。数据、权重和训练输出使用基线已经忽略的 `data/`、`ckpts/`、`work_dirs/` 目录。

也可以使用已准备好的官方模型仓库。下文命令以 `/workspace/model/MapTrv2` 为模型根目录，实际部署时可调整路径。模型源码、数据集和权重需要单独准备，模型依赖范围见[官方安装说明](https://github.com/hustvl/MapTR/blob/maptrv2/docs/install.md)，具体版本应与 HCU 环境匹配。

## 4. 准备 nuScenes 数据

从 [nuScenes 下载页](https://www.nuscenes.org/download) 获取 nuScenes v1.0 full 数据、地图扩展及 CAN bus expansion，并按[官方数据准备说明](https://github.com/hustvl/MapTR/blob/maptrv2/docs/prepare_dataset.md) 准备 MapTRv2 专用地图 info 文件。TurboPhysAI 不负责下载或转换数据集。

将解压后的数据放入模型的 `data/` 目录，也可以链接已经准备好的数据目录。预期目录结构为：

```text
/workspace/model/MapTrv2/data/
├── can_bus/
└── nuscenes/
    ├── maps/
    │   └── expansion/
    ├── samples/
    ├── sweeps/
    ├── v1.0-trainval/
    ├── v1.0-test/
    ├── nuscenes_map_infos_temporal_train.pkl
    └── nuscenes_map_infos_temporal_val.pkl
```

已有 MapTRv2 专用 info 文件时可以直接使用。尚未生成时，在模型依赖已安装的环境中执行：

```bash
cd /workspace/model/MapTrv2
python tools/maptrv2/custom_nusc_map_converter.py \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0 \
  --canbus ./data

ls -lh data/nuscenes/nuscenes_map_infos_temporal_{train,val}.pkl
```

基线转换脚本会依次处理 trainval 和 test，因此上述流程需要准备 `v1.0-test`，并允许写入输出目录。`--canbus ./data` 指向包含 `can_bus/` 的父目录。

训练配置读取的是 `nuscenes_map_infos_temporal_train.pkl` 和 `nuscenes_map_infos_temporal_val.pkl`，不能用普通的 `nuscenes_infos_temporal_*.pkl` 替代。验证配置还使用 `nuscenes_map_anns_val.json`；该基线的数据集评估代码可在首次评估时生成此文件，因此评估时相应目录也需要可写。

## 5. 准备预训练权重

R50 配置从 `ckpts/resnet50-19c8e357.pth` 加载图像 Backbone 预训练权重：

```bash
cd /workspace/model/MapTrv2
mkdir -p ckpts
wget -O ckpts/resnet50-19c8e357.pth \
  https://download.pytorch.org/models/resnet50-19c8e357.pth

ls -lh ckpts/resnet50-19c8e357.pth
```

## 6. 准备优化配置

TurboPhysAI 已安装在镜像的 Python 环境中。源码安装和环境检查方法见[安装指南](../../docs/zh/get_started/installation.md)。后续命令使用当前 Python 环境，并从实际安装位置获取 MapTRv2 配置目录，无需固定组件安装路径：

```bash
export MAPTRV2_CONFIG_DIR="$(python -c 'from pathlib import Path; import turbo_physai.optimizations.models.maptrv2 as model; print(Path(model.__file__).resolve().parent / "configs")')"
```

当前 recipe 开启 `maptrv2.fused_adamw`，需要带编译扩展 `amp_C` 的 Apex。实际构建 AdamW 优化器时会检查 Apex 和设备可用性。优化实现及专项测试说明见[MapTRv2 优化说明](../../turbo_physai/optimizations/models/maptrv2/README.md)。

## 7. 生成并核对 OptimizationConfig

在目标训练环境中，针对干净模型基线重新生成源码和 AST 证据：

```bash
cd /workspace/model/MapTrv2
mkdir -p work_dirs/turbophysai

python -m turbo_physai.cli optimization generate \
  --recipe "$MAPTRV2_CONFIG_DIR/recipe.yaml" \
  --repo /workspace/model/MapTrv2 \
  --commit e03f097abef19e1ba3fed5f471a8d80fbfa0a064 \
  --output work_dirs/turbophysai/optimization.yaml \
  --force

python -m turbo_physai.cli optimization check \
  work_dirs/turbophysai/optimization.yaml \
  --repo /workspace/model/MapTrv2
```

`--force` 仅允许覆盖已存在的输出文件，不绕过 commit、干净工作区或目标符号检查。生成的 trust 证据与模型和实际导入的 PyTorch/MMCV 等目标实现绑定；更换环境或目标源码后需要重新生成、检查。

当前 recipe 包含 16 个 MapTRv2 Group，并继承公共 `mmcv.msda`，覆盖训练布局、DataLoader、DataContainer pin、Apex FusedAdamW、BEV pooling 布局、掩码和数据向量化、匹配静态化、编译边界及 DDP static graph。实现细节见[MapTRv2 优化说明](../../turbo_physai/optimizations/models/maptrv2/README.md)。OptimizationConfig 和 RuntimeConfig 的字段说明分别见[优化配置指南](../../docs/zh/user_guide/optimization_config.md)和[运行配置指南](../../docs/zh/user_guide/runtime_config.md)。

## 8. 单机八卡训练

在模型根目录启动训练，使配置中的相对数据、权重和输出路径正确解析：

```bash
cd /workspace/model/MapTrv2

turbo-physai run \
  --optimization-config work_dirs/turbophysai/optimization.yaml \
  --runtime-config "$MAPTRV2_CONFIG_DIR/runtime.yaml" \
  --log-report \
  torchrun --nproc-per-node=8 --master-port=29501 tools/train.py \
    projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
    --launcher pytorch \
    --work-dir work_dirs/maptrv2_optimized_8card
```

`--nproc-per-node=8` 创建八个本地训练进程，应与可用设备数一致；`29501` 应为可用端口。基线配置为每卡 4 个样本、4 个 DataLoader worker，使用 FP16 训练。

RuntimeConfig 提供 MIOpen、rocBLAS、RCCL、Inductor 和 NUMA 设置。其网络与拓扑参数来自参考八卡环境，不应直接当作其他机器的通用最优值。命令行 `--set NAME=VALUE` 可以覆盖环境值，`--disable-numa` 可以关闭自动 NUMA 绑定。

需要指定通信网卡时，在 `torchrun` 前增加 `--set NCCL_SOCKET_IFNAME=<interface>`，将 `<interface>` 替换为实际网卡名。

需要临时关闭 Apex 优化时，在 `torchrun` 前增加 `--disable-group maptrv2.fused_adamw`；关闭 DataContainer pin 时增加 `--disable-group maptrv2.datacontainer_pin`。后者不会同时关闭 `maptrv2.data` 对 DataLoader `pin_memory` 的设置。永久修改 Group 开关时更新 recipe，并重新生成 OptimizationConfig。

## 9. 优化状态确认

启用 `--log-report` 后，Rank 0 在优化安装完成时输出报告：

```text
TURBO_PHYSAI_OPTIMIZATION_REPORT_BEGIN run_id=<run-id>
...
Summary: applied=17 skipped=0 blocked=0 failed=0 rolled_back=0 not_started=0
...
TURBO_PHYSAI_OPTIMIZATION_REPORT_END run_id=<run-id>
```

上述计数对应当前全部启用的 17 个 Group；临时禁用或修改 recipe 后计数会变化。报告发生在训练迭代之前，`applied` 表示替换或包装已安装，不表示所有优化路径已经执行，也不证明模型精度和性能。

确认预期 Group 为 `applied`，并检查 `blocked`、`failed`、`rolled_back`、`not_started` 及每项原因，再结合训练 loss、验证指标和稳定迭代吞吐判断实际效果。报告字段和排查方法见[优化应用报告](../../docs/zh/user_guide/report.md)及[问题排查](../../docs/zh/user_guide/troubleshooting.md)。
