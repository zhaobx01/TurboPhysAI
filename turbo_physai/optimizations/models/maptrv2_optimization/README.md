# maptrv2 TurboPhysAI 优化开发工程

该目录由 `turbo-physai optimization init` 生成，现已按
`optimizations/models/bevformer`、`optimizations/models/bevfusion` 的组织方式
完成 MapTRv2 专用优化的接入。

参考基线：官方 `hustvl/MapTR` 仓库的 **`maptrv2` 分支**（HEAD `e03f097`）。
所有 `target` 路径都按该分支的符号书写，可直接套在干净基线上。

参考优化实现：**`autonomous-driving-models/hygon-hub/models/MapTRv2`**（权威，源码里 19 处
`@torch.compile()` 挂点全部启用）。

## 目录结构

```
maptrv2_optimization/
├── configs/recipe.yaml          # Group 开关，交给 generate 生成最终 YAML
├── configs/runtime.yaml         # MapTRv2 独立 RuntimeConfig（MIOpen / RCCL / Inductor / NUMA）
├── catalog.py                   # Group 声明（唯一注册入口）
│   ├── compat.py                # torch.compile / dynamo 能力探测
│   ├── compile.py               # torch.compile / dynamo.disable 包装器
│   ├── data.py                  # build_dataloader（pin_memory=True）
│   ├── grid_mask.py             # Dynamo 安全的 GridMask.forward
│   ├── match_cost.py            # cdist(p=1) -> 广播减法
│   ├── pv_mask.py               # PV/BEV 掩码绘制路径的向量化与变换合并
│   ├── replacements.py          # 替换实现索引（仅文档字符串）
│   └── training.py              # channels-last / cuDNN / fork
└── configs/__init__.py
```

测试位于仓库级 `TurboPhysAI/test/optimizations/`，文件名为
`test_maptrv2_catalog.py` 和 `test_maptrv2_implementations.py`。

`configs/optimization.yaml` 不在仓库里：它必须由 `optimization generate` 在干净
基线 worktree 上产出（`trust` 是 target 的源码/AST 哈希，手写必然过期）。

## Group 一览

| Group ID | 作用 | 默认 | 关键环境变量 |
| --- | --- | --- | --- |
| `maptrv2.training` | channels-last（`model` 或 `backbone`）、输入 NHWC 前处理、`cudnn.benchmark=True` / `deterministic=False`、`set_float32_matmul_precision("high")`、`fork` 启动 | 开 | `TURBO_PHYSAI_CHANNELS_LAST`、`TURBO_PHYSAI_CHANNELS_LAST_SCOPE`、`TURBO_PHYSAI_MATMUL_PRECISION`、`TURBO_PHYSAI_CUDNN_BENCHMARK`、`TURBO_PHYSAI_FORK_START_METHOD`、`TURBO_PHYSAI_DATALOADER_START_METHOD` |
| `maptrv2.data` | `build_dataloader` 使用 `pin_memory=True`，Host→Device 拷贝可与计算重叠 | 开 | `TURBO_PHYSAI_PIN_MEMORY=0` 关闭 |
| `maptrv2.grid_mask` | 保留官方掩码构造，只补 `np.copy()`（PIL 缓冲区只读）并把 `.cuda()` 换成 `x.device`；`forward` 常驻 `torch._dynamo.disable` | 开 | — |
| `maptrv2.match_cost` | `OrderedPtsL1Cost` 用广播减法替换 `torch.cdist(p=1)`，避开 ROCm 覆盖不足的 kernel | 开 | — |
| `maptrv2.pv_mask` | 辅助 PV 分割 GT 的 `line_ego_to_pvmask` 改为 numpy 弧长重采样，省掉每条线每路相机 200 次 shapely 调用；BEV 语义掩码的 `line_ego_to_mask` 把 `scale` + 平移两次 `shapely.affinity` 调用合并成一次 `scale_translate_geom`，并把 `np.array(list(coords))` 换成 `np.asarray`（逐像素一致） | 开 | `TURBO_PHYSAI_DISABLE_PV_MASK=1` 关闭 |
| `maptrv2.efficientnet` | 允许 `EfficientNet` 覆盖 registry 中同名条目 | 开 | — |
| `maptrv2.compile` | 给基线上携带编译挂点的 10 个热方法套 `torch.compile(mode="max-autotune-no-cudagraphs")`：`MapTRPerceptionTransformer.format_feats`、`MapTRDecoder.forward`、`MapTRv2.extract_img_feat`、`LSSTransform.get_cam_feats`/`get_mlp_input`、`maptrv2_head` 的 `normalize_3d_pts`、`normalize_2d_bbox`、`normalize_2d_pts`、`denormalize_2d_bbox`、`denormalize_2d_pts`；由 `compat.torch_compile_available` 逐次调用判定 | 关 | `TURBO_PHYSAI_DISABLE_TORCH_COMPILE=1` 关闭 |
| `maptrv2.assigner` | 用 `torch._dynamo.disable` 把 SciPy Hungarian 求解隔离在图外，由 `compat.dynamo_available` 逐次调用判定 | 关 | — |
`maptrv2.compile`、`maptrv2.assigner` 默认关闭，需要先在目标机型上完成精度/吞吐 A/B
再打开。

## 实现约定

* `catalog.py` 顶层只做声明：`import torch`、`import mmcv` 一律
  放进函数体，否则 `optimization check` / `plan` 这类只读阶段就会拉起重型依赖。
* `wrap` 的包装器在 `prepare()` 阶段就被调用，所以「依赖是否可用」的判断必须写在
  包装器内部（见 `compile.py`），`runtime_condition` 只负责逐次调用时的开关。
* 包装器签名固定为 `(original, options)`，返回值必须是 callable；`runtime_condition`
  必须是非类 callable，按被包裹调用的 `(*args, **kwargs)` 调用并返回 `bool`，
  返回 `False` 时引擎回退到原函数。

## 使用流程

1. 本地校验：

```bash
pytest test/optimizations/test_maptrv2_catalog.py test/optimizations/test_maptrv2_implementations.py
```

2. 在干净的 `maptrv2` 基线（HEAD `e03f097`）上生成带目标证据的最终 YAML。
   不要手写该文件：`generate` 会校验仓库状态与 commit、解析每个 `target` 并采集
   source/AST 证据，「target 不存在」在这一步就会暴露。输出文件已存在时需加
   `--force` 才会覆盖：

```bash
turbo-physai optimization generate \
  --recipe configs/recipe.yaml \
  --repo /path/to/MapTR \
  --commit e03f097 \
  --output configs/optimization.yaml
```

3. 之后可用生成的 YAML 复核模型工作区是否仍然匹配（相关命令还有 `validate`、
   `show`、`diff`）：

```bash
turbo-physai optimization check configs/optimization.yaml --repo /path/to/MapTR
```

4. 用生成的 YAML 启动原训练命令。`configs/runtime.yaml` 是 MapTRv2 自包含的运行配置，
   提供参考启动脚本使用的 MIOpen、rocBLAS、RCCL、HSA、Inductor 与 NUMA 默认值，不依赖
   其他模型的 runtime 配置。这些值是参考环境下的调优默认值，不代表所有部署都必须设置；
   RCCL 拓扑相关值应按实际网络和 GPU 拓扑覆盖：

```bash
turbo-physai run \
  --optimization-config configs/optimization.yaml \
  --runtime-config configs/runtime.yaml \
  python tools/train.py <原训练参数>
```

## 与参考实现的差异说明

* **assigner**：参考实现把 GT 预处理成 `(padded, valid_mask, count)` 元组并改写
  `assign` 签名，这需要同步修改 `MapTRv2Head._get_target_single`，属于跨文件的
  调用约定变更，无法用单个原子替换表达。本包只保留可安全自动化的部分——把
  SciPy 求解隔离出 Dynamo 图，语义与官方一致。
* **grid_mask**：与参考实现保持一致——`np.asarray(PIL image)` 返回只读缓冲区，
  `torch.from_numpy` 共享该缓冲区，于是掩码张量永远不可写，Dynamo/AOTAutograd
  对 storage 的守卫会因此失效；补 `np.copy()` 拿到可写张量即可。同时把官方硬编码
  的 `.cuda()` 换成 `x.device`（DCU/HIP 构建下该名称有误导性，且输入已在设备上时
  会多一次拷贝），并沿用参考实现在 `forward` 上的 `torch._dynamo.disable`。
  曾评估把 `rotate==1` 分支改写成纯张量实现以让它留在编译图内，但官方
  `for index in range(expanded_h // distance)` 不覆盖尾部残块
  （例如 `expanded_h=48, distance=7` 只铺到 42），改写无法保证逐元素一致，故放弃。
* **pv_mask**：参考实现给 `VectorizedLocalMap` 加了 `sample_line` /
  `project_points` 两个方法并改写 `line_ego_to_pvmask`，另外把 BEV 语义掩码的
  `line_ego_to_mask` 收敛成一次几何变换。本包把这些算法放进 `pv_mask.py`，用模块级
  函数整体替换这两个方法（不往模型类里塞新方法）：
  - `line_ego_to_pvmask`：弧长等距重采样 + `z > 0` 过滤 + `cv2.polylines`。基线
    `perspective()` 本来就带 `z > 0` 过滤，掩码逐像素一致（测试断言 `array_equal`）。
  - `line_ego_to_mask`：`affinity.scale(..., origin=(0, 0))` 与随后的平移合成一个
    矩阵（`scale_translate_geom`），并把 `np.array(list(line_ego.coords))` 换成
    `np.asarray(line_ego.coords)`。合成后的 xoff/yoff 与两步调用完全相同（推导见
    `note.md` §4.11），随机线段对拍 300/300 掩码逐像素相同。
* **compile**：权威参考实现（`autonomous-driving-models/hygon-hub/models/MapTRv2`）
  在源码里留下 19 处**已启用**的 `@torch.compile()` 挂点。其中 10 处落在基线同名符号
  上（即 `_COMPILE_TARGETS` 列出的那些）；8 处挂在参考实现新抽出的 helper 上
  （`BaseTransform.matmul_1/2/3`、`extract_metas`、`LSSTransform.down_sample`、
  `initialize_queries_and_bev`、`compute_decoder_predictions`、
  `prepare_transformer_inputs`），基线没有对应符号；第 19 处是 `MapTRAssigner.assign`，
  由 `maptrv2.assigner` 单独处理。本包不改源码，只覆盖基线已有的 10 个热方法，默认
  `mode` 为 `max-autotune-no-cudagraphs`（参考实现是无参 `@torch.compile()`，可用
  `options.mode` 覆盖）。基线中这些 helper 的逻辑内联在
  `MapTRPerceptionTransformer.forward`、`BaseTransform.get_geometry`/`forward`/
  `bev_pool`、`LSSTransform.forward`、`MapTRv2Head.forward`，方法体大且含控制流，
  本包有意不包裹，这几处仍走 eager。**注意**：参考实现的 `LSSTransform.down_sample`
  挂点在基线里没有对应物——其逻辑分散在 `BaseTransform.bev_pool`（collapse Z）与
  `LSSTransform.forward`（`self.downsample`），编译 `get_cam_feats` **覆盖不到**；
  如需覆盖，把对应方法补进 `_COMPILE_TARGETS` 后重新 `generate`，并在 A/B 时先看
  graph break 与重编译次数，再比较吞吐。

## 不在本包范围内的参考实现改动

* **`setup.py` 的 `if 1:` 强制构建**：权威参考实现只把编译条件从
  `torch.cuda.is_available() and CUDA_HOME is not None` 改成恒真，以便在无可见
  GPU 的构建环境中选择 `CUDAExtension`；它没有新增 HIP 源文件，源目录仍只有
  `.cu` 文件，setup 也只 glob `*.cu`。这属于扩展构建层，不是运行时替换，因此本包
  未声明对应 Group。
* **assigner 的静态打包改造**：见上文「与参考实现的差异说明」。
* **`transform_3d.py` 的 `TransposeImage`**：参考实现在该文件里新增（且重复定义了
  两次）一个把图像转成 channels-last 的 pipeline，但基线与本参考实现的**任何
  config 都没有引用它**（`grep TransposeImage` 只命中定义处），属于未接线的实验
  代码，故本包不声明对应 Group。channels-last 由 `maptrv2.training` 在模型侧生效。
* **`gen_vectorized_samples` 的 `LineString(np.array(instance))`**：权威参考实现去掉了
  这次多余的数组拷贝（`LineString` 本来就接受顶点序列），它与已覆盖的 `line_ego_to_mask`
  属于同一批 BEV 掩码路径改动。该方法约 90 行且控制流密集（`patch_box` 裁剪、多图层遍历、
  可选的 shift/rotate 增强），为省一次拷贝而整体替换不划算，故本包只覆盖它的两个下游绘制
  方法（见 `maptrv2.pv_mask`），此项不单独声明 Group。
* **研究性质的 config 调参**：权威参考实现的 `maptrv2_nusc_r50_24ep.py` 相对基线只有四处：
  `samples_per_gpu` 4→12、`workers_per_gpu` 4→48、`total_epochs` 24→1、
  `log_config.interval` 50→1（`optimizer` 段没有任何改动）。这些属于用户 config 与调试
  开关（见 `note.md` §3.1 的边界），不进本包。

## 与 `note.md` §2.2 草案的差异

工作区根目录 `note.md` 的 §2.2 是动手前的草案，其中的 `target` 大多指向**参考实现
新增**的符号，在官方 `maptrv2` 基线上并不存在，照抄会在 `optimization check` 阶段
报「target 不存在」。本包按基线符号重写，对应关系如下：

| 草案写法 | 基线实际情况 | 本包做法 |
| --- | --- | --- |
| `tools.train.main`、`tools.train.__main__` | `tools/train.py` 是脚本，没有可替换的 `main` | 合并进 `maptrv2.training`，挂在 `custom_train_detector` 上 |
| `LSSTransform.matmul_1/2/3`、`extract_metas`、`down_sample`、`initialize_queries_and_bev`、`compute_decoder_predictions` | 参考实现从大方法里抽出的新函数，基线里没有 | 改为包装基线已有的 10 个热方法（见 `maptrv2.compile` 一行）；`down_sample` 随 `get_cam_feats` 入图，其余新 helper 仍 eager |
| `MapTRv2Head._get_target_single` + `assign()` 新签名 | 会改变 GT 调用约定，必须跨文件同步改 | 只保留 dynamo 隔离，见上文 assigner 说明 |
| `EfficientNet` 的 `wrap` + `force_register` 替换 | 覆盖注册属于 registry 层能力 | 改用 `turbo_physai.compatibility.registry_override` |

## 静态核对现状

验证状态分两半：删除非权威扩展前，单元测试已实机跑通；
`optimization generate/check/plan` 仍未执行。

* **删除前记录**：`maptrv2_optimization/` 下的 `.py` 通过 `python -m py_compile`；
  `pytest test/optimizations/test_maptrv2_catalog.py test/optimizations/test_maptrv2_implementations.py`
  在 Python 3.11 + CPU 版 torch 2.14 + numpy/shapely/opencv/pillow 下得到
  **30 passed、53 subtests passed、0 skipped**（`mmcv` 用只提供 `runner.auto_fp16` 的
  最小桩替代，所以 GridMask 的 `auto_fp16` 组合与 HCU 路径仍待上机）。覆盖点：Group /
  机制 / condition 解析、catalog 导入不拉起 torch、channels-last 作用域、matmul 精度
  默认值、GridMask 掩码与 offset 路径对官方 numpy 构造、L1 cost 对 `torch.cdist`
  （float64 严格 + float32 容差）、`pv_mask` 重采样对 shapely 插值、投影对官方
  `perspective`、矢量化绘制与官方逐点 loop **逐像素一致**、`line_ego_to_mask` 对官方
  `scale` + 平移两步变换**逐像素一致**、`scale_translate_geom` 对两步 `affinity`
  在非零 `origin` 下坐标严格相等。
* **待复核**：删除非权威扩展后重新运行上述定向测试。
* **未执行**：`optimization generate/check/plan`、训练级精度/吞吐 A/B（需模型依赖与 DCU）。

其余证据来自符号定位与人工 diff：

* 全量对照以 **`git diff --no-index` 基线 vs 权威优化版**（忽略 `.git` 与行尾空白）为准：
  `projects/` 下有 26 个文件存在差异（约 15 个含性能/硬件适配改动）；`tools/` 下有
  4 个：`train.py` 有实质改动、新增 `dist_train_numa.sh`，另 2 个仅行尾空白。
* 每个 `target` 都在基线 worktree（HEAD `e03f097`）定位到定义行，例如
  `maptr/modules/transformer.py:288`、
  `maptr/modules/decoder.py:23`、`maptr/detectors/maptrv2.py:75`、
  `maptr/modules/encoder.py:1095`/`:1175`、`maptr/dense_heads/maptrv2_head.py:25`/
  `:37`/`:49`/`:59`/`:68`，以及 `map_loss.py:510`（class 在 501）、
  `nuscenes_offlinemap_dataset.py:669`、`grid_mask.py:85`、`mmdet_train.py:28`、
  `datasets/builder.py:19`、`efficientnet.py:157`。
* 上机后先跑「使用流程」的 1-3 步，再打开 `maptrv2.compile` / `maptrv2.assigner`
  做精度与吞吐 A/B。

## 注意事项

* 文档与代码中不得出现个人绝对路径、内网地址、凭据、令牌或未脱敏日志。
* 测试位于仓库级 `test/optimizations/`，使用 `pytest` 收集；需要 HCU/DCU 设备的验证
  用 `@pytest.mark.hcu` 标记。
