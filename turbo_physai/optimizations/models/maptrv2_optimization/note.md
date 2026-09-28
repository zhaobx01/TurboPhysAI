# MapTRv2 优化项分析与 TurboPhysAI 接入方案

> 基线：`<workspace>/MapTR_v2`（官方 `hustvl/MapTR` 的 `maptrv2` 分支，HEAD `e03f097`）
> 优化侧（权威）：`/workspace/MapTRv2`；未优化基线：`/workspace/model/MapTrv2`；本文以这两份本地 checkout 为准，三方差分见 §3.8。
> 骨架：`<workspace>/TurboPhysAI/turbo_physai/optimizations/models/maptrv2_optimization`。

对比方法：`git diff --no-index` 两侧全量文件并忽略行尾空白。主体统计暂时排除
`mmdetection3d/` 子树与 `.git`：权威优化侧在 `projects/` 下有 26 个文件存在差异（其中约
15 个含性能/硬件适配改动）、`tools/` 下有 4 个（`tools/train.py` 有实质改动、新增
`tools/dist_train_numa.sh`，另 2 个仅行尾空白），其余为行尾空白与注释清理。`mmdetection3d`
子树、镜像/启动脚本以及完整覆盖状态见 §3.6。

---

## 1. 优化版相对基线的改动

分两类：
- **A. 训练配置调参 / 调试便利**（不进 TurboPhysAI，属于 config 与临时开关）。
- **B. 性能 / 硬件适配 / 编译器兼容**（应抠进 TurboPhysAI 优化包）。

### A. 训练配置与调试

#### A1. Batch 与训练调度调整

```diff
 data = dict(
-    samples_per_gpu=4,
-    workers_per_gpu=4, # TODO
+    samples_per_gpu=12,
+    workers_per_gpu=48, # TODO
 optimizer = dict(
     type='AdamW',
-total_epochs = 24
+total_epochs = 1   #24
 log_config = dict(
-    interval=50,
+    interval=1,
```

**优点**：`samples_per_gpu` 4→12、`workers_per_gpu` 4→48 打满 DCU 显存与主机 IO；`total_epochs=1` + `log interval=1` 便于快速 profiling。**属于用户 config，不进优化包**。

#### A2. `MapTRv2Head` 解码预测拆分

主要是把原本 `forward` 里内联的 decoder 后处理循环抽成独立方法：

```diff
-    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
-    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
+    #@torch.compile(mode="max-autotune-no-cudagraphs")
+    def compute_decoder_predictions(self, outputs, bs, num_vec, mlvl_feats):
+        bev_embed, depth, hs, init_reference, inter_references = outputs
+        ...
```

**优点**：为 `torch.compile` 提供更小的图边界（见 B5）。

---

### B. 性能、硬件与编译适配

#### B1. channels-last 内存布局

`tools/train.py`

```diff
     model = build_model(cfg.model, train_cfg=..., test_cfg=...)
+    model.img_backbone = model.img_backbone.to(memory_format=torch.channels_last)
     model.init_weights()
```

`projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py`
```diff
-    model = MMDistributedDataParallel(model.cuda(), device_ids=[...])
+    model = MMDistributedDataParallel(
+        model.cuda().to(memory_format=torch.channels_last),
+        device_ids=[...])
```

`projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py` 新增 `TransposeImage` pipeline，把输入 tensor 也转成 channels_last：

```python
@PIPELINES.register_module()
class TransposeImage:
    def __call__(self, results):
        def convert(x):
            if isinstance(x, torch.Tensor):
                return x.contiguous(memory_format=torch.channels_last)
            return x
        ...
```

**优点**：Conv2d 在 NHWC 布局下命中 cuDNN / MIOpen 的 Tensor Core / DCU 加速路径；输入 tensor 提前转 NHWC 避免 backbone 内部反复 permute；对精度无影响。



#### B2. cuDNN 配置与 fork 启动

`tools/train.py`
```diff
     model.CLASSES = datasets[0].CLASSES
+    torch.backends.cudnn.benchmark = True        # 启用自动寻找最优卷积算法
+    torch.backends.cudnn.deterministic = False   # 允许非确定性算法提升速度
     custom_train_model(...)

 if __name__ == '__main__':
+    torch.multiprocessing.set_start_method('fork')
     main()
```

**优点**：`benchmark=True` 让 cuDNN/MIOpen 挑最快 conv 算法（固定 shape 提速明显）；`deterministic=False` 解锁更快算法；`fork` 让 DataLoader worker 复用父进程已导入的 mmcv/mmdet3d，冷启动更快。

**已知限制**：`set_start_method('fork')` 只在单卡直接跑 `python tools/train.py` 时有效；`torchrun` 多卡场景下这行代码不生效，`DataLoader` 会退回默认的 `spawn`，进而在 pickle dataset 时报错（原因和修法见 U5）。TurboPhysAI 的做法是不依赖这个全局调用，而是在 `data.py` 的 `build_dataloader()` 里给每个 `DataLoader` 单独传 `multiprocessing_context`，创建时直接生效，不受 `torchrun` 影响。

#### B3. DataLoader `pin_memory=True`

`projects/mmdet3d_plugin/datasets/builder.py`
```diff
     data_loader = DataLoader(
         ...
-        pin_memory=False,
+        pin_memory=True,
         worker_init_fn=init_fn,
         **kwargs)
```

**优点**：Host→Device 拷贝走 pinned memory，可与计算 overlap，降低 IO 等待。

#### B4. GridMask Dynamo 兼容

`projects/mmdet3d_plugin/models/utils/grid_mask.py`
```diff
     @auto_fp16()
+    @torch._dynamo.disable
     def forward(self, x):
         ...
-        mask = torch.from_numpy(mask).to(x.dtype).cuda()
+        mask = torch.from_numpy(mask.copy()).to(x.dtype).cuda()
```

**优点**：
- `@torch._dynamo.disable`：GridMask 含 `np.random`、PIL 旋转等 Python 分支，dynamo 无法安全捕获；显式跳过避免整图 fallback。
- `mask.copy()`：PIL 转出的 numpy 数组是 non-writeable，`torch.from_numpy` 会警告并在 AOT 阶段触发 guard 失败，`.copy()` 后可写。

#### B5. `torch.compile` 计算边界拆分

优化后把内联热点抽成独立方法，并在每个方法上挂 `@torch.compile`。权威优化版中这 19 处挂点
**全部处于启用状态**，其中 18 处显式写成 `@torch.compile(mode="max-autotune-no-cudagraphs")`
（只有 `MapTRAssigner.assign` 用 `options={"triton.cudagraphs": True, ...}`）。
TurboPhysAI 的 recipe 默认 mode 同样是 `max-autotune-no-cudagraphs`，两者一致。
下面只示意挂点位置：`projects/mmdet3d_plugin/maptr/modules/transformer.py`

```diff
+    @torch.compile()
+    def initialize_queries_and_bev(self, object_query_embed, bev_embed, bs, bev_h, bev_w):
+        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
+        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
+        query = query.unsqueeze(0).expand(bs, -1, -1)
+        reference_points = self.reference_points(query_pos).sigmoid()
+        init_reference_out = reference_points
+        query = query.permute(1, 0, 2); query_pos = query_pos.permute(1, 0, 2)
+        bev_embed = bev_embed.permute(1, 0, 2)
+        spatial_shapes = torch.tensor([[bev_h, bev_w]], device=query.device)
+        level_start_index = torch.tensor([0], device=query.device)
+        return query, bev_embed, query_pos, reference_points, spatial_shapes, level_start_index, init_reference_out
...
-    query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
-    ...
-    bev_embed = bev_embed.permute(1, 0, 2)
+    query, bev_embed, query_pos, reference_points, spatial_shapes, level_start_index, init_reference_out \
+        = self.initialize_queries_and_bev(object_query_embed, bev_embed, bs, bev_h, bev_w)
```

`projects/mmdet3d_plugin/maptr/modules/encoder.py`（新增拆分 + 挂点）
```python
#@torch.compile()
def matmul_1(self, x, y, trans): ...
#@torch.compile()
def matmul_2(self, x, y, trans, lidar2ego_trans, points): ...
#@torch.compile()
def matmul_3(self, x, y, trans): ...
#@torch.compile()
def extract_metas(self, images, img_metas): ...
#@torch.compile()
def down_sample(self, x): ...
```

`projects/mmdet3d_plugin/maptr/modules/decoder.py`
```diff
+    @torch.compile()
     def forward(self, query, *args, ...):
```

`projects/mmdet3d_plugin/maptr/detectors/maptrv2.py`
```diff
-
+    @torch.compile()
     def extract_img_feat(self, img, img_metas, len_queue=None):
```

**优点**：`torch.compile` 对"纯张量、无副作用"的小函数图捕获成功率最高；把 host-side组装（`torch.tensor(...)`、`.permute`）与算子调用分离后，热点段可以单独编译，也便于A/B 开关。

#### B6. 参考实现新增 Helper 编译点的等价接入

参考实现把 5 个大方法内联的逻辑拆成了 8 个纯张量 helper（`matmul_1/2/3`、`extract_metas`、`down_sample`、`initialize_queries_and_bev`、`compute_decoder_predictions`、`prepare_transformer_inputs`），每个单独挂 `@torch.compile()`。这 8 个名字是参考实现新增的，官方基线里不存在，`replace(target=...)` 找不到同名符号，没法直接声明成替换目标（原因见 §4.12）。

**接入方式**：`reference.py` 把 8 个 helper 留作模块级纯函数，转而重写调用它们的 **5 个基线原生方法**（`maptrv2.reference_boundaries` group 的 5 个 `wrap` 目标），让新实现在内部调用 helper——调用方的输入输出契约不变，只是内部实现换掉，不新增、也不改动模型源码里的任何符号。

**总体概述**：

| # | 替换目标（`wrap`，基线原生符号） | 内部调用的 helper | 保留 eager 的部分 |
| --- | --- | --- | --- |
| B6.1 | `BaseTransform.get_geometry_v1` | `matmul_1` / `matmul_2` / `matmul_3` | frustum 缓存判断、`extra_rots`/`extra_trans` 可选分支 |
| B6.2 | `BaseTransform.forward` | `extract_metas` | `get_geometry_v1`/`get_mlp_input`/`get_cam_feats` 调用、`bev_pool` 5D 布局校验 |
| B6.3 | `LSSTransform.forward` | `down_sample` | 无（整段计算搬进 helper） |
| B6.4 | `MapTRPerceptionTransformer.forward` | `initialize_queries_and_bev` | `get_bev_features`、`format_feats`、`decoder` 调用 |
| B6.5 | `MapTRv2Head.forward` | `prepare_transformer_inputs`、`compute_decoder_predictions` | `compute_aux_seg_outputs`（分割头，依赖 `self.aux_seg[...]` 配置分支，刻意不进图） |

**共用机制**：5 个 wrapper 在返回替换函数前都先调一次 `_prepare_helpers(options)`：

```python
_COMPILED = {}
_ACTIVE_MODE = None

def _prepare_helpers(options):
    global _ACTIVE_MODE
    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return
    mode = options.get("mode", "max-autotune-no-cudagraphs")
    _ACTIVE_MODE = mode
    for name, function in _HELPERS.items():   # 一次性编译全部 8 个，不只是当前 wrapper 用到的
        key = (name, mode)
        if key not in _COMPILED:
            _COMPILED[key] = torch.compile(function, mode=mode)
```

- **缓存全局共享**：8 个 helper 一次性编译进 `_COMPILED`，5 个 wrapper 里谁先跑到都会把全部 8 个编译好，其余几次调用直接命中缓存。`mode` 默认 `max-autotune-no-cudagraphs`，取自 group 的 `options`（当前是空字典），与 B5（`maptrv2.compile`）一致；5 个 wrap 共用同一份 `options`，没法单独改其中一个的 mode。
- **`TURBO_PHYSAI_DISABLE_TORCH_COMPILE=1` 是两层开关**：外层是 5 个 `wrap()` 共用的 `runtime_condition=compat.torch_compile_available`，引擎每次实际调用前都会查——变量一设，直接回退未改动的原始基线方法，`reference.py` 整段代码都不会跑；`_prepare_helpers` 内部对同一变量的判断是装载阶段的第二次检查，避免在明知用不上时还跑一遍 `torch.compile()`。
- **`force_fp32` 只套 3 处，是复刻基线**：`get_geometry_v1`、`BaseTransform.forward`（基线 `encoder.py:121`/`:293` 本来就是 `@force_fp32()`）、`MapTRv2Head.forward`（基线 `@force_fp32(apply_to=('mlvl_feats','prev_bev'))`，只强转这两个参数）。`LSSTransform.forward`、`MapTRPerceptionTransformer.forward` 基线里本来就没有这个装饰器（前者的精度敏感部分已被它内部调用、已强制 fp32 的 `BaseTransform.forward` 覆盖），所以不套。`wrap`/`replace` 会连带原方法上的装饰器一起换掉，这里是手动把基线原有的装饰器复原，不是新加的保护。

##### B6.1 `BaseTransform.get_geometry_v1` → `matmul_1` / `matmul_2` / `matmul_3`

**概述**：把"图像系逆投影 → 相机系转 ego 系 → 转 lidar 系"这三段矩阵乘法，从一个大函数体拆成三个独立的纯张量函数，函数整体的输入输出契约不变。

```
优化前                                    优化后
get_geometry_v1(...)                     get_geometry_v1(...)   ← 仍是同一个 wrap 目标
  ├─ inverse(post_rots) 变换 frustum        ├─ matmul_1(...)     可单独编译/单独重编译
  ├─ rots@inverse(intrins) 变换+平移        ├─ matmul_2(...)     可单独编译/单独重编译
  └─ inverse(lidar2ego_rots) 变换            └─ matmul_3(...)     可单独编译/单独重编译
  （三段写在同一个函数体里，只能整体 eager）     （三段各自是独立 FX 图，互不拖累）
```

本包实现（`reference.py`）：

```python
points = self.frustum - post_trans.view(batch, cameras, 1, 1, 1, 3)
points = _helper("matmul_1")(self, torch.inverse(post_rots), points, trans)
points = _helper("matmul_2")(self, rots, torch.inverse(intrins), trans, lidar2ego_trans, points)
points = _helper("matmul_3")(self, torch.inverse(lidar2ego_rots), points, trans)
```

**优点**：三段矩阵乘法本身无副作用、无 Python 分支，是 Dynamo 编译成功率最高的一类代码；拆开后 Inductor 能分别对每段做算子融合，某一段因 shape 变化触发重编译也不会拖累另外两段。`extra_rots`/`extra_trans` 两个可选分支（配置不启用时恒为假）留在外层 `get_geometry_v1` 里 eager 判断，不进 helper。

##### B6.2 `BaseTransform.forward` → `extract_metas`

**概述**：把 `img_metas`（Python list，逐相机一个 dict）里的 `camera2ego`/`camera_intrinsics`/`img_aug_matrix`/`lidar2ego` 抠出来、numpy→tensor、`stack` 成 batched 张量的过程独立成一个 helper。

```python
def _stack_metas(metas, key, device, dtype):
    tensors = []
    for meta in metas:
        value = meta[key]                     # 可能是 np.ndarray 或 list
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        elif isinstance(value, list):
            value = torch.stack([...])
        tensors.append(value)
    return torch.stack(tensors, dim=0).to(device=device, dtype=dtype)
```

```
BaseTransform.forward(images, img_metas)
  ├─ extract_metas(...)            ← host-side 元数据组装，独立成 helper
  ├─ self.get_geometry_v1(...)     ← B6.1
  ├─ self.get_mlp_input(...)
  ├─ self.get_cam_feats(...)
  └─ _bev_pool_5d(...)             ← 附带的布局防御，见下
```

**为什么单独拆出来**：这段是 Python `for` 循环加 numpy↔tensor 转换，numpy 交互必然触发 graph break，本身就不适合进编译图；拆成独立 helper 后，至少能单独决定它是否编译，不拖累调用它的 `forward` 主体。

**附带的 ROCm 防御**：这条路径内部的 `_bev_pool_5d` 对 `bev_pool` 输出做一次形状识别——NHWC `[B,Z,H,W,C]` 和 NCHW `[B,C,Z,H,W]` 两种都认，统一转成 NHWC 返回。这是**独立于** `maptrv2.bev_pool_fix`（另一条 ROCm layout 修正，见 §3.7.1/U3）的第二道防线，只作用于 `reference_boundaries` 这条调用路径，两者互不覆盖。

##### B6.3 `LSSTransform.forward` → `down_sample`

**概述**：参考实现的 `down_sample` 在基线里本来就没有对应的单一符号——collapse Z 轴的逻辑原本在 `BaseTransform.bev_pool` 尾部，`self.downsample` 调用原本在 `LSSTransform.forward` 里，是**两个文件两个方法**分开写的（详细的载体分析见 §4.12）。本包把两段合成一个 helper：

```python
def down_sample(self, x):
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    x = torch.cat(x.unbind(dim=2), 1)      # collapse Z
    x = x.permute(0, 1, 3, 2).contiguous()
    return self.downsample(x)
```

```
优化前（两个文件两个方法各管一段）              优化后（合成一个 helper，整段可编译）
BaseTransform.bev_pool:  collapse Z          LSSTransform.forward:
LSSTransform.forward:    self.downsample(x)    bev = down_sample(self, features)
```

**优点**：collapse-Z 的 `permute`+`cat`+`permute` 和随后的 `self.downsample`（Conv2d）现在是同一个可编译单元，不需要在两个函数之间来回切换 eager/graph 状态。

##### B6.4 `MapTRPerceptionTransformer.forward` → `initialize_queries_and_bev`

**概述**：把 query/query_pos 的 split+expand、`reference_points` 预测+sigmoid、query/BEV 的 permute，以及静态 `spatial_shapes`/`level_start_index` 构造这段初始化逻辑拆出来。

```python
def initialize_queries_and_bev(self, object_query_embed, bev_embed, batch_size, bev_h, bev_w):
    query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
    query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
    query = query.unsqueeze(0).expand(batch_size, -1, -1)
    reference_points = self.reference_points(query_pos).sigmoid()
    query = query.permute(1, 0, 2); query_pos = query_pos.permute(1, 0, 2)
    bev_embed = bev_embed.permute(1, 0, 2)
    spatial_shapes = torch.tensor([[bev_h, bev_w]], device=query.device)
    level_start_index = torch.tensor([0], device=query.device)
    return query, bev_embed, query_pos, reference_points, spatial_shapes, level_start_index, reference_points
```

```
transformer_forward(...)
  ├─ get_bev_features(...)               ← BEV 编码器，留 eager（体积大、含控制流）
  ├─ initialize_queries_and_bev(...)     ← 拆出的纯张量段，可编译
  ├─ self.format_feats(...)
  └─ self.decoder(...)                   ← decoder 主体，留 eager（含注意力）
```

**优点**：这段全是 tensor split/expand/permute，没有 Python 分支，独立编译能缩小 transformer 主 `forward` 的图边界；体积大、含控制流的 `get_bev_features` 和 `decoder` 不强行进图。

##### B6.5 `MapTRv2Head.forward` → `prepare_transformer_inputs` + `compute_decoder_predictions`

**概述**：head 的 `forward` 拆成两段可编译 helper：一段是"decoder 之前的输入准备"（`prepare_transformer_inputs`：按 `query_embed_type` 选路径、构造 BEV query/位置编码、构造 one2one/one2many 的 `self_attn_mask`），一段是"decoder 输出之后的逐层后处理"（`compute_decoder_predictions`：`inverse_sigmoid` + cls/reg branch + sigmoid + one2one/one2many 切分）。

```
head_forward(...)
  ├─ prepare_transformer_inputs(...)        ← 配置态组装，可编译
  ├─ self.transformer(...)                  ← 见 B6.4
  ├─ compute_decoder_predictions(...)       ← 逐层张量后处理，可编译
  └─ compute_aux_seg_outputs(...)           ← 分割头，刻意留 eager（见下）
```

`compute_aux_seg_outputs` **不进编译**：它依赖 `self.aux_seg["use_aux_seg"]`/`["bev_seg"]`/`["pv_seg"]` 三层配置分支，进图会导致 graph break；本包按参考实现同样的取舍，把它单独列成一个 eager-only 函数，不参与 `_prepare_helpers` 缓存。`head_forward_wrapper` 额外套 `force_fp32(apply_to=("mlvl_feats", "prev_bev"))`，只强转这两个参数。

**优点**：`for level in range(hidden_states.shape[0])` 这段逐层循环加 `list.append` 汇总现在是独立编译单元，不再和"要不要跑分割头"这类配置判断混在同一张图里。



#### B7. `cdist` 替换为广播减法

`projects/mmdet3d_plugin/maptr/losses/map_loss.py`
```diff
     bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)
     gt_bboxes = gt_bboxes.flatten(2).view(num_gts*num_orders, -1)
-    bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
+    bbox_cost = (bbox_pred[:, None, :] - gt_bboxes[None, :, :]).abs().sum(dim=-1)
     return bbox_cost * self.weight
```

**优点**：`torch.cdist` 在 ROCm/DCU 上算子覆盖差且反传路径不稳；改成显式广播减+求和后走通用 elementwise/reduce kernel，正反向都稳定，同时对 dynamo/compile 更友好。

#### B8. Assigner 静态化与 Hungarian 隔离

`projects/mmdet3d_plugin/maptr/assigners/maptr_assigner.py`

**原始代码（基线）**：

```python
def assign(self, bbox_pred, cls_pred, pts_pred, gt_bboxes,
           gt_labels, gt_pts, gt_bboxes_ignore=None, eps=1e-7):
    assert gt_bboxes_ignore is None
    assert bbox_pred.shape[-1] == 4
    
    ##-------------------------删除--------------------------------##
    num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

    assigned_gt_inds = bbox_pred.new_full(
        (num_bboxes,), -1, dtype=torch.long)
    assigned_labels = bbox_pred.new_full(
        (num_bboxes,), -1, dtype=torch.long)

    if num_gts == 0 or num_bboxes == 0:
        if num_gts == 0:
            assigned_gt_inds[:] = 0
        return AssignResult(
            num_gts, assigned_gt_inds, None,
            labels=assigned_labels), None

    cls_cost = self.cls_cost(cls_pred, gt_labels)
    normalized_gt_bboxes = normalize_2d_bbox(gt_bboxes, self.pc_range)
    reg_cost = self.reg_cost(
        bbox_pred[:, :4], normalized_gt_bboxes[:, :4])
    ##-------------------------删除--------------------------------##

    _, num_orders, num_pts_per_gtline, _ = gt_pts.shape
    normalized_gt_pts = (
        normalize_2d_pts(gt_pts, self.pc_range)
        if not self.z_cfg["gt_z_flag"]
        else normalize_3d_pts(gt_pts, self.pc_range)
    )
    if pts_pred.size(1) != num_pts_per_gtline:
        pts_pred_interpolated = F.interpolate(
            pts_pred.permute(0, 2, 1),
            size=(num_pts_per_gtline,),
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1).contiguous()
    else:
        pts_pred_interpolated = pts_pred

    pts_cost_ordered = self.pts_cost(
        pts_pred_interpolated, normalized_gt_pts)
    pts_cost_ordered = pts_cost_ordered.view(
        num_bboxes, num_gts, num_orders)
    pts_cost, order_index = torch.min(pts_cost_ordered, 2)

    bboxes = denormalize_2d_bbox(bbox_pred, self.pc_range)
    iou_cost = self.iou_cost(bboxes, gt_bboxes)
    cost = cls_cost + reg_cost + iou_cost + pts_cost
    ##-------------------------删除--------------------------------##
    cost = cost.detach().cpu()
    if linear_sum_assignment is None:
        raise ImportError("Please run pip install scipy first.")
    matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
    matched_row_inds = torch.from_numpy(matched_row_inds).to(
        bbox_pred.device)
    matched_col_inds = torch.from_numpy(matched_col_inds).to(
        bbox_pred.device)

    assigned_gt_inds[:] = 0
    assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
    assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
    
    return AssignResult(
        num_gts, assigned_gt_inds, None,
        labels=assigned_labels), order_index
    ##-------------------------删除--------------------------------##
```

**优化后代码**：

```python
##------------------------------新增-----------------------------------##
@torch._dynamo.disable
def hungarian_match(self, cost, gt_labels, assigned_gt_inds,
                    assigned_labels, num_gts, device):
    cost = cost.detach().cpu()
    if linear_sum_assignment is None:
        raise ImportError("Please run pip install scipy first.")

    matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
    matched_row_inds = torch.as_tensor(matched_row_inds, device=device)
    matched_col_inds = torch.as_tensor(matched_col_inds, device=device)

    assigned_gt_inds[:] = 0
    assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
    assigned_labels[matched_row_inds] = (
        gt_labels[0][matched_col_inds])

    return AssignResult(
        num_gts, assigned_gt_inds, None,
        labels=assigned_labels)
 ##------------------------------新增-----------------------------------##

@torch.compile(options={
    "triton.cudagraphs": True,
    "triton.cudagraph_trees": False,
})
def assign(self, bbox_pred, cls_pred, pts_pred, gt_bboxes,
           gt_labels, gt_pts, gt_bboxes_ignore=None, eps=1e-7):
    assert gt_bboxes_ignore is None
    assert bbox_pred.shape[-1] == 4

    ##------------------------------新增-----------------------------------##
    num_bboxes = bbox_pred.size(0)

    assigned_gt_inds = bbox_pred.new_full(
        (num_bboxes,), -1, dtype=torch.long)
    assigned_labels = bbox_pred.new_full(
        (num_bboxes,), -1, dtype=torch.long)

    cls_cost = self.cls_cost(cls_pred, gt_labels[0])
    normalized_gt_bboxes = normalize_2d_bbox(
        gt_bboxes[0], self.pc_range)
    ##------------------------------新增-----------------------------------##
    _, num_orders, num_pts_per_gtline, _ = gt_pts[0].shape
    normalized_gt_pts = (
        normalize_2d_pts(gt_pts[0], self.pc_range)
        if not self.z_cfg["gt_z_flag"]
        else normalize_3d_pts(gt_pts[0], self.pc_range)
    )
    if pts_pred.size(1) != num_pts_per_gtline:
        pts_pred_interpolated = F.interpolate(
            pts_pred.permute(0, 2, 1),
            size=(num_pts_per_gtline,),
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1).contiguous()
    else:
        pts_pred_interpolated = pts_pred

    bboxes = denormalize_2d_bbox(bbox_pred, self.pc_range)
    pts_cost_ordered = self.pts_cost(
        pts_pred_interpolated, normalized_gt_pts)
    pts_cost_ordered = pts_cost_ordered.view(
        num_bboxes, gt_bboxes[0].size(0), num_orders)
    pts_cost, order_index = torch.min(pts_cost_ordered, 2)

    reg_cost = self.reg_cost(
        bbox_pred[:, :4], normalized_gt_bboxes[:, :4])
    iou_cost = self.iou_cost(bboxes, gt_bboxes[0])
    cost = cls_cost + reg_cost + iou_cost + pts_cost
 ##------------------------------新增-----------------------------------##
    assign_result = self.hungarian_match(
        cost[:, gt_bboxes[1]],
        gt_labels,
        assigned_gt_inds,
        assigned_labels,
        gt_bboxes[2],
        bbox_pred.device,
    )
    return assign_result, order_index
 ##------------------------------新增-----------------------------------##
```

**关键优化差异（删减版）**：以下 `...` 表示未改动的中间代码：

```diff
-    num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
-    if num_gts == 0 or num_bboxes == 0:
-        ...
-        return AssignResult(
-            num_gts, assigned_gt_inds, None,
-            labels=assigned_labels), None
-    cls_cost = self.cls_cost(cls_pred, gt_labels)
-    normalized_gt_bboxes = normalize_2d_bbox(gt_bboxes, self.pc_range)
+    num_bboxes = bbox_pred.size(0)
+    # GT 已由外层 pad，并携带 valid_mask 和真实 num_gts
+    cls_cost = self.cls_cost(cls_pred, gt_labels[0])
+    normalized_gt_bboxes = normalize_2d_bbox(gt_bboxes[0], self.pc_range)
...
-    matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
-    ...
-    return AssignResult(
-        num_gts, assigned_gt_inds, None,
-        labels=assigned_labels), order_index
+    assign_result = self.hungarian_match(
+        cost[:, gt_bboxes[1]], gt_labels, assigned_gt_inds,
+        assigned_labels, gt_bboxes[2], bbox_pred.device)
+    return assign_result, order_index
+
+    @torch._dynamo.disable
+    def hungarian_match(self, cost, gt_labels, assigned_gt_inds, assigned_labels, num_gts, device):
+        cost = cost.detach().cpu()
+        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
+        matched_row_inds = torch.as_tensor(matched_row_inds, device=device)
+        matched_col_inds = torch.as_tensor(matched_col_inds, device=device)
+        assigned_gt_inds[:] = 0
+        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
+        assigned_labels[matched_row_inds] = gt_labels[0][matched_col_inds]
+        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
```

**关键调用约定变化**：`gt_bboxes` 从张量改为 `(padded_bboxes, valid_mask, num_gts)` 元组；`gt_labels`/`gt_pts` 也取 `[0]` 索引。相当于外层把每 batch 的 GT 预先 pad 到固定形状 + mask，让匹配路径的张量形状不再依赖当前 batch 的 GT 数量。

**优点**：

- 消除 `num_gts == 0 or num_bboxes == 0` 这类动态分支 + 早退，让 dynamo 能追踪。
- 将 CPU-only 的 SciPy `linear_sum_assignment` 单独打上 `@torch._dynamo.disable`，避免拖垮整图。
- GT 形状固定 → `torch.compile` guard 不会因每步 batch 内 GT 数量抖动而失效。

**实现修正（2026-09-23）**：启用 `maptrv2.assigner` 后曾在反向阶段触发 Inductor stride 断言，例如：

```text
AssertionError: expected size 256==256, stride 375==1 at dim=1
```

原因是 assignment/matching 本身不可导，但此前整个 `assign()` 默认进入`torch.compile(..., cudagraphs=True)`，在 channels-last 模型上与反向图/布局发生冲突。当前实现将其修正为：

- `assign()` 的匹配计算统一放入 `torch.no_grad()`，避免建立无意义的 backward 图；
- `torch.compile` 改为显式 opt-in，默认读取
  `TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE=0`，不再默认编译 Assigner；
- `configs/runtime.yaml` 同样显式写入 `0`，作为运行时双重保险；
- 现有 GT padding、valid mask、`_get_target_single()` 和 Hungarian 隔离逻辑不变。

服务器验证：仅上传 `assigner.py` 修改后，训练已可正常启动，说明问题由 Assigner默认编译路径引起，修复不依赖模型源码或其他优化 Group。已增加默认 eager 路径回归测试。

#### B9. EfficientNet 注册覆盖

`projects/mmdet3d_plugin/models/backbones/efficientnet.py`
```diff
-@BACKBONES.register_module()
+@BACKBONES.register_module(force=True)
 class EfficientNet(BaseModule):
```

**优点**：容器镜像里 mmcls / mmdet 可能已注册同名 backbone；`force=True` 保证 MapTR 版本覆盖，避免 registry 冲突报错。

#### B10. 强制选择 CUDAExtension 构建

权威版本**没有新增** `.hip` 文件，源文件仍是原有的
`geometric_kernel_attn_cuda.cu`。实际 diff 只有：

`.../setup.py`
```diff
-    if torch.cuda.is_available() and CUDA_HOME is not None:
+    if 1:
+    # if torch.cuda.is_available() and CUDA_HOME is not None:
         extension = CUDAExtension
         sources += source_cuda
         define_macros += [("WITH_CUDA", None)]
```

**作用**：构建镜像时通常看不到 GPU，`torch.cuda.is_available()` 可能为 `False`，原逻辑会直接
进入 `else` 并报 `NotImplementedError`。改成 `if 1:` 后，setup 始终选择 `CUDAExtension` 并
把已有的 `.cu` 文件加入构建；在 ROCm PyTorch 环境中，该分支由 ROCm 扩展构建链处理。

**注意**：这不会让纯 CPU 环境自动生成可用算子，只是绕过 Python 层的设备可用性判断；
后续仍需要可用的 CUDA/HIP 编译工具链。

**接入方式**：这是构建脚本改造，不是运行时函数替换，所以不能通过 recipe 管理，改由
`build.py:163 force_geometric_kernel_extension()` 以离线补丁形式覆盖（见 §3.7.2 K1）。

#### B11. PV 掩码向量化重采样

`nuscenes_offlinemap_dataset.py` 的 `VectorizedLocalMap.line_ego_to_pvmask`：

```diff
-        distances = np.linspace(0, line_ego.length, 200)
-        coords = np.array([list(line_ego.interpolate(distance).coords)
-                           for distance in distances]).reshape(-1, 2)
+        coords = self.sample_line(line_ego, 200)        # numpy 弧长等距重采样
         ...
-        pix_coords = perspective(lidar_coords, lidar2feat)
+        pix_coords = self.project_points(coords, lidar2feat, z=z)
```

参考实现新增了 `sample_line`（`line_ego.xy` → 累积弧长 → `np.interp`）与`project_points`（齐次坐标矩阵乘 + `z > 0` 过滤）两个方法，`line_ego_to_pvmask`只剩两行。调用方是 `for cam_index in range(num_cam)`（nuScenes 为 6 路），所以每条中心线要付 `6 × 200` 次 shapely `interpolate`；`interpolate` 每次都会构造一个 GEOS `Point` 对象，属于 dataloader worker 里的纯 CPU 开销。

**优点**：采样退化成 numpy 向量运算，掩码结果与官方逐点采样一致（shapely 在段内
线性插值，`np.interp` 在同一弧长轴上做同一件事，只差 float64 舍入）。

**同一文件里的另外两处小改动**（同属 dataloader 的 CPU 路径）：

```diff
-                vectors.append((LineString(np.array(instance)), self.CLASS2LABEL.get(vec_class, -1)))
+                vectors.append((LineString(instance), self.CLASS2LABEL.get(vec_class, -1)))
```
`LineString` 本来就接受顶点序列，`np.array()` 只是多一次拷贝。

```diff
-        line_ego = affinity.scale(line_ego, self.scale_x, self.scale_y, origin=(0, 0))
-        line_ego = affinity.affine_transform(line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
-        coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
-        cv2.polylines(mask, np.int32([coords]), False, color=color, thickness=thickness)
+        line_ego = self.scale_translate_geom(line_ego, self.scale_x, self.scale_y, trans_x, trans_y)
+        coords = np.asarray(line_ego.coords, dtype=np.int32)
+        cv2.polylines(mask, [coords], False, color=color, thickness=thickness)
```
`scale_translate_geom` 把「先缩放再平移」合成一次 `affinity.affine_transform`
（推导见 §4.11），结果与两次变换逐点一致。

**本包落地**：`scale_translate_geom` 已作为模块级函数放进 `pv_mask.py`，并由 `maptrv2.pv_mask` 的第二个 `replace` 成员替换 `VectorizedLocalMap.line_ego_to_mask`（不往模型类里塞新方法）。回归测试对拍官方「`scale` + 平移」两步写法：随机线段
300/300 掩码逐像素相同，非零 `origin` 下坐标严格相等（见 §4.11 与包内 README）。同文件 `gen_vectorized_samples` 里的 `LineString(np.array(instance))` → `LineString(instance)`
**仍未覆盖**：该方法约 90 行且控制流密集，为省一次顶点拷贝整体替换不划算，理由记在该包 README 的「不在本包范围内的参考实现改动」。

#### B12. NUMA 亲和启动

优化侧新增了一个启动脚本，把每个 rank 绑到该 DCU 所属的 NUMA 节点上：

```bash
torchrun $DISTRIBUTED_ARGS --no-python bash -c '
    numa_map=( $(hy-smi --showtopo | grep "Numa Node" | awk "{print \$6}") );
    LOCAL_RANK=${LOCAL_RANK:-0}
    NUMA_ID=${numa_map[$LOCAL_RANK]}
    numactl --cpunodebind=${NUMA_ID} --membind=${NUMA_ID} python tools/train.py "$@"
    ' _ $CONFIG --launcher pytorch ${@:5} --deterministic
```

`--no-python` 让 torchrun 先起 bash，由 bash 调 `numactl` 之后再拉起 python，因此 CPU 亲和与内存分配都落在该 DCU 所属的 NUMA 节点。

**优点**：8 卡训练时避免跨 NUMA 访存；dataloader 的 CPU 预处理与 H2D 拷贝都吃这一项。`hy-smi --showtopo` 负责「物理卡 → NUMA 节点」映射，不需要人工维护拓扑表。

**归属**：这是**启动/部署层**的改法，TurboPhysAI 已把它收敛成 RuntimeConfig 的`process.numa`（本包 `configs/runtime.yaml` 里 `numa: true`），由 runtime 注入`TURBO_PHYSAI_RANK_NUMA`，因此不另设 Group。



#### B13. FP32 Matmul 精度配置

`projects/mmdet3d_plugin/maptr/modules/encoder.py` 顶部（紧随 import）：

```diff
 from projects.mmdet3d_plugin.bevformer.modules.encoder import BEVFormerEncoder

+torch.set_float32_matmul_precision('high')
+
```

**优点**：允许 cuBLAS / hipBLASLt 在 fp32 GEMM 上选 TF32 等同级内核，`LSSTransform`里的 `torch.matmul`（`matmul_1/2/3`、`get_geometry`、`bev_pool`）与 attention 的`QK^T` / `PV` 直接受益，精度损失通常在小数点后 3 位量级。**归属**：TurboPhysAI 收进
`maptrv2.training`（`training.py` 的 `_matmul_precision`），默认 `high`，可用`TURBO_PHYSAI_MATMUL_PRECISION=off` 回到 torch 默认的 `highest`。

#### B14. lightop DCU 自定义 Deformable Attention Kernel

`projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py`

```diff
-from mmcv import _ext as ext_module

+try:
+    from lightop import op as ext_module   # Hygon DCU 优化的 MS-Deformable-Attn 算子
+except ImportError:
+    from mmcv import _ext as ext_module    # fallback 到 mmcv 原版
```

同文件中 `im2col_step` 由关键字传参改为位置传参，避免 Python 关键字查找开销。

**优点**：`lightop` 是 Hygon 针对 DCU 优化的 MS-Deformable-Attention CUDA 扩展，性能优于 mmcv 通用实现；`try/except` 保证在 lightop 未安装时自动 fallback 到 mmcv，不影响可移植性。

**归属**：这是运行时 import 级别的替换，写在模型源码中。TurboPhysAI 的通用 `mmcv.msda` Group patch 的是 mmcv 层的符号，与 lightop 是不同路径；若 lightop 已安装则 mmcv.msda 不会在关键路径生效。**当前未通过 TurboPhysAI recipe 管理**，属于需要随模型源码部署 lightop 库的硬件适配项。

#### B15. MIOpen / rocBLAS 运行时调优

> 背景知识

```python
# NVIDIA CUDA 生态
PyTorch
  ↓
CUDA 软件栈
  ├── CUDA Driver/Runtime → 设备、显存、流和 kernel 调度
  ├── nvcc/NVRTC          → CUDA 编译与运行时编译
  ├── cuDNN               → Conv2d、池化、归一化等
  ├── cuBLAS/cuBLASLt     → GEMM、Linear、matmul、bmm 等
  ├── NCCL                → 多卡、多机集合通信
  ├── TensorRT            → 推理图优化和部署
  ├── cuSPARSE            → 稀疏矩阵运算
  ├── cuFFT               → FFT
  ├── cuSOLVER            → 稠密与稀疏线性方程求解、特征值分解
  └── Nsight              → 性能分析与调试

# AMD ROCm 生态
PyTorch
  ↓
ROCm 软件栈
  ├── ROCm Runtime / HIP → 设备、显存、流和 kernel 调度
  ├── hipcc             → HIP/C++ 编译
  ├── MIOpen            → Conv2d、池化、归一化等
  ├── rocBLAS           → GEMM、Linear、matmul、bmm 等 BLAS 实现
  ├── hipBLAS           → BLAS 封送库(支持 rocBLAS / cuBLAS 后端)
  ├── hipBLASLt         → 高性能 GEMM 算法和布局优化
  ├── RCCL              → 多卡、多机集合通信
  ├── MIGraphX          → 推理图优化和部署
  ├── rocSPARSE         → 稀疏矩阵运算
  ├── rocFFT            → FFT
  ├── rocSOLVER         → 线性方程求解
  └── rocprof           → 性能分析
    
# Hygon DCU 兼容生态
PyTorch
  ↓
Hygon DCU 软件栈
  ├── DCU Driver + hyhal       → 硬件抽象、设备管理和资源访问
  ├── DTK/HIP Runtime          → 显存、流、kernel 调度和 HIP 接口
  ├── DTK/HIP 编译工具链         → HCU 平台上的 HIP / C++ 编译
  ├── MIOpen / hipDNN          → Conv2d、池化、归一化等
  ├── rocBLAS                  → GEMM、Linear、matmul、bmm 等 BLAS 实现
  ├── hipBLASLt              → 高性能 GEMM 算法和布局优化
  ├── RCCL                     → 多卡、多机集合通信
  ├── hy-smi                   → 设备状态和 NUMA 拓扑查询
  ├── LightOp                  → HCU 原生高性能算子
  └── TurboPhysAI              → 算子替换、图优化和训练性能优化
```

ROCm 是 AMD 推出的开源 GPU/HPC 计算平台，可类比 NVIDIA CUDA，为 PyTorch 等框架在 AMD GPU 及兼容硬件（如 Hygon DCU）上执行训练和推理提供完整软件栈。**包括：**运行时与驱动接口、HIP 编程模型、编译器及算子生成工具，以及高性能算子库（MIOpen 负责卷积等深度学习算子，rocBLAS 提供 GEMM 等基础线性代数算子，hipBLASLt 提供面向 GEMM 的高性能扩展）：

- **MIOpen** 是 ROCm 平台上的深度学习算子库，作用类似于 CUDA 平台的 cuDNN。PyTorch 在 ROCm/HCU 上执行 `Conv2d` 等卷积算子时，通常由 MIOpen 提供具体 kernel，**MIOpen 卷积**即模型中的卷积层最终由 MIOpen 实现和调度。
- **rocBLAS** 是 ROCm 平台上的 BLAS 库，作用类似于 CUDA 平台的 cuBLAS，主要**提供 GEMM 等稠密线性代数算子**。PyTorch 的 `Linear`、`matmul`、`bmm` 及 attention 中的投影矩阵运算，在 ROCm 上可能由 rocBLAS 或 hipBLASLt 执行，具体取决于 PyTorch/ROCm 版本和输入条件。
- **hipBLASLt** 是面向 ROCm/HCU 的**高性能 GEMM 扩展库**，提供算法选择、矩阵布局和 fused epilogue 等能力，可视为 rocBLAS 在部分矩阵乘场景下的专用后端。PyTorch 会按版本、shape、dtype 和布局条件决定是否使用 hipBLASLt；`ROCBLAS_MATH_MODE` 不保证覆盖 hipBLASLt 路径。

>  优化侧启动脚本 `start_mmdet3d.sh` 设置了以下环境变量：

```
export PYTORCH_MIOPEN_SUGGEST_NHWC=1
export MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32=1
export MIOPEN_FIND_MODE=1
export ROCBLAS_MATH_MODE=1
```

| 环境变量 | 作用 | 配合模型代码 | 适用范围 |
| --- | --- | --- | --- |
| `PYTORCH_MIOPEN_SUGGEST_NHWC=1` | 允许 MIOpen 为卷积选择 NHWC 实现 | 需配合 channels-last | **MIOpen 卷积：**要求输入 channels-last，不限定 FP32 |
| `MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32=1` | 允许 MIOpen 的 FP32 卷积使用 TF32 型计算配置 | 不需要 | **MIOpen 的 FP32 卷积：**不优化 FP16/BF16 卷积，不优化 rocBLAS GEMM |
| `MIOPEN_FIND_MODE=1` | 为固定卷积 shape 搜索并缓存更快的算法 | 不需要 | **MIOpen 卷积：**算法搜索，不限精度 |
| `ROCBLAS_MATH_MODE=1` | 允许 rocBLAS 为 FP32 GEMM 选择快速数学内核 | 不需要 | **FP32 GEMM/matmul：**不优化 MIOpen 卷积 |

四个变量都是库级运行时配置，只有训练实际经过对应算子时才可能生效如果只设置变量但没有相应 Conv/GEMM 或 channels-last 输入，不会自动获得收益。变量是进程级环境变量，不是可原子替换的 Python 符号，应在 `python`/`torchrun` 启动前注入。TurboPhysAI 已将其映射到本包 `configs/runtime.yaml` 的 `environment.set`，通过 `turbo-physai run --runtime-config ...` 注入。

**适用与限制**：

- `PYTORCH_MIOPEN_SUGGEST_NHWC` 和 `MIOPEN_FIND_MODE` 作用于 MIOpen 卷积，不限定具体精度；`MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32` 只作用于 FP32 Conv；`ROCBLAS_MATH_MODE` 只作用于对应 FP32 GEMM/matmul。
- MapTRv2 配置启用了 `fp16`，因此图像 backbone 的 Conv 通常走 FP16，不应把这项优化描述为覆盖整个 backbone；强制 FP32 的 LSSTransform、几何和部分 head 计算才是主要验证范围。
- 仅设置变量不保证加速，实际效果取决于 MIOpen/rocBLAS 版本、输入 layout、shape 稳定性和硬件拓扑。
- `MIOPEN_FIND_MODE` 首次运行会增加算法搜索时间，应保留算法缓存并区分预热与稳态性能。
- TF32/快速数学模式可能改变数值结果，应比较 loss、梯度、mAP/chamfer 和不确定性。

---

## 2. TurboPhysAI 接入与注册

**状态：已实现**，代码在 `TurboPhysAI/turbo_physai/optimizations/models/maptrv2_optimization/`。

它现已按 TurboPhysAI 的内置模型包布局组织：模块直接位于`turbo_physai/optimizations/models/maptrv2_optimization/`，测试位于仓库`test/optimizations/`，通过 `turbo_physai.optimizations.models.maptrv2_optimization.catalog`导入，无需为本模型包单独执行 `pip install -e .`；`configs/optimization.yaml` 由 `generate` 在干净基线 worktree 上产出，不进版本库（`trust` 是 target 的源码/AST 哈希，手写必然过期）。

下面 §2.1-§2.6 保留动手前的方案草稿以便对照，落地时的三处主要差异是：

1. **target 一律改用基线符号**。草稿里的 `tools.train.main`、`LSSTransform.matmul_1/2/3`、
   `extract_metas`、`down_sample`、`initialize_queries_and_bev`、`compute_decoder_predictions`、
   `models.opt.adamw.AdamW2` 只存在于参考实现，官方 `maptrv2` 分支没有；
2. **实现按能力分模块**（`training.py`/`data.py`/`grid_mask.py`/`compile.py`/`match_cost.py`/
   `pv_mask.py` + `compat.py`），没有再建 `replacements/` 子包；
3. **新增 B11**（PV 掩码重采样），草稿里没有。

逐条对照表见该包 `README.md` 的「与 `note.md` §2.2 草案的差异」，实际声明见 §2.2b。

### 2.1 目录结构

实际落地（草稿的 `replacements/*` 子包被按能力拆分的模块取代）：

```
maptrv2_optimization/
├── configs/
│   ├── recipe.yaml                  # Group 开关（手写，唯一真源）
│   ├── runtime.yaml                 # RuntimeConfig：MIOpen / Inductor / NUMA
│   └── optimization.yaml            # generate 产物，不手写（当前未生成）
├── maptrv2_optimization/
│   ├── __init__.py
│   ├── catalog.py                   # Group 声明（唯一注册入口）
│   ├── compat.py                    # runtime_condition 与能力探测（懒加载 torch）
│   ├── training.py                  # B1/B2：channels-last、cuDNN、fork
│   ├── data.py                      # B3 + fork 修复：pin_memory、DataLoader 级 multiprocessing_context
│   ├── grid_mask.py                 # B4：Dynamo 安全的 GridMask.forward
│   ├── compile.py                   # B5/B8：compile_wrapper / dynamo_disable_wrapper
│   ├── match_cost.py                # B7：cdist(p=1) → 广播减法
│   ├── pv_mask.py                   # B11：PV 重采样 + BEV 掩码变换合并
│   └── replacements.py              # 实现索引（只有文档字符串，供人读）
├── pyproject.toml
├── README.md                        # Group 表、差异说明、上机步骤
└── tests/
    ├── test_catalog.py              # 结构 + 懒加载 + condition 契约
    └── test_implementations.py      # 与官方实现的数值等价（unittest）
```

B9（EfficientNet）不占模块：它由 `registry_override` 在 catalog 里直接声明。

B10（强制选择 `CUDAExtension`）**不通过 recipe yaml 管理**，因为它是扩展构建逻辑而不是
运行时替换。若后续要提供独立算子实现，应把 kernel 源码放到
`TurboPhysAI/kernel/geometric_kernel_attn/`，Python wrapper 放到
`turbo_physai/operators/geometric_kernel_attention.py`，再通过 `replace` 引用；不能把权威
版本误写成“新增了 HIP 源文件”。

### 2.2 Catalog 声明草案

关键点：`target` 用**基线**（v2 分支）中的符号；`replacement` / `wrapper` 指向本包内实现；**不要在模块顶层 `import torch`**（保持规划期 lazy）。

> 下面这段是当时的初稿：意图是"用基线符号"，但 `target` 实际抄自参考实现，**没有**逐个
> 回基线核对，照抄会在 `optimization check` 报 target 不存在。最终声明见 §2.2b。

```python
# maptrv2_optimization/catalog.py
from turbo_physai import group, replace, wrap

# B1
CHANNELS_LAST = group(
    "maptrv2.channels_last",
    wrap(
        target="tools.train.main",
        wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.channels_last.wrap_backbone_channels_last",
    ),
    wrap(
        target="projects.mmdet3d_plugin.bevformer.apis.mmdet_train.custom_train_detector",
        wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.channels_last.wrap_ddp_channels_last",
    ),
    replace(
        target="projects.mmdet3d_plugin.datasets.pipelines.transform_3d.TransposeImage",
        replacement="turbo_physai.optimizations.models.maptrv2_optimization.replacements.channels_last.TransposeImage",
    ),
)

# B2
CUDNN_FLAGS = group(
    "maptrv2.cudnn_flags",
    wrap(
        target="tools.train.main",
        wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.cudnn_flags.enable_cudnn_bench",
    ),
    wrap(
        target="tools.train.__main__",
        wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.cudnn_flags.set_fork_start_method",
    ),
)

# B3
DATALOADER_PIN = group(
    "maptrv2.dataloader.pin_memory",
    replace(
        target="projects.mmdet3d_plugin.datasets.builder.build_dataloader",
        replacement="turbo_physai.optimizations.models.maptrv2_optimization.replacements.dataloader.build_dataloader_pinned",
    ),
)

# B4
GRID_MASK_DYNAMO = group(
    "maptrv2.grid_mask.dynamo_safe",
    replace(
        target="projects.mmdet3d_plugin.models.utils.grid_mask.GridMask.forward",
        replacement="turbo_physai.optimizations.models.maptrv2_optimization.replacements.grid_mask.forward_dynamo_safe",
    ),
)

# B5
COMPILE_HOOKS = group(
    "maptrv2.torch_compile",
    wrap(target="projects.mmdet3d_plugin.maptr.modules.transformer.MapTRPerceptionTransformer.initialize_queries_and_bev",
         wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.compile_hooks.compile_maxautotune"),
    wrap(target="projects.mmdet3d_plugin.maptr.modules.encoder.LSSTransform.matmul_1",
         wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.compile_hooks.compile_maxautotune"),
    wrap(target="projects.mmdet3d_plugin.maptr.modules.encoder.LSSTransform.matmul_2",
         wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.compile_hooks.compile_maxautotune"),
    wrap(target="projects.mmdet3d_plugin.maptr.modules.encoder.LSSTransform.matmul_3",
         wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.compile_hooks.compile_maxautotune"),
    wrap(target="projects.mmdet3d_plugin.maptr.modules.encoder.LSSTransform.extract_metas",
         wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.compile_hooks.compile_maxautotune"),
    wrap(target="projects.mmdet3d_plugin.maptr.modules.encoder.LSSTransform.down_sample",
         wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.compile_hooks.compile_maxautotune"),
    wrap(target="projects.mmdet3d_plugin.maptr.modules.decoder.MapTRDecoder.forward",
         wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.compile_hooks.compile_maxautotune"),
    wrap(target="projects.mmdet3d_plugin.maptr.detectors.maptrv2.MapTRv2.extract_img_feat",
         wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.compile_hooks.compile_maxautotune"),
    wrap(target="projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.MapTRv2Head.compute_decoder_predictions",
         wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.compile_hooks.compile_maxautotune"),
    runtime_condition="turbo_physai.optimizations.models.maptrv2_optimization.compat.torch_compile_available",
)

# B7
CDIST_BBOX_COST = group(
    "maptrv2.cdist_bbox_cost",
    replace(
        target="projects.mmdet3d_plugin.maptr.losses.map_loss.OrderedPtsL1Cost.__call__",
        replacement="turbo_physai.optimizations.models.maptrv2_optimization.replacements.cdist_bbox_cost.ordered_pts_l1_broadcast",
    ),
)

# B8 —— 注意会改变 assign() 的调用约定，需要同步替换调用侧的 GT 打包函数
ASSIGNER_STATIC = group(
    "maptrv2.assigner.static_hungarian",
    replace(
        target="projects.mmdet3d_plugin.maptr.assigners.maptr_assigner.MapTRAssigner.assign",
        replacement="turbo_physai.optimizations.models.maptrv2_optimization.replacements.assigner_static.assign_static",
    ),
    replace(
        target="projects.mmdet3d_plugin.maptr.assigners.maptr_assigner.MapTRAssigner.hungarian_match",
        replacement="turbo_physai.optimizations.models.maptrv2_optimization.replacements.assigner_static.hungarian_match",
    ),
    wrap(
        target="projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.MapTRv2Head._get_target_single",
        wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.assigner_static.pack_gt_for_static_assign",
    ),
)

# B9
EFFICIENTNET_FORCE = group(
    "maptrv2.efficientnet.force_register",
    wrap(
        target="projects.mmdet3d_plugin.models.backbones.efficientnet.EfficientNet",
        wrapper="turbo_physai.optimizations.models.maptrv2_optimization.replacements.efficientnet.force_register",
    ),
)

__all__ = [
    "CHANNELS_LAST", "CUDNN_FLAGS", "DATALOADER_PIN",
    "GRID_MASK_DYNAMO", "COMPILE_HOOKS", "CDIST_BBOX_COST",
    "ASSIGNER_STATIC", "EFFICIENTNET_FORCE",
]
```

### 2.2b 实际 Catalog 声明

上面 §2.2 是初稿，`target` 大多取自参考实现。下面是当前与权威基线差异对应的 Group
声明（节选，完整文件见 `maptrv2_optimization/catalog.py`）：

```python
# maptrv2_optimization/catalog.py（节选）
_COMPILE_CONDITION = "turbo_physai.optimizations.models.maptrv2_optimization.compat.torch_compile_available"
_DYNAMO_CONDITION = "turbo_physai.optimizations.models.maptrv2_optimization.compat.dynamo_available"
_PV_MASK_CONDITION = "turbo_physai.optimizations.models.maptrv2_optimization.compat.pv_mask_sampling_enabled"

TRAINING = group("maptrv2.training", wrap(
    target="projects.mmdet3d_plugin.bevformer.apis.mmdet_train.custom_train_detector",
    replacement="turbo_physai.optimizations.models.maptrv2_optimization.training.training_runtime_wrapper"))

DATA = group("maptrv2.data", replace(
    target="projects.mmdet3d_plugin.datasets.builder.build_dataloader",
    replacement="turbo_physai.optimizations.models.maptrv2_optimization.data.build_dataloader"))

GRID_MASK = group("maptrv2.grid_mask", replace(
    target="projects.mmdet3d_plugin.models.utils.grid_mask.GridMask.forward",
    replacement="turbo_physai.optimizations.models.maptrv2_optimization.grid_mask.grid_mask_forward"))

COMPILE = group("maptrv2.compile", *(
    wrap(target=target,
         replacement="turbo_physai.optimizations.models.maptrv2_optimization.compile.compile_wrapper",
         runtime_condition=_COMPILE_CONDITION)
    for target in _COMPILE_TARGETS))            # 10 个挂点，见下表

MATCH_COST = group("maptrv2.match_cost", replace(
    target="projects.mmdet3d_plugin.maptr.losses.map_loss.OrderedPtsL1Cost.__call__",
    replacement="turbo_physai.optimizations.models.maptrv2_optimization.match_cost.ordered_pts_l1_cost_call"))

PV_MASK = group("maptrv2.pv_mask",
    replace(target="projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
                   "VectorizedLocalMap.line_ego_to_pvmask",
            replacement="...pv_mask.line_ego_to_pvmask",
            runtime_condition=_PV_MASK_CONDITION),
    replace(target="...VectorizedLocalMap.line_ego_to_mask",
            replacement="...pv_mask.line_ego_to_mask",
            runtime_condition=_PV_MASK_CONDITION))   # 两个成员共用同一个开关

ASSIGNER = group("maptrv2.assigner", wrap(
    target="projects.mmdet3d_plugin.maptr.assigners.maptr_assigner.MapTRAssigner.assign",
    replacement="turbo_physai.optimizations.models.maptrv2_optimization.compile.dynamo_disable_wrapper",
    runtime_condition=_DYNAMO_CONDITION))

EFFICIENTNET = group("maptrv2.efficientnet", registry_override(
    module="projects.mmdet3d_plugin.models.backbones.efficientnet",
    registry="mmdet.models.builder.BACKBONES",
    names=("EfficientNet",)))

```

10 个 compile target（都在基线 worktree `e03f097` 上定位过）：

| # | target（基线符号） | 基线位置 | 参考实现对应挂点 |
| --- | --- | --- | --- |
| 1 | `MapTRPerceptionTransformer.format_feats` | `maptr/modules/transformer.py:288` | 同名 |
| 2 | `MapTRDecoder.forward` | `maptr/modules/decoder.py:23` | 同名 |
| 3 | `MapTRv2.extract_img_feat` | `maptr/detectors/maptrv2.py:75` | 同名 |
| 4 | `LSSTransform.get_cam_feats` | `maptr/modules/encoder.py:1095` | 同名；参考实现的 `down_sample` 另有独立挂点，基线无对应符号（见下） |
| 5 | `LSSTransform.get_mlp_input` | `maptr/modules/encoder.py:1175` | 同名 |
| 6-10 | `normalize_3d_pts`、`normalize_2d_bbox`、`normalize_2d_pts`、`denormalize_2d_bbox`、`denormalize_2d_pts` | `maptr/dense_heads/maptrv2_head.py:25/37/49/59/68` | 同名（模块级函数） |

参考实现另外 8 处挂点落在它新抽出的 helper 上（`matmul_1/2/3`、`extract_metas`、`initialize_queries_and_bev`、`compute_decoder_predictions`、`prepare_transformer_inputs`、`down_sample`），基线没有对应符号，同样的逻辑内联在
（`MapTRPerceptionTransformer.forward`、`BaseTransform.get_geometry_v1`/`forward`/`bev_pool`、`LSSTransform.forward`、`MapTRv2Head.forward`）里，体积大且含控制流，本包有意不包裹，仍走 eager。第 19 处挂点是 `MapTRAssigner.assign`，由 `maptrv2.assigner`单独处理（方向不同，见 §3.2）。逐条对照、断裂点分析与替代路线见 **§4.12**。

**`down_sample` 特别说明**：它的逻辑在基线里并不位于 `get_cam_feats` 内，而是分散在`BaseTransform.bev_pool`（collapse Z 的 `permute`+`cat`）与 `LSSTransform.forward`（`self.downsample`）两处，因此编译 `get_cam_feats` 覆盖不到它。如需覆盖，应把
`BaseTransform.bev_pool` 或 `LSSTransform.forward` 补进 `_COMPILE_TARGETS`。

### 2.3 实现模块约定

- **`wrap` 用于外套壳**（`training` / `compile` / `assigner`）：签名固定为`def wrapper(original, options) -> callable`；`options` 是 Group 声明里的 `options:`，没写就是空 dict。
- **`replace` 用于整体替换**（`data` / `grid_mask` / `match_cost` / `pv_mask`）：新函数签名必须兼容目标；方法型目（`GridMask.forward`）写成带 `self` 的普通函数即可。
- **`runtime_condition` 必须是非类 callable**，按被包裹调用的 `(*args, **kwargs)` 调用并返回 `bool`，返回 `False` 时引擎回退原函数；本包统一用 `(*args, **kwargs)` 签名，实现见 `compat.py`。
- **所有 `torch`/`mmcv`/`cv2`/`scipy` import 放函数体内**，保证 `check`/`plan`这类只读阶段不拉起重型依赖`test/optimizations/test_maptrv2_catalog.py` 用子进程断言导入 catalog 后`sys.modules` 里没有 `torch`。
- **依赖可用性探测写在 wrapper 内部**：`wrap` 的 wrapper 在 `prepare()` 阶段就会被调用一次，在那里 `import torch` 失败会直接把 apply 打断；`compile_wrapper` 就是据此在内部判断 `compat.torch_compile_available()`，不可用时原样返回 `original`。
- 参考例子：

```python
# replacements/compile_hooks.py
def compile_maxautotune(original):
    import torch
    return torch.compile(original, mode="max-autotune-no-cudagraphs")
```

```python
# replacements/channels_last.py
def wrap_backbone_channels_last(original):
    import torch, tools.train as _t
    real_build = _t.build_model
    def build_and_convert(*a, **kw):
        m = real_build(*a, **kw)
        m.img_backbone = m.img_backbone.to(memory_format=torch.channels_last)
        return m
    def _main(*args, **kwargs):
        _t.build_model = build_and_convert
        try: return original(*args, **kwargs)
        finally: _t.build_model = real_build
    return _main
```

```python
# replacements/cdist_bbox_cost.py
def ordered_pts_l1_broadcast(self, bbox_pred, gt_bboxes):
    import torch
    num_gts, num_orders = gt_bboxes.shape[:2]
    bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)
    gt_bboxes = gt_bboxes.flatten(2).view(num_gts * num_orders, -1)
    return (bbox_pred[:, None, :] - gt_bboxes[None, :, :]).abs().sum(dim=-1) * self.weight
```

```python
# replacements/assigner_static.py
def hungarian_match(self, cost, gt_labels, assigned_gt_inds, assigned_labels, num_gts, device):
    import torch
    from scipy.optimize import linear_sum_assignment
    from mmdet.core.bbox.assigners.assign_result import AssignResult
    row, col = linear_sum_assignment(cost.detach().cpu())
    row = torch.as_tensor(row, device=device); col = torch.as_tensor(col, device=device)
    assigned_gt_inds[:] = 0
    assigned_gt_inds[row] = col + 1
    assigned_labels[row] = gt_labels[0][col]
    return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
# assign_static / pack_gt_for_static_assign 分别按 diff 里的新签名实现
```

### 2.4 Recipe 配置

> **这是动手前的草案，不是当前生效的配置。** 草案里的 group id（`maptrv2.channels_last`、
> `maptrv2.cudnn_flags`、`maptrv2.dataloader.pin_memory`、`maptrv2.grid_mask.dynamo_safe`、
> `maptrv2.cdist_bbox_cost`、`maptrv2.efficientnet.force_register`、
> `maptrv2.assigner.static_hungarian`、`maptrv2.torch_compile`）与 `extends` 的三条并列 id
> 都没有落地。实际生效的是 `configs/recipe.yaml`（手写源）与 `configs/optimization.yaml`
> （生成产物），group id 为 12 个 `maptrv2.*`，`extends` 处于注释状态——对照 §2.6 与 §3.8.4。
> 下面保留草案原文，用于说明设计意图与最终形态的差异。

```yaml
schema_version: turbophysai/optimization-config/v1
kind: OptimizationConfig

metadata:
  id: model.maptrv2.development.hcu
  version: "0.1.0"
  description: MapTRv2 HCU/DCU optimizations

model:
  name: maptrv2

optimization_modules:
  - turbo_physai.optimizations.models.maptrv2_optimization.catalog

extends:
  - common.hcu.base
  - common.mmcv.hcu          # 若引用了 mmcv.mdc / mmcv.msda
  - common.mmdet3d.hcu       # 若引用了 bev_pool / voxelization

compatibility:
  torch: ">=2.0"
  mmcv: ">=1.5,<2.0"

optimization_groups:
  - id: maptrv2.channels_last
    enabled: true
  - id: maptrv2.cudnn_flags
    enabled: true
  - id: maptrv2.dataloader.pin_memory
    enabled: true
  - id: maptrv2.grid_mask.dynamo_safe
    enabled: true
  - id: maptrv2.cdist_bbox_cost
    enabled: true
  - id: maptrv2.efficientnet.force_register
    enabled: true
  - id: maptrv2.assigner.static_hungarian
    enabled: false           # 调用约定变更，需业务侧配合验证
  - id: maptrv2.torch_compile
    enabled: false           # 收敛/精度通过后再开
```

### 2.5 配置生成与运行

在 MapTR 仓库（`maptrv2` 分支）干净、commit 正确时，**先生成再校验**（`check` 的入参是生成出来的 YAML，没有 `--recipe`/`--commit`）：

```bash
cd /path/to/TurboPhysAI

turbo-physai optimization generate \
  --recipe configs/recipe.yaml \
  --repo /path/to/MapTR \
  --commit e03f097 \
  --output configs/optimization.yaml

turbo-physai optimization check configs/optimization.yaml --repo /path/to/MapTR

turbo-physai run \
  --optimization-config configs/optimization.yaml \
  --runtime-config configs/runtime.yaml \
  python tools/train.py <原训练参数>
```

`generate` 会校验仓库状态与 commit、逐个解析 `target` 并采集 source/AST 证据，
所以「target 不存在」在这一步就会暴露；输出文件已存在时要加 `--force`。

### 2.6 与公共优化层的关系

`turbo_physai/optimizations/common/{mmcv,mmdet3d}` 已提供通用替换（`mdc`、`msda`、`bev_pool`、`voxelization` 等）。若 MapTRv2 用到，**只通过 `extends` 引用**，不要在本包重复声明。

**设计上的继承链**（`extends` 只有一层 `common.hcu.base`，不是本节最初设想的 `common.hcu.base`/`common.mmcv.hcu`/`common.mmdet3d.hcu` 三个并列 id——那三个 id 在 §2.4 的草稿 YAML 里出现过，是没有落地的早期设计）：

```
maptrv2 的 recipe.yaml（id: model.maptrv2.development.hcu）
  extends: common.hcu.base
      │
      ▼
common/configs/recipe.yaml（id: common.hcu.base）
  extends:
    - framework.mmcv.hcu
    - framework.mmdet3d.hcu
      │
      ├──▼ common/mmcv/configs/optimization.yaml（id: framework.mmcv.hcu）
      │     optimization_modules: [...common.mmcv.catalog]
      │     optimization_groups:
      │       - id: mmcv.msda, enabled: true
      │
      └──▼ common/mmdet3d/configs/optimization.yaml（id: framework.mmdet3d.hcu）
            optimization_groups:
              - mmdet3d.gaussian
              - mmdet3d.bev_pool
              - mmdet3d.quick_cumsum
              - mmdet3d.voxelization
              - mmdet3d.canonical_indice_pairs
              - mmdet3d.sparse_tensor
```

解析机制在 `turbo_physai/engine/config/loader.py`：`OptimizationConfigCatalog.from_builtin_files()` 启动时 glob 全部 `optimizations/**/configs/optimization.yaml`，按各自 `metadata.id` 注册进同一个目录；`_resolve_extends()` 递归取出 `extends` 列出的父配置先展开，再用 `_merge()` 按 group `id` 去重合并（子配置同 id 覆盖父配置字段，否则原样带过来），`optimization_modules` 同样合并去重。所以继承链接通时，最终 `configs/optimization.yaml` 底部会列出三个模块。

> **注意：这条链当前处于断开状态。**
> `maptrv2_optimization/configs/recipe.yaml:18-20` 里 `extends` 整段被注释掉：
>
> ```yaml
> # extends:
> #   # 继承自 common.hcu.base 配置，包含了基础的优化配置和通用设置
> #   - common.hcu.base
> ```
>
> 因此生成的 `optimization.yaml` **没有 `extends` 键**，`optimization_modules` 只有一行
> （`optimization.yaml:183-184`）：
>
> ```yaml
> optimization_modules:
> - turbo_physai.optimizations.models.maptrv2_optimization.catalog
> ```
>
> 上面那张图描述的是**设计意图**，不是当前生效的状态。在当前配置下，
> `mmcv.msda` 与 6 个 `mmdet3d.*` 这 7 个框架层 group **不会被激活**——这是 §3.6.2
> 「核实结论」需要随之改写的原因。同族的 `bevformer/configs/recipe.yaml:14-15` 与
> `bevfusion/configs/recipe.yaml:14-15` 仍然接着这条链，只有 MapTRv2 的 recipe 断开了
> （MapTRv2 自己用 `maptrv2.bev_pool_fix` 等专属 group 覆盖了同一条路径，见 §3.7.1 U3）。
>
> 另外 `common/mmcv/catalog.py:11-25` 定义了一个 `mmcv.mdc` group，但它**没有被任何
> 配置文件的 `optimization_groups` 引用**，属于定义了但从未启用的项。

`maptrv2_optimization/catalog.py` 的 `__all__` 列出 12 个 `maptrv2.*` group；`mmcv.msda` 和 6 个 `mmdet3d.*` group 的定义（`id`、`optimization_modules`、默认 `enabled`）完全来自这条 `extends` 链，本包代码里既不定义也不注册它们。继承链一旦重新接通，它们会被并入最终生成的配置；当前断开状态下则完全不在配置里。

---

## 3. 实施要点

落地时需要同时守住三条边界：**什么该进优化包、哪些改动会改变调用约定、哪些开关只能在
运行时决定**。下面按“结论 -> 原因 -> 操作方式”展开。

### 3.1 改动分类边界

| 类型 | 典型内容 | 放置位置 | 是否进 recipe |
| --- | --- | --- | --- |
| A. 用户配置 / 调试 | batch size、worker 数、epoch、日志间隔 | MapTR 模型仓库 | 否 |
| B. 性能 / 硬件适配 | channels-last、cuDNN 标志、compile、算子替换 | TurboPhysAI 优化包 | 是 |
| 版本差异 | MapTRv2 相对 v1 的模型结构与新能力 | MapTR 模型仓库 | 否 |

### 3.2 Assigner 静态化约束

**结论：`maptrv2.assigner` 不是一个函数替换，而是 head 与 assigner 之间的数据契约改造，
所以它必须一次性覆盖整条调用链。**

```text
基线调用约定
MapTRv2Head._get_target_single
        │  gt_bboxes: (N, 4), gt_labels: (N,), gt_pts: (...)
        └──> MapTRAssigner.assign(...)

静态化调用约定
MapTRv2Head._get_target_single
        │  pad_to_static_list(...)
        │  [(padded(MAX_GTS, 4), valid_mask(MAX_GTS), num_gts), ...]
        └──> MapTRAssigner.assign(...)
```

固定 shape 有利于 Dynamo，但上游必须同步产出“预 pad + valid mask + num_gts”三元组，
否则会在 assigner 内发生 shape 不匹配。完整改造还会涉及`loss`、`get_targets`、`_get_target_single`、`assign`、`sampler.sample` 等多处调用方，少改一处就可能静默出错，无法用一个原子替换安全表达。

本包的落地方式是在 `assigner.py` 里实现完整契约，再用 3 个 `replace` 目标一次性覆盖这条链：
`MapTRAssigner.assign`、`MapTRv2Head.loss`、`MapTRv2Head._get_target_single`。其中
`pad_to_static_list` 负责产出"预 pad + valid mask + num_gts"三元组，`_hungarian_match_impl`
用 `torch._dynamo.disable` 把 SciPy 求解隔离出 Dynamo 图。详见 §3.7.1 U4。

```yaml
- id: maptrv2.assigner
  enabled: true
```

`assign` 自身的 `torch.compile` 不在这三个目标的行为里，由
`TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE` 单独控制，默认 `"0"`（不编译）；开启前应做业务级
精度 A/B。

### 3.3 `runtime_condition` 机制

catalog 中保存的是 condition 的字符串路径，engine 在运行时解析并调用它：

```text
catalog.runtime_condition
        │ resolve_replacement(path)
        ▼
condition(*args, **kwargs) -> bool
        ├── True  -> 使用 replacement
        └── False -> 回退 original
```

因此 condition 统一使用宽松签名，不依赖被包装方法的真实参数：

```python
def condition(*_args, **_kwargs) -> bool:
    return True  # True 启用替换；False 使用原函数
```

`compat.py` 为权威优化提供三个能力探测：

| 探测函数 | 用途 | 判断方式 |
| --- | --- | --- |
| `torch_compile_available` | `maptrv2.compile` 的 `runtime_condition` | `TURBO_PHYSAI_DISABLE_TORCH_COMPILE=1` 或缺少 `torch.compile` 时返回 `False` |
| `dynamo_available` | `maptrv2.assigner` 的 `runtime_condition` | 缺少 `torch._dynamo.disable` 时返回 `False` |
| `pv_mask_sampling_enabled` | `maptrv2.pv_mask` 的 `runtime_condition` | `TURBO_PHYSAI_DISABLE_PV_MASK=1` 时返回 `False` |
这三个都是 `runtime_condition`。另一类环境变量（如
`TURBO_PHYSAI_CHANNELS_LAST_SCOPE`、`TURBO_PHYSAI_MATMUL_PRECISION`）只是 wrapper 内部的
运行时选项，不是 condition。

### 3.4 测试与验证

测试分三层，从低成本静态检查逐步走到设备和训练验证：

```text
静态检查
  py_compile + catalog 声明检查（不导入 torch）
        │
        ▼
CPU 行为测试
  替换实现与官方实现数值/逐像素对拍
        │
        ▼
HCU/DCU 上机验证
  generate/check -> 短训练 -> loss/mAP 与性能 A/B
```

仓库级 `test/optimizations/` 当前包含三个文件：

- `test_maptrv2_catalog.py`：检查全部 Group、registry/`__all__` 一致性、可选 Group 的
  condition 与成员数、condition 契约和 `target` 是否存在，并断言导入 catalog 不会拉起 torch。
- `test_maptrv2_implementations.py`：覆盖 channels-last 作用域、`OrderedPtsL1Cost` 与
  `torch.cdist(p=1)` 数值等价、GridMask 只读缓冲区、PV mask/几何变换与官方结果一致，以及
  `bev_pool_fix` 对 NCHW/NHWC 两种 layout 的输出形状回归（§3.7.1/U3）。
- `test_maptrv2_data.py`：对 `build_dataloader()` 打桩 `mmcv`/`mmdet`/`projects.*` 依赖，验证
  默认 `fork` context、显式 `spawn` 覆盖、`TURBO_PHYSAI_FORK_START_METHOD=0` 关闭覆盖三种分支
  都传给底层 `DataLoader` 正确的 `multiprocessing_context`（§3.7.1/U5）。

测试由 `TurboPhysAI/pytest.ini` 的 `testpaths = test` 收集。需要 HCU/DCU 的用例放在仓库级
`test/`，并标记为 `@pytest.mark.hcu`。

完整验证顺序：

```bash
cd /path/to/TurboPhysAI

pytest test/optimizations/test_maptrv2_catalog.py \
       test/optimizations/test_maptrv2_implementations.py \
       test/optimizations/test_maptrv2_data.py

turbo-physai optimization generate \
  --recipe turbo_physai/optimizations/models/maptrv2_optimization/configs/recipe.yaml \
  --repo /path/to/MapTR --commit e03f097 --output configs/optimization.yaml

turbo-physai optimization check configs/optimization.yaml --repo /path/to/MapTR

turbo-physai run \
  --optimization-config configs/optimization.yaml \
  --runtime-config configs/runtime.yaml \
  python tools/train.py <原训练参数>
```

`generate` 会校验仓库状态、commit 和每个 `target`，所以基线符号不存在时会在此处直接报错。训练跑通后，`maptrv2.compile` / `maptrv2.reference_boundaries` / `maptrv2.assigner` 在配置里已经是 `enabled: true`（§3.8.4），所以 A/B 的做法是**逐个改成 `enabled: false` 取对照组**，分别与基线对齐 loss 曲线和 nuScenes mAP，并记录性能数据。

**当前状态**：删除非权威扩展前，`maptrv2_optimization/` 下 12 个 `.py` 已通过`python -m py_compile`；Python 3.11 + CPU torch 2.14 + numpy/shapely/opencv/pillow 环境下，上述两个测试文件得到 **30 passed、53 subtests passed、0 skipped**。测试用最小 `mmcv` 桩替代
`runner.auto_fp16`，GridMask 的装饰器组合及 HCU 路径尚未实测。删除相关代码后需要重新执行定向测试。所有 `target` 已在基线 worktree `e03f097` 定位到定义行，Group ID、recipe 和环境变量保持一致。`generate/check/run`、训练级精度/性能 A/B 仍需上机完成。核对期间还修正了两处测试缺陷：`project_points` 的期望值补上官方 `perspective()` 的 `+ 1e-7`；`MatchCost` 由 float32
逐位比较改为 float64 严格比较 + float32 `1e-5` 容差。

### 3.5 文档同步

| 文档 | 应包含 | 状态 |
| --- | --- | --- |
| `maptrv2_optimization/README.md` | Group ID、作用、启停建议、验证记录 | 已完成 |
| `model_examples/MapTRv2/README.md` / 支持清单 | 模型版本、验证 commit、启用 Group、性能对比 | 待上机 A/B 后补充 |

文档中禁止出现内部路径、凭据、令牌或未脱敏日志。支持清单必须基于已跑通的训练命令和 A/B数据，避免文档领先于事实。

### 3.6 参考实现覆盖审计

以本地 `/workspace/MapTRv2`（侵入式优化版）为准，对照本地未优化基线 `/workspace/model/MapTrv2`，使用 `diff -rq` / `diff -u` 逐文件比对，并额外检查 `mmdetection3d/`、镜像构建和启动脚本。

结论：**主要优化已经拆出并接入 TurboPhysAI，但仍有若干缺口，且部分接入项在当前配置下有冗余或旁路**。本节的逐条核实口径见 §3.8 的三方差分审计。

#### 3.6.1 MapTRv2 主体改动

| 参考实现改动 | note 记录 | `maptrv2_optimization` 现状 |
| --- | --- | --- |
| channels-last、DDP channels-last、输入 NHWC | B1 | 已接入 `maptrv2.training`；模型内存格式在 train wrapper 中设置，输入布局由 backbone pre-hook 等价实现 |
| `TransposeImage` pipeline | B1 | 参考实现定义了该类但 config 未引用；本包用 backbone pre-hook 覆盖实际训练路径，没有注册同名 pipeline |
| cuDNN benchmark / deterministic、fork | B2 | 已接入 `maptrv2.training`，均有独立 env 开关 |
| `pin_memory=True` | B3 | 已接入 `maptrv2.data` |
| GridMask dynamo 隔离、只读 numpy buffer 修复 | B4 | 已接入 `maptrv2.grid_mask`；额外改为 `x.device`，不复用参考实现的硬编码 `.cuda()` |
| 10 个基线同名 `@torch.compile()` 挂点 | B5、§2.2b | 已在 `maptrv2.compile` 声明且 `enabled: true`；参考实现 18/19 个挂点用 `mode="max-autotune-no-cudagraphs"`，与本包默认 mode **一致** |
| 另外 8 个新 helper 挂点 | B6、§4.12 | **已接入且 `enabled: true`**：`matmul_1/2/3`、`extract_metas`、`down_sample`、`initialize_queries_and_bev`、`compute_decoder_predictions`、`prepare_transformer_inputs` 在基线中不存在，无法直接声明为 `replace` 目标；`maptrv2.reference_boundaries` group 改为包装 5 个基线原生调用方（`get_geometry_v1`/`BaseTransform.forward`/`LSSTransform.forward`/`transformer.forward`/`head.forward`），让新实现内部调用这 8 个 helper，不改模型源码 |
| `cdist` 改广播减法 | B7 | 已接入 `maptrv2.match_cost` |
| assigner 静态打包 + Hungarian 隔离 | B8、§4.13 | **已接入**：`assigner.py` 实现了 `pad_to_static_list`、`get_label_result`、`_hungarian_match_impl`（`torch._dynamo.disable` 隔离 SciPy 求解）、`static_loss`、`get_target_single`，通过 `maptrv2.assigner` 的 3 个 `replace` 目标（`assign`/`loss`/`_get_target_single`）落地。唯一差异：参考对 `assign` 用 `@torch.compile(options={"triton.cudagraphs": True, "triton.cudagraph_trees": False})`，本包由 `TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE` 控制且**默认 `"0"`（不编译）**，编译选项也不同 |
| EfficientNet `force=True` 注册 | B9 | 已通过 `registry_override` 接入 `maptrv2.efficientnet` |
| SparseConv 全部 `register_module(force=True)` | §3.6.2 | 已通过 `registry_override` 接入 `maptrv2.spconv_registry`；但**当前运行时环境中是空操作**——实际导入的 mmdet3d 是 dist-packages 里那个由参考树构建的 wheel，它本身已经带 `force=True`（见 §3.6.2、§3.8） |
| 数据集向量化第二梯队（`_interpolate_line_points_vectorized`、`shift_fixed_num_sampled_points_v2` 缓存） | §3.7.1 U6 | **未接入**：`pv_mask.py` 只覆盖 `line_ego_to_pvmask`/`line_ego_to_mask`/`gen_vectorized_samples`，没有 patch 这两个 |
| DDP `ddp_static_graph=True` + `find_unused_parameters=False` | §3.7.1 U7 | **已接入** `maptrv2.ddp_static_graph`：基线 `mmdet_train.py` 没有 `ddp_static_graph` 字段，config 改不动，所以改为 wrap `mmcv.parallel.distributed.MMDistributedDataParallel.__init__`，在构造点注入 |
| GDK `geometric_kernel_attn_cuda.cu` 的 `AT_DISPATCH_FLOATING_TYPES(value.type(), …)` → `value.scalar_type()` | §3.7.2 K3 | **未接入**：`build.py` 覆盖了其它构建补丁，但不含这条 `.cu` 改动 |
| 强制选择 `CUDAExtension` 构建 | B10 | **已接入（离线构建补丁，非 recipe）**：`build.py:163 force_geometric_kernel_extension()` 对 `geometric_kernel_attn/setup.py` 做同一个 `if 1:` 替换；因为目标是构建脚本而不是运行时函数，走 `python -m ...build` 而不是 group |
| PV mask numpy 重采样、BEV affine 合并 | B11 | 已接入 `maptrv2.pv_mask` |
| `LineString(np.array(instance))` → `LineString(instance)` | B11 | **已接入**：随 `maptrv2.pv_mask` 的 `gen_vectorized_samples` 替换一起落地（`pv_mask.py` 中即 `LineString(instance)`），不需要为它单独设 group |
| NUMA rank 绑定 | B12 | 已映射到 RuntimeConfig `process.numa` |
| `set_float32_matmul_precision("high")` | B13 | 已接入 `maptrv2.training` |
| head 中 `get_label_result`、`pad_to_static_list` 抽取 | B8、§4.13 | **已接入**：两者都在 `assigner.py` 中实现（而不是作为 head 的新方法），由 `maptrv2.assigner` 的 `loss` / `_get_target_single` 替换在内部调用 |
| encoder/BEV 路径中的 `bev_pool` 返回契约和 `down_sample` 合并 | B5/B6.3、§4.12 | **已接入，但当前配置下 `bev_pool_fix` 被旁路**：ROCm `bev_pool` 返回 `[B,Z,H,W,C]` 而非 CUDA 的 `[B,C,Z,H,W]`，会让后续 `downsample` Conv2d channel 维不匹配崩溃。`maptrv2.bev_pool_fix` 在 monkey-patch 层修正 layout 并顺带做了参考的 `index_select`→`x[kept]` 布尔掩码替换。`down_sample` 的 collapse-Z+downsample 合并通过 `maptrv2.reference_boundaries` 的 `LSSTransform.forward`/`BaseTransform.forward` 包装接入（见 B6.3），现在 `enabled: true`。**两者是同一段逻辑的两个实现，而 `reference_boundaries` 的 `base_transform_forward` 直接调 `_bev_pool_5d`、不再走 `self.bev_pool`，所以 Group 全开时 `bev_pool_fix` 在主训练路径上是死代码**（它仍是 `reference_boundaries` 关闭时的兜底）。详见 §3.8 |
| `lightop` DCU Deformable Attention 算子替换 | B14 | **未接入 TurboPhysAI recipe**：以 try/except 写在模型源码中；`mmcv.msda` 替换的是 mmcv 层符号，与 lightop 路径不重叠 |

#### 3.6.2 `mmdetection3d` 与部署层

| 参考实现改动 | 当前处理 |
| --- | --- |
| `mmdet3d.ops.bev_pool.bev_pool` 去掉尾部 `permute(0,4,1,2,3)` | **在运行时已经生效，但来源不是本包**：当前安装的 mmdet3d wheel 由参考树构建，其 `bev_pool.py:96` 已经是注释态并 `return x`（5D）（见 §3.8.1）。package 侧对应的是 `maptrv2.bev_pool_fix`（默认开启）与 `maptrv2.reference_boundaries` 的 `_bev_pool_5d`，两者都按 5D 处理；`common.hcu.base` 那条通用 `mmdet3d.bev_pool` group 当前未接线 |
| SparseConv 系列全部 `register_module(force=True)` | **已接入** `maptrv2.spconv_registry`（registry_override）；但当前运行时 mmdet3d wheel 本身已带 `force=True`，实际是空操作，见 §3.7.1 U8 |
| 多个 point ops 去掉 `THC/THC.h`、改用 `ATen/cuda/CUDAContext.h` | **已接入**：`build.py` 的 `_point_op_patches()` 补 `#include <ATen/cuda/CUDAContext.h>`；属构建层，不做成 Group |
| `__CUDA_ARCH__` → `__CUDACC__`、C++14 → C++17 | **已接入**：`build.py` 的 `_build_compatibility_patches()` 覆盖这两项；属构建层 |
| mmcv 上限 1.4.0 → 1.6.2、numba import、requirements 版本解绑 | **已接入**：`build.py` 覆盖 `mmcv_maximum_version`、`numba.errors`→`numba.core.errors`、放开 numba 版本锁定；属镜像/依赖层，不做成 recipe Group |
| `start_mmdet3d.sh` 的 MIOpen、rocBLAS、NCCL、Inductor、HSA 环境变量 | **已按参考启动脚本写入 MapTRv2 runtime**：MIOpen/rocBLAS 部分见 B15；这些是参考环境的调优默认值，不代表每项都是运行必需，RCCL 拓扑项需按实际部署覆盖 |
| Dockerfile、空 `build.sh`、数据下载脚本 | **不纳入模型优化包**；属于镜像和部署资产 |
| 根目录新增 `test.py` | **不纳入模型优化包**；内容是临时 CUDA tensor 构造试验 |

**前提：运行时真正导入的 mmdet3d 不是仓库树。** `import mmdet3d` 解析到
`/usr/local/lib/python3.10/dist-packages/mmdet3d`（0.17.2）；该安装的 `direct_url.json` 为
`file:///workspace/MapTRv2/mmdetection3d/dist/mmdet3d-0.17.2-...whl`——**这个 wheel 是用侵入式参考树构建的**。
证据一致：安装版 `ops/bev_pool/bev_pool.py:96` 已经是
`#x = x.permute(0, 4, 1, 2, 3).contiguous()`，安装版 `ops/spconv/conv.py` 的 10 处
`register_module(force=True)` 已经存在。基线自带的 `mmdetection3d/` 子目录（`bev_pool.py` 仍是
`x = x.permute(...)`、spconv 0 处 `force=True`）**不被导入**，只是构建资产。
下表描述的是**仓库树**里的代码，用来对照参考实现的 diff；判断运行时行为要以上面这个 wheel 为准。

**核实结论**：§2.6 那条 `extends` 链带来的 7 个框架层公共优化（`mmcv.msda` + 6 个 `mmdet3d.*`），
在**当前配置下根本没被激活**——`recipe.yaml:18-20` 的 `extends` 整段被注释，生成的
`optimization.yaml` 无 `extends` 键，只有 `maptrv2.*` 一个模块。所以这 7 个 group 既不来自
模型源码，也不在生效配置里：

| Group | 目标符号 | 仓库源码树里的状态（不等于运行时，见上） |
| --- | --- | --- |
| `mmdet3d.bev_pool` | `mmdet3d.ops.bev_pool.bev_pool.bev_pool` | 参考树 `mmdetection3d/mmdet3d/ops/bev_pool/bev_pool.py:96` 把尾部 `x.permute(0, 4, 1, 2, 3).contiguous()` 注释掉并 `return x`（5D）——**参考确实改了 bev_pool，但方式和通用 group 不是一回事**：它把 collapse Z 挪进了 `LSSTransform.down_sample`（见 §3.6.1），而通用 `mmdet3d.bev_pool` group 是在 Python 层重写整个函数。基线树 `:96` 仍是 `x = x.permute(...)` |
| `mmdet3d.quick_cumsum` | `...bev_pool.bev_pool.QuickCumsum.forward/backward` | 两侧树的 CPU 版 `QuickCumsum` 都是原始写法：`kept[:-1] = ...`、`x[kept]`、`back[kept] -= 1`，未被侵入式改过 |
| `mmdet3d.gaussian` | `mmdet3d.core.utils.gaussian.gaussian_2d` | `core/utils/gaussian.py`：纯 `np.ogrid`/`np.exp`/`h[h < eps] = 0`，`draw_heatmap_gaussian` 里再 `torch.from_numpy(...)` 搬回 GPU；两侧树一致 |
| `mmdet3d.voxelization` | `...voxel.voxelize._Voxelization.forward` | `ops/voxel/voxelize.py`：调 `from .voxel_layer import dynamic_voxelize, hard_voxelize`——mmdet3d 自带、随源码编译的扩展；两侧树一致 |
| `mmdet3d.canonical_indice_pairs` | `mmdet3d.ops.spconv.ops.get_indice_pairs` | `ops/spconv/ops.py`：调 `sparse_conv_ext.get_indice_pairs_2d/3d/4d`——mmdet3d 自带扩展，不是 `turbo_physai.ops` |
| `mmdet3d.sparse_tensor` | `...structure.SparseConvTensor.sparity` | `ops/spconv/structure.py`：`return self.indices.shape[0] / np.prod(self.spatial_shape) / self.batch_size`，`np.prod` 写法，未替换成显式三维乘法 |
| `mmcv.msda` | `mmcv._ext.ms_deform_attn_forward/backward` | `mmcv` 是 pip 安装依赖，不在仓库源码树内。参考树在调用侧做了替换（`lightop` fallback，见 K4），但没有对 `mmcv._ext` 做 monkeypatch |

六个 `mmdet3d.*` 替换调用的都是 mmdet3d/mmdetection3d **自带的原生编译扩展**（`bev_pool_ext`、`voxel_layer`、`sparse_conv_ext`），不是 `turbo_physai.ops` 打包的版本。其中 `bev_pool` 是唯一一个参考实现真正动过源码的（改法见上表）；其余 5 个在两侧树里都是上游原样。

这 7 个框架层 group 设计上不要求改动模型源码，能否生效完全取决于配置里对应 `enabled` 开关是否打开、monkeypatch 是否被触发。当前配置下它们不在 `optimization.yaml` 里，因此**一个都没生效**；要让它们生效，需要取消 `recipe.yaml:18-20` 的 `extends` 注释并重新生成配置。

需要说明的是，baseline 源码里**没有**任何 `@torch.compile` 装饰器（19 个挂点全在参考树里），所以 `maptrv2.compile` 这类 group 不会被"源码里写死的装饰器"绕过——早先版本的本节文字有过这个推断，已在 §3.8 更正。

#### 3.6.3 覆盖结论

```text
参考实现 19 个 compile 挂点
├── 10 个基线同名挂点       已接入（maptrv2.compile，enabled: true）；mode 与参考一致
├──  8 个新 helper 挂点    已接入（maptrv2.reference_boundaries，见 B6）；enabled: true
└──  1 个 assigner 挂点    已接入（maptrv2.assigner）；静态契约完整，但 assign 的
                             torch.compile 由 TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE 控制，
                             默认 "0"（不编译），且参考的 cudagraphs 选项未复刻

MapTRv2 专属优化（11 个 group 全部 enabled: true）
├── training / data / grid_mask / match_cost / pv_mask / efficientnet  生效
├── spconv_registry                                                  生效但当前冗余（wheel 已有 force=True）
├── bev_pool_fix                                                     生效但被 reference_boundaries 旁路
├── compile / reference_boundaries                                   生效
├── assigner                                                         静态化生效，assign 编译不生效
└── ddp_static_graph                                                 生效（需与 assigner 配对）

未接入的缺口
├── 数据集向量化第二梯队（_interpolate_line_points_vectorized、shift_fixed_num_sampled_points_v2 缓存）
├── GDK geometric_kernel_attn_cuda.cu 的 AT_DISPATCH value.scalar_type() 改动
└── lightop 算子替换（需要 import 层切换，replace/wrap 够不到）

mmdet3d 与部署层
├── 框架层通用优化：extends 链当前断开，7 个 group 全部未生效
└── 构建兼容补丁：build.py 已离线覆盖 mmcv_maximum_version / numba / ATen include /
                 __CUDACC__ / c++17 / GDK setup.py "if 1:"，不含上面那条 .cu 改动
```

因此当前的说法应该是：**12 个 `maptrv2.*` group 全部已接入并处于启用状态，但其中两项
（`spconv_registry`、`bev_pool_fix`）在当前运行时环境下分别表现为冗余和被旁路；
真正的缺口集中在数据集向量化第二梯队和一条 GDK 内核补丁上。**
7 个框架层 group 不是"接入了但没生效"，而是 `extends` 链根本没接线。

开启 `maptrv2.compile` / `maptrv2.reference_boundaries` 之后，编译耗时/重编译次数/精度/吞吐
的 A/B 仍然是必要步骤——默认 mode 与参考一致只说明配置对得上，不说明收益已在目标机型验证过。

### 3.7 未接入与部分接入项

本节把尚未接入、或虽然接入了但当前不产生效果的优化集中列出。每项都按“优化前 -> 优化后、
设计作用、接入方式/未接入原因”说明。作用是参考实现的设计目标；截至目前没有对应的上机性能
数据，不能视为已验证收益。

当前状态分为四类：

- **已接入并启用**：`maptrv2.compile` 的 10 个同名目标、`maptrv2.reference_boundaries` 的 8 个新增
  Helper 挂点（见 B6）、`maptrv2.assigner` 的静态契约（U4）、`maptrv2.ddp_static_graph`（U7）。
- **已接入但当前不生效**：`maptrv2.assigner` 内 `assign` 的 `torch.compile`（U4）、
  `maptrv2.spconv_registry`（U8）、`maptrv2.bev_pool_fix` 的主路径（U3）。
- **未接入**：数据集向量化第二梯队（U6）、GDK `.cu` 补丁（K3）。
- **已有等价替代**：`TransposeImage` 的输入布局处理（L2）。

#### 3.7.1 跨文件与源码改造

**U1. 10 个同名 `compile` 挂点（已接入并启用）**

优化前是普通 eager 函数，参考实现给它们加了编译装饰器：

```python
# 优化前：普通 Python 函数
def hot_method(...):
    ...

# 参考实现：
@torch.compile(mode="max-autotune-no-cudagraphs")
def hot_method(...):
    ...
```

- 接入状态：10 个目标通过 `_COMPILE_TARGETS` 声明并注册到 `maptrv2.compile`，配置中
  `enabled: true`。引擎在这些目标上安装 `compile_wrapper`；如果目标环境没有可用编译能力或
  设置了 `TURBO_PHYSAI_DISABLE_TORCH_COMPILE=1`，`runtime_condition` 会逐次调用回退原函数。

- 作用：减少 Python/调度开销，并让 Inductor 融合相邻 elementwise 算子。

- 与参考的差异：**没有差异**。参考实现 19 个挂点里有 18 个显式写了
  `mode="max-autotune-no-cudagraphs"`（只有 `MapTRAssigner.assign` 例外，见 U4），
  与本包 `compile.py:43` 的默认 mode 相同。早先版本的本节文字写成"参考用无参
  `@torch.compile()`、本包用 `max-autotune-no-cudagraphs`"，把 mode 差异当成了默认关闭的
  理由，这条已在 §3.8 更正。

- 仍需 A/B 的原因：mode 对得上只说明配置等价，不说明收益在目标机型上已经验证过。开启后
  要观察首次编译耗时、重编译次数、精度与吞吐。

**U2. 8 个新增 Helper 挂点（已接入并启用）**

8 项的具体抽取内容、替换目标和编译收益见 §B6；此处保留接入方式与启用状态说明。

参考实现把大函数中的局部逻辑抽成小函数，再单独编译：

```python
# 优化前：
def large_forward() {
    输入/布局/循环/张量计算全部内联，只能整体 eager
}

# 参考实现：
@torch.compile(mode="max-autotune-no-cudagraphs")
def helper(...):
    纯局部计算

def large_forward():
    整理 Python 侧输入 -> helper(...)
```

8 处包括 `down_sample`（B6.3）在内共 8 个 helper：

- `BaseTransform.matmul_1/2/3`（B6.1）：从 `get_geometry_v1` 拆出 view、矩阵乘、cat 和加减，缩小图边界并增加融合机会。
- `BaseTransform.extract_metas`（B6.2）：把 `img_metas` 的 Python 循环和 numpy/device 转换集中处理，减少主计算路径上的 host 工作。
- `LSSTransform.down_sample`（B6.3）：把分散在 `BaseTransform.bev_pool` 和 `LSSTransform.forward` 两处的 collapse-Z 与 downsample 合并成一个可编译 helper。
- `MapTRPerceptionTransformer.initialize_queries_and_bev`（B6.4）：把 query/bev 的 split、expand、reference 计算和 permute 从 transformer `forward` 中拆出。
- `MapTRv2Head.compute_decoder_predictions`、`prepare_transformer_inputs`（B6.5）：把逐 decoder level 的分类/回归/list 拼接，以及 query embedding、BEV query、attention mask 的配置态组装分别拆出。

**接入方式**：基线不存在这些新符号，`replace(target=...)` 无法把它们凭空声明成替换目标；`reference.py` 改为把 8 个 helper 实现为模块级纯函数，转而用 `wrap` 重写调用它们的 5 个基线原生方法（`maptrv2.reference_boundaries` group），让新实现内部调这些 helper，不新增模型源码符号（详见 §B6、§4.12）。

**当前状态**：配置里 `enabled: true`，与参考实现同样使用 `max-autotune-no-cudagraphs`，并共用
`_COMPILE_CONDITION`（`TURBO_PHYSAI_DISABLE_TORCH_COMPILE` 或环境无编译能力时逐次回退）。
和 U1 一样，仍需要在目标机型确认编译耗时、重编译次数、精度和吞吐。

**U3. `bev_pool` ROCm layout 修正（已接入）**

ROCm/HIP 的 `bev_pool` 返回 `[B, Z, H, W, C]`（channels-last），而 CUDA 版本返回
`[B, C, Z, H, W]`（NCHW）。原始 `BaseTransform.bev_pool` 直接在 dim=2 做 `unbind` 展开 Z
轴，ROCm 下实际切到的是 H 维，产出形状变成 `[B, 200, 400, C]`，接进 `LSSTransform` 的
`self.downsample`（`Conv2d(256,256,3,3)`）后 channel 数不匹配，抛出：

```
RuntimeError: Given groups=1, weight of size [256, 256, 3, 3],
expected input[4, 200, 256, 400] to have 256 channels, but got 200 channels instead
```

**接入方式**：新增 `bev_pool_fix.py`，完整复制 `BaseTransform.bev_pool` 逻辑并在
`bev_pool()` 调用之后插入 layout 检测与修正：

```python
x = _bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

# ROCm/HIP bev_pool 返回 [B, Z, H, W, C]；CUDA 返回 [B, C, Z, H, W]
if x.dim() == 5 and x.shape[-1] == self.C:
    x = x.permute(0, 4, 1, 2, 3).contiguous()

final = torch.cat(x.unbind(dim=2), 1)
```

`catalog.py` 注册 `maptrv2.bev_pool_fix` group，通过 `replace` 在模型实例化前把`BaseTransform.bev_pool` 换成修正版；`recipe.yaml` 和 `configs/optimization.yaml` 均设为`enabled: true`。`encoder.py` 全程未改动。

条件 `x.shape[-1] == self.C` 保证 CUDA 路径（`shape[-1] != self.C`）不会误触发，向前兼容。

`bev_pool_fix.py` 同时吸收了参考实现把 `torch.where(kept)[0]` + `index_select` 换成布尔掩码
`x = x[kept]` / `geom_feats = geom_feats[kept]` 的改动（`bev_pool_fix.py:47-48`），
所以它并不是"只加了一个 permute 判断"。

**当前配置下的实际归属**：参考实现的「把 collapse Z 移入新 `down_sample` 并统一编译」优化
由 `maptrv2.reference_boundaries` 的 `BaseTransform.forward` / `LSSTransform.forward` 包装承担
（见 B6.3），而 `reference.py` 的 `base_transform_forward` 直接调 `_bev_pool_5d(self, features, geometry)`，
**不再经过 `self.bev_pool`**。因此当两个 group 都开启（当前就是）时，本节的 `bev_pool_fix`
在主训练路径上不会被调用——它是 `reference_boundaries` 关闭时的兜底，不是叠加生效的第二层。
两者是同一段逻辑的两个实现，不要理解为互补。

**启动方式**（须先设置 PYTHONPATH，再通过 TurboPhysAI 包裹启动）：

```bash
export PYTHONPATH=/workspace/TurboPhysAI:$PYTHONPATH

cd /workspace/model/MapTrv2
turbo-physai run \
  --optimization-config /workspace/TurboPhysAI/configs/optimization.yaml \
  --runtime-config /workspace/TurboPhysAI/turbo_physai/optimizations/models/maptrv2_optimization/configs/runtime.yaml \
  python tools/train.py projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
  --work-dir work_dirs/maptrv2_optimized
```

`PYTHONPATH` 优先于系统 `/usr/local/lib/python3.10/dist-packages/turbo_physai/`（该版本缺少
`maptrv2` model 支持）；`pip install -e` 在无卡构建环境因找不到 `torch` 而失败，因此用
`PYTHONPATH` 替代。

**U4. Assigner 静态化（已接入；`assign` 的编译默认不开）**

```text
优化前：
gt_bboxes: (N, 4)，N 每步变化
assign()
    ├── num_gts == 0 动态早退
    ├── Python cost 计算
    └── SciPy Hungarian

参考实现：
pad_to_static_list()
    └── (padded(200, 4), valid_mask(200), num_gts)
assign()  [@torch.compile]
    └── hungarian_match()  [@torch._dynamo.disable]
```

作用：固定 GT shape，减少 Dynamo guard 失效和重编译；把 CPU-only SciPy 求解隔离在图外。

**接入方式**：契约同时跨越 `loss`、`get_targets`、`_get_target_single`、`assign`、
`sampler.sample` 和 `hungarian_match`，只替换 assigner 会直接 shape/语义不匹配，所以
`maptrv2.assigner` 用 3 个 `replace` 目标一次性覆盖整条链：

| 目标 | 包内实现（`assigner.py`） |
| --- | --- |
| `MapTRAssigner.assign` | `assign`（`:274`）→ `_assign_impl` / `_assign_no_grad` |
| `MapTRv2Head.loss` | `static_loss`（`:457`）→ `_static_loss_impl`（`:331`），内部用 `pad_to_static_list` 把 GT 补到固定长度 |
| `MapTRv2Head._get_target_single` | `get_target_single`（`:97`）+ `get_label_result`（`:47`） |

`_hungarian_match_impl`（`:131`）包住 SciPy 的 `linear_sum_assignment`，通过
`torch._dynamo.disable`（`:159-165`）把 CPU-only 求解隔离在图外——与参考实现的
`hungarian_match` 做法一致。

**与参考的差异**：参考对 `assign` 用的是
`@torch.compile(options={"triton.cudagraphs": True, "triton.cudagraph_trees": False})`；
本包由 `TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE` 控制，**默认 `"0"` 即不编译**，开启时用的是
`max-autotune-no-cudagraphs`，没有复刻那组 cudagraphs 选项。也就是说：静态化收益当前生效，
`assign` 的编译收益当前不生效。

**U5. DataLoader `spawn` 下 `dict_keys` pickle 崩溃（已接入）**

`torchrun` 拉起 8 卡分布式训练后，每个 rank 的 `DataLoader` 在创建 worker 子进程时抛出：

```
File "/usr/lib/python3.10/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
TypeError: cannot pickle 'dict_keys' object
```

**根因**：`tools/train.py` 里的 `torch.multiprocessing.set_start_method('fork')` 只在单进程直接
`python tools/train.py` 时生效（见 B2 已知限制）。`torchrun` 已经把每个 rank 起成独立进程，
该全局调用在此时不再改变 worker 的启动方式，`DataLoader` 退回 PyTorch 默认的 `spawn`。
`spawn` 需要把整个 dataset 对象 pickle 后传给 worker，而 nuScenes 的
`DetectionConfig`（`self.eval_detection_configs`）内部某个字段以 `dict.keys()` 视图形式保存，
`dict_keys` 不可 pickle，worker 启动阶段直接崩溃。单卡或 `workers_per_gpu=0` 时不触发，因为
根本没有走到子进程创建。

**接入方式**：`data.py` 新增 `_dataloader_multiprocessing_context()`，在
`build_dataloader()` 创建 `DataLoader` 的调用点显式传入 `multiprocessing_context`，而不是依赖
`custom_train_detector` 里已经失效的全局 `set_start_method`：

```python
def _dataloader_multiprocessing_context():
    if not _env_flag("TURBO_PHYSAI_FORK_START_METHOD", True):
        return None
    start_method = os.getenv(
        "TURBO_PHYSAI_DATALOADER_START_METHOD", "fork"
    ).strip().lower()
    if start_method in {"", "default", "none"}:
        return None
    return torch.multiprocessing.get_context(start_method)

...
if num_workers > 0:
    context = _dataloader_multiprocessing_context()
    if context is not None:
        kwargs["multiprocessing_context"] = context
```

默认选 `fork`：worker 直接复用父进程已 import 好的 mmcv/mmdet3d 与已构造好的 dataset 对象，
不走 pickle，`dict_keys` 也就不需要被序列化。保留三档可配置：

| 环境变量 | 行为 |
| --- | --- |
| `TURBO_PHYSAI_FORK_START_METHOD=0` | 关闭本项覆盖，`DataLoader` 走 PyTorch 默认行为 |
| `TURBO_PHYSAI_DATALOADER_START_METHOD=default` | 同上，显式声明恢复原生行为 |
| `TURBO_PHYSAI_DATALOADER_START_METHOD=spawn` | 显式改用 `spawn`（需要先修掉 dataset 里的 `dict_keys`） |

`runtime.yaml` 中 `TURBO_PHYSAI_DATALOADER_START_METHOD: fork` 只是显式声明默认值；即使该变量
缺失，`data.py` 的默认值同样是 `fork`。

**归属**：这是 `maptrv2.data`（B3）group 内的追加修正，不是新 group；`catalog.py` 的
`replace` 目标未变，仍是 `projects.mmdet3d_plugin.datasets.builder.build_dataloader`。回归测试
见 `test/optimizations/test_maptrv2_data.py`（§3.4）。

**U6. 数据集向量化第二梯队（未接入）**

`maptrv2.pv_mask` 覆盖的是 PV mask 相关的三个方法，但参考实现对
`nuscenes_offlinemap_dataset.py` 的向量化改造还有另外两处，本包没有接：

```text
优化前：逐点 shapely 插值（每个 instance 一次 Python 循环）
优化后：
  _interpolate_line_points_vectorized(line, distances)   # shapely 2.0 批量
  LiDARInstanceLines.shift_fixed_num_sampled_points_v2
      -> @cached_property，内部走 _shift_fixed_num_sampled_points_v2_vectorized
         （批量插值 + unique_instance_ids/duplicate_meta 去重 + 预算 shift 索引矩阵）
```

| 参考实现位置（`/workspace/MapTRv2/.../nuscenes_offlinemap_dataset.py`） | 内容 | 本包 |
| --- | --- | --- |
| `:38-48` | `_interpolate_line_points_legacy` / `_interpolate_line_points_vectorized`，`:47` 按 shapely 版本二选一 | 无 |
| `:184, :204, :258, :368, :379, :557, :566` | 7 个调用点从逐点循环换成向量化版本 | 无 |
| `:347` | 原 `shift_fixed_num_sampled_points_v2` 降级为 `_shift_fixed_num_sampled_points_v2_legacy` | 无 |
| `:406-532` | 新增 `_shift_fixed_num_sampled_points_v2_vectorized()` | 无 |
| `:523` + `:129` | `shift_fixed_num_sampled_points_v2` 变成 `@cached_property`，并加缓存字段 | 无 |

基线对应的只有 `nuscenes_offlinemap_dataset.py:321` 的原始循环实现。未接入原因：这两处的
受益面在 dataloader worker 侧，`pv_mask.py` 目前只承载了 PV mask 的三条路径；要接入需要先
确认目标环境的 shapely 版本（向量化分支依赖 shapely 2.0 的 `line_interpolate_point` /
`get_coordinates`），再按同样方式增加 `replace` 目标。

**U7. DDP 静态图（已接入）**

```text
优化前（基线 maptrv2_nusc_r50_24ep.py:337）
    find_unused_parameters=True          # 每步遍历参数判断是否需要规约

参考实现（:341-342）
    find_unused_parameters=False
    ddp_static_graph=True                # 交给 MMDistributedDataParallel 静态图优化
bevformer/apis/mmdet_train.py:113,121,128  增加 ddp_static_graph 的读取与 static_graph 透传
```

作用：关掉 `find_unused_parameters` 的每步参数扫描，并按静态图假设做规约规划，
减少 DDP 的反向通信开销。

**为什么不能只改 config**：基线 `mmdet_train.py` 里**没有** `cfg.get('ddp_static_graph', ...)`
这行，也没有 `static_graph=` 传参。只把字段塞进 `cfg` 会被静默忽略——参考实现是"config 字段 +
训练 API 透传"两处同时改才成立的。

**接入方式**：`maptrv2.ddp_static_graph` 用 `wrap` 挂在 DDP 构造点上，
在构造时注入 kwargs，两端都不改源码：

```python
# catalog.py
_DDP_API = "mmcv.parallel.distributed.MMDistributedDataParallel.__init__"
DDP_STATIC_GRAPH = group(
    "maptrv2.ddp_static_graph",
    wrap(target=_DDP_API, replacement="...maptrv2_optimization.ddp.ddp_constructor_wrapper"),
)

# ddp.py
def wrapped(self, *args, **kwargs):
    kwargs.setdefault("static_graph", True)
    kwargs["find_unused_parameters"] = False
    return original(self, *args, **kwargs)
```

选择这个挂点有两层原因：

| | 原因 |
| --- | --- |
| 为什么是 `MMDistributedDataParallel` | 基线 `mmdet_train.py:12` 是 `from mmcv.parallel import MMDistributedDataParallel`，类定义在 `mmcv/parallel/distributed.py:13`，且**没有重写 `__init__`**，直接继承 torch 的 |
| 为什么这样安全 | 引擎的 `set_attribute` 是 `setattr(MMDistributedDataParallel, "__init__", wrapper)`，只在**子类上新建属性遮蔽继承**；`torch.nn.parallel.distributed.DistributedDataParallel.__init__` 不受影响（`test/optimizations/test_maptrv2_ddp.py` 里有断言） |

一处挂点同时覆盖 `mmdet_train.py:75`（训练模型）和 `:81`（`eval_model`）两个构造点。

**开关与默认值**（`ddp.py`，`options` > env > 默认值）：

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `TURBO_PHYSAI_DDP_STATIC_GRAPH` | `"1"` | 设 `0` 则不注入 `static_graph` |
| `TURBO_PHYSAI_DDP_FIND_UNUSED_PARAMETERS` | `"1"` | 设 `0` 则保留基线 config 的 `find_unused_parameters`（当前 `True`） |

默认值对齐参考实现（两个都生效），`runtime.yaml` 里显式写出。

**前置条件**：`static_graph=True` 只在**第一轮迭代**推断"哪些参数不被使用"，之后一直沿用。
若某参数首轮未被使用、之后被使用，其梯度**永远不会被规约**——静默的正确性错误。所以它必须与
"计算图不随数据变化"配套，也就是必须和 `maptrv2.assigner`（`pad_to_static_list` 把 GT 固定到
200）一起开。关闭 assigner 时应同步关闭本 group。

已核实基线侧没有会让参数使用随数据变化的早退：`loss_single`（`maptrv2_head.py:621`）里
cls/bbox/pts 三项损失无条件计算，`cls_avg_factor` 与 `num_total_pos` 都有 `max(..., 1)` /
`clamp(min=1)` 兜底。

torch 侧对 `static_graph=True` + `find_unused_parameters=True` 只发 warning 不报错，所以这里
显式把 `find_unused_parameters` 也压成 `False`，与参考一致，同时避免日志噪声。

**U8. `maptrv2.spconv_registry` 在当前环境是空操作（已接入但冗余）**

`maptrv2.spconv_registry` 的目标是 `mmdet3d.ops.spconv.conv`，它解析到 **dist-packages 里安装的
mmdet3d**，而这个 wheel 由 `/workspace/MapTRv2/mmdetection3d/dist/` 构建（见 §3.6.2 前提），
其 `conv.py` 的 10 处 `@CONV_LAYERS.register_module(force=True)` 已经就位。因此该 group 在当前
环境下不会改变任何注册行为。

它并非无用：在"镜像预装了未打补丁的 mmdet3d"这类环境里，它正是让仓库自带的 SparseConv
覆盖已注册项的开关。判断它是否生效，要看运行时 `import mmdet3d` 来自哪个构建。

#### 3.7.2 Kernel、公共层与镜像

**K1. 强制扩展构建（已接入，走离线构建补丁）**

```python
# 优化前：torch.cuda.is_available() 为 False 时构建中止
if torch.cuda.is_available() and CUDA_HOME is not None:
# 优化后：
if 1:
# if torch.cuda.is_available() and CUDA_HOME is not None:
```

作用：绕过构建阶段的设备可用性判断，避免在无卡镜像构建环境中直接进入 `else`；ROCm
PyTorch 会按自身的扩展构建链处理该 CUDAExtension。

需要澄清：权威版本没有新增 `.hip` 文件，仍只 glob `*.cu`。`if 1:` 也不会让纯 CPU 环境
自动获得可运行算子，后续仍需要 CUDA/HIP 编译工具链。

**接入方式**：`build.py:163 force_geometric_kernel_extension()` 对
`projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn/setup.py` 做同一个替换，
由 `apply_build_compatibility()`（`build.py:186`）统一调度，通过
`python -m turbo_physai.optimizations.models.maptrv2_optimization.build` 显式执行。

**为什么不做成 group**：这是扩展构建逻辑，不是 Python 运行时函数，无法用 recipe 的
`target`/`replace` 表达；它作用于构建阶段，与运行时 monkeypatch 生命周期不重叠，所以放在
`build.py` 这个离线补丁工具里。

**K2. SparseConv 注册覆盖（已接入；当前环境冗余）**

```python
# 优化前
@CONV_LAYERS.register_module()
class SparseConv2d(...): ...

# 参考实现
@CONV_LAYERS.register_module(force=True)
class SparseConv2d(...): ...
```

参考实现共修改 11 个 SparseConv 类型。作用是避免容器镜像同时加载多套 mmdet3d/spconv 时出现
registry 冲突。

**接入方式**：`maptrv2.spconv_registry` 以 `registry_override` 覆盖
`mmdet3d.ops.spconv.conv` 的 10 个 `SparseConv*`/`SubMConv*` 名称，配置中 `enabled: true`。

**当前环境下的实际效果：无**。运行时 `import mmdet3d` 解析到 dist-packages 的 0.17.2，
而该 wheel 由参考树构建、`conv.py` 里 10 处已经是 `register_module(force=True)`
（见 §3.8.1），所以这个 group 不会改变任何注册结果。它的价值在于镜像里预装的是**未打补丁**
的 mmdet3d 时——那种环境下它是必需的。

**归属说明**：它保护的是通用 `mmdet3d` 能力，却声明在 MapTRv2 专属 catalog 里。如果要迁到
`common/mmdet3d` 层，前提是先把 `extends: common.hcu.base` 接通（§2.6）——在继承链断开的
状态下，放进通用层不会被这个 recipe 继承到。

**K3. 构建兼容（大部分已接入，一条缺口）**

```diff
- 使用旧版 THC/THC.h
+ 使用 ATen/cuda/CUDAContext.h
- 判断 __CUDA_ARCH__
+ 判断 __CUDACC__
- C++14
+ C++17
- 严格锁定旧 numba/numpy/mmcv 版本
+ 放宽为当前镜像可用版本
```

作用：让扩展能在当前 PyTorch、ROCm/HCU 和 Python 环境上成功构建。

**已由 `build.py` 离线覆盖的部分**（`_build_compatibility_patches()` `:57` /
`_point_op_patches()` `:142`）：

| 补丁 | `build.py` 位置 |
| --- | --- |
| `mmcv_maximum_version = '1.4.0'` → `'1.6.2'` | `:62-63` |
| `numba.errors` → `numba.core.errors` | `:72-73` |
| `#ifdef __CUDA_ARCH__` → `#ifdef __CUDACC__`（及两处版本判断） | `:109-111` |
| point ops 补 `#include <ATen/cuda/CUDAContext.h>` | `:150-154` |
| 放开 numba 版本锁定 | `:118` |
| `-std=c++14` → `-std=c++17` | `:127` |
| GDK `setup.py` 的 `if 1:` | `:163`（见 K1） |

**缺口**：参考实现还改了 GDK 内核源码
`projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn/src/geometric_kernel_attn_cuda.cu:57,127`：

```diff
- AT_DISPATCH_FLOATING_TYPES(value.type(), "multiscale_kernel_attn_forward_cuda", ...)
+ AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "multiscale_kernel_attn_forward_cuda", ...)
```

`build.py` 里没有这条替换。它在已有 `.so` 的环境下不影响运行，只有在重新编译 GDK 扩展时
才会暴露；属于"重建扩展时需要补上"的构建项，不是运行期缺口。

**为什么不做成 group**：这些是构建补丁，不是运行时优化；recipe 的 Python 替换无法处理
C++/CUDA 编译条件，所以统一由 `build.py` 这个离线工具承载。

**K4. lightop DCU Deformable Attention 算子替换（未接入）**

参考实现在 `multi_scale_deformable_attn_function.py` 顶部以 try/except 优先加载 Hygon DCU 专用的 `lightop` 库：

```python
try:
    from lightop import op as ext_module
except ImportError:
    from mmcv import _ext as ext_module
```

作用：`lightop` 针对 DCU 的 MS-Deformable-Attention 前反向传播做了针对性优化，与 mmcv 通用实现相比在 Hygon 硬件上有明显性能差距。

未接入原因：import 时已经决定加载哪个库；TurboPhysAI 的 replace/wrap 在模块导入之后才介入，无法在 import 层切换后端。`mmcv.msda` Group 替换的是 mmcv `_ext` 的符号，lightop 安装后会绕过该路径。若要通过 TurboPhysAI 管理，需要把 lightop 的前反向实现封装成独立 operator 并注册到 `turbo_physai/operators/`。

#### 3.7.3 低收益项与替代方案

**L1. LineString 拷贝消除（已接入，随 `maptrv2.pv_mask` 一起）**

```diff
- vectors.append((LineString(np.array(instance)), label))
+ vectors.append((LineString(instance), label))
```

作用：省掉一次顶点数组分配和拷贝。

**接入状态**：`maptrv2.pv_mask` 的三个 `replace` 目标里包含 `gen_vectorized_samples`，
`pv_mask.py` 中该函数已经是 `LineString(instance)`，所以这条改动已经随该 group 一起落地，
不需要为它单独设 group（对照 §3.6.1 B11 一行）。它本身收益很小，价值在于消掉了每帧一次
无用拷贝。

**L2. TransposeImage（已有等价替代）**

```python
# 参考实现新增 pipeline
class TransposeImage:
    def __call__(self, results):
        results['img'] = results['img'].contiguous(
            memory_format=torch.channels_last
        )
        return results
```

作用：让输入从 NCHW 提前转成 channels-last，使 Conv2d 从输入开始走 NHWC。

这不是未接入缺口：参考实现定义了该类，但 config 最终没有引用；TurboPhysAI 用 backbone
`forward_pre_hook` 在运行时把输入转成 channels-last，作用等价。

**统一结论**

```text
已接入并生效
├── 10 个同名 compile 挂点（maptrv2.compile）
├── 8 个新增 Helper 挂点（maptrv2.reference_boundaries，见 B6）
├── assigner 静态化（maptrv2.assigner；assign 的编译默认不开）
├── DDP 静态图（maptrv2.ddp_static_graph，U7）
├── CUDAExtension 强制构建（build.py，非 group）
├── mmdet3d / point op 构建兼容补丁（build.py）
├── SparseConv registry override（maptrv2.spconv_registry，当前环境冗余）
└── LineString 单次拷贝消除（随 maptrv2.pv_mask）

已接入但当前不生效
├── maptrv2.assigner 内 assign 的 torch.compile（TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE 默认 0）
├── maptrv2.bev_pool_fix 的主路径（被 reference_boundaries 旁路）
└── 7 个框架层 group（extends 链未接线）

未接入
├── 数据集向量化第二梯队（U6）
├── GDK geometric_kernel_attn_cuda.cu 的 scalar_type 改动（K3）
└── lightop 算子替换（需封装到 turbo_physai/operators/）

收益小或已有安全替代
└── TransposeImage（L2）
```

因此当前最需要继续处理的是 **U6 数据集向量化第二梯队**；
`lightop` 算子替换需要先封装到 `turbo_physai/operators/` 才能进入配置管理；
GDK 的 `.cu` 补丁只在重建扩展时需要；`spconv_registry` 与 `bev_pool_fix` 不是缺口，
但在当前环境下分别是冗余和被旁路，启用配置时要知道它们没有叠加效果。

10 个同名 `compile` 挂点、8 个新增 Helper 挂点和 `maptrv2.ddp_static_graph` 已经由配置启用，
不需要再改接入代码，剩下的是在目标机型完成 A/B 观察。`ddp_static_graph` 上机时要注意两点：
必须与 `maptrv2.assigner` 同时开启（U7 的前置条件），以及**验证方式是看 loss 曲线而不是看是否报错**
——静态图接错的表现是某些参数静默不更新。

### 3.8 三方差分审计

§3.6/§3.7 的结论是在看不到本地参考源码、且配置状态与现在不同的条件下写的，本节用三份
可直接打开的文件重新核对一遍，也是 §2.6、§3.6.2、U1、U2 那几处更正的依据。

三条线：

| 代号 | 路径 | 说明 |
| --- | --- | --- |
| **A** | `/workspace/model/MapTrv2` | 未优化基线，PhysAI 实际包裹的代码 |
| **B** | `.../models/maptrv2_optimization` | 非侵入接入层（12 个 `maptrv2.*` group + `build.py`） |
| **C** | `/workspace/MapTRv2` | 侵入式优化参考实现 |

真实启动命令（`/workspace/run_maptrv2.sh:2,17-26`）：`cd /workspace/model/MapTrv2` 后
`turbo-physai run --optimization-config .../maptrv2_optimization/configs/optimization.yaml
--runtime-config .../maptrv2_optimization/configs/runtime.yaml --disable-numa
torchrun --nproc-per-node=8 --master-port=6005 --no-python bash -c "$NUMA_SCRIPT" _ ...`。
`--disable-numa` 关掉 `runtime.yaml` 的 `process.numa`，NUMA 绑定改由脚本内联的
`numactl --cpunodebind/--membind` 完成。

#### 3.8.1 前提：运行时 mmdet3d 不是仓库树

`import mmdet3d` 解析到 `/usr/local/lib/python3.10/dist-packages/mmdet3d`（0.17.2），
其 `direct_url.json` 记录的是
`file:///workspace/MapTRv2/mmdetection3d/dist/mmdet3d-0.17.2-...whl`——**wheel 由参考树 C 构建**。
两侧自带树的状态对比证实了这一点：

| | 安装版（实际运行） | A 自带树 | C 自带树 |
| --- | --- | --- | --- |
| `ops/bev_pool/bev_pool.py:96` | `#x = x.permute(0, 4, 1, 2, 3).contiguous()` | `x = x.permute(...)` | `#x = x.permute(...)` |
| `ops/spconv/conv.py` 的 `force=True` 处数 | 10 | 0 | 10 |

两个直接推论：

1. A 与 C **共用同一个已被参考改动过的 mmdet3d**，所以"未优化基线"在 mmdet3d 这一层
   并不干净——这正是 `maptrv2.bev_pool_fix` 在基线上必需的原因（`bev_pool` 已经不再自己
   permute，调用方必须处理 `[B,Z,H,W,C]`）。
2. `maptrv2.spconv_registry` 在当前环境是空操作（见 U8）。

#### 3.8.2 Q1：有无未接入的优化项

| 参考实现（C）的改动 | B 的接入情况 |
| --- | --- |
| 19 个 `@torch.compile` 挂点（18 个 `max-autotune-no-cudagraphs`） | 10 个走 `maptrv2.compile`，8 个 helper 走 `maptrv2.reference_boundaries`，第 19 个（`MapTRAssigner.assign`）在 `maptrv2.assigner` 内 |
| `torch._dynamo.disable`（`hungarian_match`、`GridMask.forward`） | 均已接入（`assigner.py:159-165`、`grid_mask.py`） |
| `set_float32_matmul_precision('high')`、`cudnn.benchmark`、`fork` | 均已接入（`maptrv2.training`） |
| `cdist(p=1)` → 广播减法 | 已接入（`maptrv2.match_cost`） |
| PV mask 向量化 + 合并 affine + `LineString` 拷贝消除 | 已接入（`maptrv2.pv_mask`） |
| **数据集向量化第二梯队**（`_interpolate_line_points_vectorized`、`shift_fixed_num_sampled_points_v2` 缓存） | **未接入**（U6） |
| DDP `ddp_static_graph=True` + `find_unused_parameters=False` | 已接入（`maptrv2.ddp_static_graph`，U7）；基线 `mmdet_train.py` 没有该字段，所以改在 DDP 构造点注入 |
| **GDK `.cu` 的 `value.scalar_type()`** | **未接入**（K3） |
| `bev_pool` 三文件协同改造（mmdet3d 去 permute / `bev_pool` return 5D / 编译版 `down_sample`） | 已接入（`maptrv2.reference_boundaries` + `maptrv2.bev_pool_fix`），但两者重叠，见 3.8.3 |
| `SparseConv` / `EfficientNet` `force=True` | 已接入（`spconv_registry` / `efficientnet`） |
| 构建补丁（`c++17`、`__CUDA_ARCH__`、`mmcv_maximum_version`、numba、ATen include、`if 1:`） | 已由 `build.py` 离线覆盖 |
| `lightop` MSDeformAttn 后端切换 | 未接入；import 层决策，`replace`/`wrap` 够不到（K4） |
| `ProfilerHook`、`test.py` DDP 兼容补丁、`map_ann_file` 硬编码、`evaluation interval=2→6` | 不属于模型优化包，不视为缺口 |

#### 3.8.3 Q2：接入项是否正确接入

| 项 | 结论 | 依据 |
| --- | --- | --- |
| 10 个同名 compile 挂点 | 正确 | 目标全部可解析；`compile.py:43` 默认 mode 与参考的 18 处一致。`LSSTransform.get_cam_feats` 在基线上已有 `@force_fp32()`，本包 wrap 的是装饰后的对象，等价于参考把 `@torch.compile` 叠在 `@force_fp32` 之上 |
| 8 个 helper | 正确 | `reference_boundaries` 用 5 个 wrap 目标重建调用边界；`force_fp32` 只加在基线上本来就有 `@force_fp32` 的 3 处（`get_geometry_v1`/`BaseTransform.forward`/`head.forward`），`transformer.forward` 与 `LSSTransform.forward` 不加——与参考一致 |
| `training` | 等价 | cudnn / matmul 两项与参考硬编码等价；channels-last 由"包 DDP 后转换"变为"转换后交给 `original`"，`Module.cuda()` 保留 memory_format |
| `data` | 等价且更稳 | `pin_memory=True` 一致；额外加了 `multiprocessing_context`（参考没有），修掉 spawn 下 `dict_keys` 不可 pickle 的崩溃 |
| `grid_mask` | 等价且更稳 | 用 `x.device` 替代参考硬编码的 `.cuda()` |
| `match_cost` | 一致 | 与参考 `map_loss.py:527` 的广播减法同形 |
| `pv_mask` | 部分覆盖 | 3 个目标与参考对应实现一致；数据集第二梯队未覆盖（Q1） |
| `bev_pool_fix` | 正确但重叠 | 含参考的 `x[kept]` 布尔掩码替换 + ROCm layout 修正；**与 `reference_boundaries` 是同一段逻辑的两个实现** |
| `assigner` | 正确 | `pad_to_static_list`/`hungarian_match`/`get_target_single` 均在包内实现；差异只在 `assign` 的编译选项 |
| `spconv_registry` / `efficientnet` | 正确但前者冗余 | 见 U8 |

**重叠的具体后果**：`reference.py:418-447` 的 `base_transform_forward` 直接调用
`_bev_pool_5d(self, features, geometry)`，**不经过 `self.bev_pool`**。而 `bev_pool_fix` 的
作用点正是 `BaseTransform.bev_pool`。两个 group 都开启时，主训练路径走的是
`_bev_pool_5d`，`bev_pool_fix` 不会被调用。它不是"第二层保险"，而是 `reference_boundaries`
关闭时的替代方案。

#### 3.8.4 Q3：当前配置下是否都能生效

配置事实（`configs/optimization.yaml` / `recipe.yaml` 核对）：

- 12 个 `maptrv2.*` group **全部 `enabled: true`**，没有任何 `enabled: false`。
- `recipe.yaml:18-20` 的 `extends: - common.hcu.base` **整段被注释**；`optimization.yaml`
  无 `extends` 键，`optimization_modules` 只有 `maptrv2_optimization.catalog` 一行。
- `runtime.yaml` 未设置 `TURBO_PHYSAI_DISABLE_TORCH_COMPILE` /
  `TURBO_PHYSAI_DISABLE_ASSIGNER_STATIC`，两者的运行时开关都在"允许"侧。

| Group | 配置 | 实际生效 | 说明 |
| --- | --- | --- | --- |
| `training` / `data` / `grid_mask` / `match_cost` / `pv_mask` | enabled | 生效 | |
| `efficientnet` | enabled | 生效 | |
| `compile` | enabled，env 未禁 | 生效 | 与参考同 mode |
| `reference_boundaries` | enabled，env 未禁 | 生效 | |
| `ddp_static_graph` | enabled，env 默认 `"1"` | 生效 | 需与 `assigner` 配对，见 U7 |
| `assigner` | enabled | 静态化生效，`assign` 编译**不生效** | `TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE` 默认 `"0"` |
| `spconv_registry` | enabled | **冗余空操作** | 运行时 wheel 已带 `force=True`（3.8.1） |
| `bev_pool_fix` | enabled | **被旁路** | `reference_boundaries` 抢先接管 `BaseTransform.forward`（3.8.3） |
| `mmcv.msda` + 6 个 `mmdet3d.*` | **不在配置里** | **不生效** | `extends` 链未接线（§2.6） |

#### 3.8.5 结论

```text
A（未优化基线）
  └── 自带 mmdet3d 树不被导入；运行时用的是 C 构建的 wheel
        └── bev_pool 已 return 5D，所以 B 必须有 bev_pool_fix 才不崩

B（PhysAI 接入层）
  ├── 12 个 maptrv2.* group 全部 enabled: true
  ├── 与 C 对齐的：19 个编译边界（10 + 8 + assigner）、精度/后端开关、
  │   cdist 替换、PV mask 向量化、bev_pool 契约、注册表覆盖、构建补丁、
  │   DDP 静态图
  ├── 与 C 有差异的：assign 的编译选项（默认不开）
  ├── 与 C 缺口：数据集向量化第二梯队、GDK .cu
  └── 自身重叠/冗余：bev_pool_fix 被 reference_boundaries 旁路、
      spconv_registry 在已打补丁的 wheel 上空转

C（侵入式参考）
  └── 另有 7 个框架层公共优化（mmcv.msda + 6 个 mmdet3d.*）
        但 B 的 extends 链未接线，这 7 项当前一个都没进配置
```

一句话：**接入的项在语义上都对得上参考实现，主要缺口只剩数据集向量化第二梯队和 GDK 的 `.cu` 补丁；
需要留意的是三项"接了但当前不产生效果"——`bev_pool_fix`（被旁路）、`spconv_registry`（冗余）、
`assign` 的编译（默认关闭）——以及整条 `extends` 框架层继承链没有接线。**

---

## 4. 优化项实现解析

每一项都用 MapTRv2 里对应的真实 diff 作起点，讲清楚"这行代码到底改了什么、GPU 侧发生了什么、为什么快"。

---

### 4.1 channels-last 与 NHWC

**实际代码**（`tools/train.py`）：

```python
model = build_model(cfg.model, train_cfg=..., test_cfg=...)
# 改变的是权重在内存中的排列方式（stride）
model.img_backbone = model.img_backbone.to(memory_format=torch.channels_last)
model.init_weights()
```

以及 pipeline 侧（`transform_3d.py` 里的 `TransposeImage`）：

```python
def convert(x):
    if isinstance(x, torch.Tensor):
        return x.contiguous(memory_format=torch.channels_last)
    return x
```

**tensor 在内存里的实际排布**

一张 `[N=1, C=3, H=2, W=4]` 的图，NCHW 下内存是先 C0 的一整张，再 C1，再 C2：

```
NCHW (channels_first) —— stride = (24, 8, 4, 1)
地址 →
[ C0H0W0 C0H0W1 C0H0W2 C0H0W3 | C0H1W0 C0H1W1 C0H1W2 C0H1W3 ]
[ C1H0W0 C1H0W1 C1H0W2 C1H0W3 | C1H1W0 C1H1W1 C1H1W2 C1H1W3 ]
[ C2H0W0 C2H0W1 C2H0W2 C2H0W3 | C2H1W0 C2H1W1 C2H1W2 C2H1W3 ]
```

`.to(memory_format=torch.channels_last)` 后 stride 变成 `(24, 1, 12, 3)`，物理排布变成"同一个 (H,W) 位置的 C0 C1 C2 挨在一起"：

```
NHWC (channels_last) —— stride = (24, 1, 12, 3)
地址 →
[ C0H0W0 C1H0W0 C2H0W0 | C0H0W1 C1H0W1 C2H0W1 | C0H0W2 C1H0W2 C2H0W2 | C0H0W3 C1H0W3 C2H0W3 ]
[ C0H1W0 C1H1W0 C2H1W0 | C0H1W1 C1H1W1 C2H1W1 | ... ]
```

**为什么快**

一个 3×3 conv 的输出点是 `∑_{c=0..Cin} ∑_{kh,kw} W[cout,c,kh,kw] * X[c, h+kh, w+kw]`。GPU kernel 里最内层循环通常是 **在 C 方向做 MAC 累加**。

```
NCHW:  访问同一 (h,w) 的不同 C，地址步长 = H*W（跨很多 byte）
       ┌───► C0 位置 A
       │
       │      ×××××××× H*W 距离 ××××××××
       │
       ▼    C1 位置 A+H*W  → cache line miss 概率高

NHWC:  访问同一 (h,w) 的不同 C，地址步长 = 1（相邻）
       C0 C1 C2 ... 连续 → 一次 128-bit load 拿到 8 个 fp16
```

Tensor Core / Matrix Core 一次要吃一个 `[M×K] × [K×N]` 的 tile（比如 K=16 个 fp16），NHWC 下 K 维就是 C 维、天然连续；NCHW 下 kernel 得先在 shared memory 里做一次隐式转置，浪费带宽。

**联动三处的原因**

如果只把 backbone 转 NHWC，DataLoader 送进来的 tensor 还是 NCHW，第一层 conv 前 cuDNN 会插一个隐式 `nchw_to_nhwc` kernel；反过来若只转输入，后面 permute 又转回去。所以 MapTRv2 三处联动（模型 + DDP 前一次 + pipeline 里）才能真正消掉这些转置。

在 profiler trace 里应能看到：

```
优化前:  Conv2d ─┐
                 ├─ nchw_to_nhwc  ← 每层前面都有
                 │
                 └─ implicit_gemm
优化后:  Conv2d ─── implicit_gemm  ← 直接进 kernel
```

---

### 4.2 cuDNN 与 fork 启动

**实际代码**（`tools/train.py`）：

```python
torch.backends.cudnn.benchmark = True        # 启用自动寻找最优卷积算法
torch.backends.cudnn.deterministic = False   # 允许非确定性算法提升速度
...
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
```

**`cudnn.benchmark` 干了什么**

cuDNN/MIOpen 对每个 conv 配置有多个候选算法（`IMPLICIT_GEMM`, `WINOGRAD`, `FFT`, `DIRECT`, ...）。默认策略是启发式选，可能不是最快的。开 benchmark 后：

```
第一次遇到 shape=(N,3,H,W), weight=(64,3,7,7), stride=2, padding=3：
   ┌──────────────────────────────────────┐
   │ 试 IMPLICIT_GEMM     → 2.1ms         │
   │ 试 WINOGRAD_NONFUSED → 1.4ms   ★     │
   │ 试 FFT               → 3.8ms         │
   └──────────────────────────────────────┘
                    ↓
   缓存 (input_shape, weight_shape, ...) → WINOGRAD_NONFUSED
第二次起：直接用 WINOGRAD_NONFUSED，跳过 heuristic。
```

BEV 检测里图像尺寸是固定的（如 nuScenes 常见 `1600×900` 或统一 resize），命中缓存率 100%，一般能拿 10–30% conv 加速。**代价**：训练前几个 step 会稍慢（跑 benchmark）；shape 一变就要重跑，所以 detection 里如果有变长 proposal 走 conv 就不适合。

**`deterministic = False`**

允许 cuDNN 挑用 `atomicAdd` 累加的 kernel。同一个位置多个线程并发累加，浮点加法非结合，结果有最低位差异，但通常快 5–15%。**副作用**：同种子跑两次 loss 不完全一样，复现论文时要关。

**`set_start_method('fork')`**

DataLoader 起 `num_workers=32` 时的差异：

```
spawn:  父进程 ─┐
                ├─► worker0: python -c "import mmcv; import mmdet3d; ..."  ← 30s
                ├─► worker1: python -c "import mmcv; import mmdet3d; ..."  ← 30s
                ...
                └─► 32 个 worker 各自重新 import 一遍

fork:   父进程 ─┐   已经 import 好了 mmcv/mmdet3d
                ├─► worker0: 内存页 COW 复制  ← <1s
                ...
                └─► 32 个 worker 秒起
```

必须在 `if __name__ == '__main__':` 最前面设，晚了会踩 "CUDA has been initialized" 报错 —— 因为一旦父进程 `.cuda()` 过，CUDA context 就不能安全 fork。

**`torchrun` 分布式下这个全局调用会失效**

上面这段代码只在单进程 `python tools/train.py` 直接跑时有效。用 `start_mmdet3d.sh` 走
`torchrun` 拉起 8 卡后，`tools/train.py` 是被 `torchrun` 已经 fork/spawn 出来的子进程里执行的，
`if __name__ == '__main__':` 最前面那次 `set_start_method('fork')` 要么被 Python 多进程框架
判定为"进程已有上下文"而静默失败，要么因为顺序问题根本不会在 `DataLoader` 创建 worker 之前
生效。结果是 `DataLoader` 退回 PyTorch 默认的 `spawn`。

`spawn` 需要把 dataset 对象整个 pickle 后传给 worker 子进程。nuScenes 官方 `DetectionConfig`
把 `class_names` 存成 `dict.keys()` 视图（见 `nuscenes/eval/detection/config.py`），而
`dict_keys` 不能被 pickle，8 卡训练一启动就在每个 rank 上炸：

```
File "/usr/lib/python3.10/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
TypeError: cannot pickle 'dict_keys' object
```

单卡、或者把 `workers_per_gpu` 设成 0 不会触发，因为根本没有创建 DataLoader 子进程。

**TurboPhysAI 的等价修正**：不依赖全局 `set_start_method`，而是在 `build_dataloader()` 创建
每个 `DataLoader` 的调用点显式传入 `multiprocessing_context`（`data.py`，详见
§3.7.1/U5）。这个上下文对象在 `DataLoader.__init__` 里直接生效，不受 `torchrun` 已建立的
进程拓扑影响：

```python
if num_workers > 0:
    context = _dataloader_multiprocessing_context()   # 默认 fork
    if context is not None:
        kwargs["multiprocessing_context"] = context
```

---

### 4.3 `pin_memory=True`

**实际代码**（`datasets/builder.py`）：

```python
data_loader = DataLoader(
    dataset,
    ...
    pin_memory=True,      # 由 False 改为 True
    worker_init_fn=init_fn,
    **kwargs)
```

**发生了什么**

Host → Device 拷贝路径：

```
pin_memory=False:
   DataLoader worker 生产 tensor (pageable 内存)
        │
        ▼
   .cuda() 触发拷贝
        │
        ├─► CUDA driver: 分配临时 pinned buffer
        ├─► 把 pageable 拷到 pinned                   ← 额外一次 memcpy
        └─► DMA: pinned → GPU
        (这段全程同步，GPU 空转)

pin_memory=True:
   DataLoader worker 直接分配 pinned tensor
        │
        ▼
   .cuda(non_blocking=True)
        │
        └─► DMA: pinned → GPU
        (与当前 step 的 GPU 计算 overlap，异步进行)
```

时间线示意：

```
不用 pin:    [H2D copy][forward][backward][H2D copy][forward][backward]
                        ▲                         ▲
                  copy 期间 GPU 空转

用 pin:      [H2D copy][forward       ][backward      ]
                       [ H2D copy 下一批 ]  ← 与计算重叠
```

**副作用**：pinned 内存来自不可换出物理页，`num_workers=32 × samples_per_gpu=8 × 6 张图` 的规模下能占几十 GB 主机内存，容易 OOM；`/dev/shm` 不够会报 `Bus error`。

---

### 4.4 GridMask 的 Dynamo 兼容修复

**实际代码**（`grid_mask.py`）：

```python
@auto_fp16()
@torch._dynamo.disable                              # ①
def forward(self, x):
    ...
    mask = np.asarray(mask)
    mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]
    mask = torch.from_numpy(mask.copy()).to(x.dtype).cuda()   # ②
    ...
```

**背景：`torch.compile` 的工作流**

```
用户代码
   │
   ▼
Dynamo 追踪字节码 ── 遇到不认识的 Python 语义 ─► graph break
   │                                            │
   ▼                                            ▼
FX Graph（纯张量 op）                     eager 执行这一段
   │
   ▼
AOTAutograd 生成前反向图
   │
   ▼
Inductor 编译成 Triton kernel
```

Dynamo 追踪不了的东西：`np.random.rand()`、`PIL.Image.rotate()`、依赖 tensor 值的 Python `if`、外部 C 扩展调用。GridMask 里同时踩到 numpy 随机数和 PIL 旋转，dynamo 会反复尝试→失败→回退，日志刷屏且实际比 eager 慢。

**① `@torch._dynamo.disable` 的作用**

```
未 disable:
   compile 边界 = forward 外壳
   ┌── GridMask.forward ────────────┐
   │  np.random.rand()   ← graph break│  → dynamo 切子图1
   │  PIL rotate         ← graph break│  → dynamo 切子图2
   │  torch.from_numpy   ← graph break│  → dynamo 切子图3
   │  mask * x           ← 张量 op    │  → 尝试 compile
   └─────────────────────────────────┘
   最后：3 段 eager + 1 小段 compile，得不偿失

disable 后:
   compile 边界 = forward 之外
   ┌── GridMask.forward ────┐
   │   整个函数直接 eager    │  ← dynamo 不追踪
   └────────────────────────┘
   编译器专心优化其他真正是热点的模块
```

**② `mask.copy()` 修复 read-only 数组**

```python
mask = np.asarray(mask)      # 从 PIL 转来，buffer 是 read-only
mask = mask[...]             # 切片，仍然 read-only
mask = torch.from_numpy(mask)              # ✗ 共享 buffer，read-only 带过来
mask = torch.from_numpy(mask.copy())       # ✓ 分配可写新内存
```

`torch.from_numpy` 是零拷贝共享 storage，read-only 属性带过来后，AOTAutograd 建反向图时 guard 检查会失败（因为不能安全 in-place），触发 warning + 潜在未定义行为。`.copy()` 把 buffer 变可写，切断和 PIL 的关系。

---

### 4.5 `torch.compile` 计算边界拆分

**实际代码**（`transformer.py`）：

优化前 `forward` 是一大段内联：

```python
def forward(self, mlvl_feats, bev_queries, object_query_embed, ..., **kwargs):
    ...
    bev_embed = self.get_bev_features(...)
    bs = mlvl_feats[0].size(0)
    query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
    query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
    query     = query.unsqueeze(0).expand(bs, -1, -1)
    reference_points   = self.reference_points(query_pos).sigmoid()
    init_reference_out = reference_points
    query     = query.permute(1, 0, 2)
    query_pos = query_pos.permute(1, 0, 2)
    bev_embed = bev_embed.permute(1, 0, 2)
    inter_states, inter_references = self.decoder(
        query=query, key=None, value=bev_embed, query_pos=query_pos,
        reference_points=reference_points, ...,
        spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
        level_start_index=torch.tensor([0], device=query.device),
        **kwargs)
```

优化后抽出小函数：

```python
#@torch.compile(mode="max-autotune-no-cudagraphs")
def initialize_queries_and_bev(self, object_query_embed, bev_embed, bs, bev_h, bev_w):
    query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
    query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
    query     = query.unsqueeze(0).expand(bs, -1, -1)
    reference_points   = self.reference_points(query_pos).sigmoid()
    init_reference_out = reference_points
    query     = query.permute(1, 0, 2)
    query_pos = query_pos.permute(1, 0, 2)
    bev_embed = bev_embed.permute(1, 0, 2)
    spatial_shapes    = torch.tensor([[bev_h, bev_w]], device=query.device)
    level_start_index = torch.tensor([0], device=query.device)
    return query, bev_embed, query_pos, reference_points, spatial_shapes, level_start_index, init_reference_out

def forward(self, ...):
    ...
    query, bev_embed, query_pos, reference_points, spatial_shapes, level_start_index, init_reference_out \
        = self.initialize_queries_and_bev(object_query_embed, bev_embed, bs, bev_h, bev_w)
```

**为什么拆开更好**

```
拆之前：@torch.compile 装在 forward 上
┌──────────────── forward ────────────────┐
│ get_bev_features(...) ← 里面还有很多分支│
│ split / expand / linear / sigmoid       │
│ permute × 3                              │
│ torch.tensor(...) ← host-side 组装       │  ← graph break!
│ decoder(...)  ← 里面又是一堆 Attention   │
└─────────────────────────────────────────┘
Dynamo 追踪时，每遇到 graph break 就切子图，
最后可能变成 5-6 个小 FX 图，各自单独编译，launch 数不降反升。

拆之后：只 compile initialize_queries_and_bev
┌── initialize_queries_and_bev ──┐
│ split → linear → sigmoid       │  一段纯张量 op
│ permute × 3                    │  ← inductor 能 fuse
│ torch.tensor(...) 也在里面     │  ← 边界清晰，不影响 forward 其他部分
└────────────────────────────────┘
    ↓ Inductor
生成 1 个融合 Triton kernel，替代原来 ~8 次 launch。
```

**Kernel launch 数目对比**（示意）：

```
eager:                   launch: 12 次
  split / expand / linear / sigmoid / permute × 3 / to_tensor × 2 / clone × 2

compile(initialize...):  launch: 2-3 次
  一个 fused elementwise + 一次 linear (matmul) + 一个小 memset
```

`mode="max-autotune-no-cudagraphs"` 里两个词：
- `max-autotune`：inductor 对每个候选 tile 大小、block 数都跑 micro-benchmark，选最快版本。编译慢（首个 step 可能 30s+），运行快。
- `no-cudagraphs`：不开 CUDA Graph。训练里 optimizer state 每步变化、随机数发生器状态变化，CUDA Graph 需要形状/指针不变，容易踩坑。生产环境先不用。

同样的拆分手法在优化后的 `encoder.py` 里也能看到：`matmul_1/matmul_2/matmul_3`、`extract_metas`、`down_sample` 都被抽出来，每个上面都留一个 `#@torch.compile` 挂点。这些挂点为什么在本包补不了、基线里的"等价载体"分别是什么，见 **§4.12**。

---

### 4.6 `cdist` 的向量化替换

**实际代码**（`losses/map_loss.py`）：

```python
class OrderedPtsL1Cost(object):
    def __call__(self, bbox_pred, gt_bboxes):
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)          # (N, D)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts*num_orders, -1)  # (M, D)
        # 优化前
        # bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        # 优化后
        bbox_cost = (bbox_pred[:, None, :] - gt_bboxes[None, :, :]).abs().sum(dim=-1)
        return bbox_cost * self.weight
```

**广播减法的形状变化**

```
bbox_pred:  shape (N, D)
                │
                ▼
bbox_pred[:, None, :]:  (N, 1, D)          ┐
                                            │  broadcast
gt_bboxes[None, :, :]:  (1, M, D)          ┘
                │
                ▼
差:                     (N, M, D)          ← elementwise sub
.abs():                 (N, M, D)          ← elementwise abs
.sum(dim=-1):           (N, M)             ← reduce along D
                │
                ▼
bbox_cost:              (N, M)             ← 与 cdist(p=1) 数学等价
```

**cdist 在 ROCm/DCU 上的问题**

`torch.cdist(a, b, p=1)` 是一个"专用融合 kernel"：一个 kernel 里同时算差、abs、sum。CUDA 上覆盖全，ROCm 上 p=1/p=2 并非都有优化实现，很多时候走 fallback（拆成慢通路），反向传播还有额外 bug 记录。

改成广播减法后，路径变成三个**通用 kernel**：

```
cdist 版本:            [ cdist_kernel (可能 fallback) ]
广播版本:              [ sub ] [ abs ] [ sum_reduce ]
              inductor：可以把三个融合成一个 Triton kernel（.compile 时）
```

**代价：中间张量 `(N, M, D)` 显存**

```
cdist:      峰值显存 ≈ (N, D) + (M, D) + (N, M)
广播版:      峰值显存 ≈ (N, D) + (M, D) + (N, M, D) + (N, M)
                                        ↑
                                    多一个 (N,M,D)
```

MapTRv2 里 N ~ 几百，M ~ 几百，D ~ 40（20 点 × 2 坐标），`N*M*D ≈ 几 MB`，完全够用。但如果 N/M 到几万，就会 OOM，那时应改用分块或保留 cdist。

**通用启示**

这类"专用融合 op → 显式广播"的替换，同样适用于：
- `torch.cdist(a, b, p=2)` → `((a[:,None]-b[None,:])**2).sum(-1).sqrt()`
- `torch.pdist` → 上三角切片
- 遇到 kernel 覆盖差 / dynamo 追踪失败的场景，广播减法几乎总能救场。

---

### 4.7 Assigner 静态化

**实际代码**（`maptr_assigner.py`）：

优化前：

```python
def assign(self, bbox_pred, cls_pred, pts_pred, gt_bboxes, gt_labels, gt_pts, ...):
    num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
    if num_gts == 0 or num_bboxes == 0:                     # ← 依赖张量形状的 Python 分支
        if num_gts == 0:
            assigned_gt_inds[:] = 0
        return AssignResult(num_gts, assigned_gt_inds, None,
                            labels=assigned_labels), None
    cls_cost = self.cls_cost(cls_pred, gt_labels)
    normalized_gt_bboxes = normalize_2d_bbox(gt_bboxes, self.pc_range)  # (num_gts, 4)
    ...
    cost = cost.detach().cpu()
    matched_row_inds, matched_col_inds = linear_sum_assignment(cost)     # scipy CPU
    matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
    ...
    assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
    assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
    return AssignResult(num_gts, assigned_gt_inds, None,
                        labels=assigned_labels), order_index
```

优化后：

```python
def assign(self, bbox_pred, cls_pred, pts_pred, gt_bboxes, gt_labels, gt_pts, ...):
    #                              ↑ 现在 gt_bboxes = (padded, valid_mask, real_num_gts)
    num_bboxes = bbox_pred.size(0)
    # 动态 early-return 被删除，靠 padding + mask 处理

    cls_cost = self.cls_cost(cls_pred, gt_labels[0])
    normalized_gt_bboxes = normalize_2d_bbox(gt_bboxes[0], self.pc_range)  # (MAX_GTS, 4) 固定
    ...
    # 匹配矩阵按 padded 形状算，最后再用 valid_mask 切
    assign_result = self.hungarian_match(
        cost[:, gt_bboxes[1]],       # ← 用 valid_mask 切出真实列
        gt_labels, assigned_gt_inds, assigned_labels,
        gt_bboxes[2],                # ← 真实 num_gts
        bbox_pred.device)
    return assign_result, order_index

@torch._dynamo.disable               # ← scipy 调用隔离
def hungarian_match(self, cost, gt_labels, assigned_gt_inds, assigned_labels, num_gts, device):
    cost = cost.detach().cpu()
    matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
    matched_row_inds = torch.as_tensor(matched_row_inds, device=device)
    matched_col_inds = torch.as_tensor(matched_col_inds, device=device)
    assigned_gt_inds[:] = 0
    assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
    assigned_labels[matched_row_inds] = gt_labels[0][matched_col_inds]
    return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
```

**核心是把动态 shape 变成静态 shape**

```
优化前每个 batch：
  step 0:  num_gts=5   → cost shape (100, 5)   ┐
  step 1:  num_gts=12  → cost shape (100, 12)  │  每步形状都变
  step 2:  num_gts=0   → early return           │  → torch.compile 反复 recompile
  step 3:  num_gts=8   → cost shape (100, 8)   ┘

优化后（外部 pad 到 MAX_GTS=200，`pad_to_static_list` 里写死的 max_len）：
  step 0:  padded=(200,), mask=[1,1,1,1,1,0,...,0], real=5 ┐
  step 1:  padded=(200,), mask=[1]*12+[0]*188,      real=12 │  cost shape 恒为 (100, 200)
  step 2:  padded=(200,), mask=[0]*200,             real=0  │  → compile 一次搞定
  step 3:  padded=(200,), mask=[1]*8+[0]*192,       real=8  ┘
```

**Dynamo 视角**

```
Python:   if num_gts == 0 or num_bboxes == 0:
                            ▲
Dynamo:   这是依赖张量形状的 Python if（num_gts = gt_bboxes.size(0)）
          → 只能生成两条不同图，或干脆 graph break

改成：    padded 计算完再切
          cost[:, valid_mask]
                    ▲
          纯 tensor 索引，dynamo 能追踪
```

**scipy 隔离**

`linear_sum_assignment` 是 CPU 上的 C 实现，没法 compile 也没必要 compile（不是热点）。`@torch._dynamo.disable` 相当于给 dynamo 画禁区：

```
dynamo 追踪路径：
  assign(...) ──┐
                ├─► ... 张量计算 ...    ← 正常追踪
                ├─► hungarian_match     ← 遇到 @disable，跳过
                │       │
                │       └─► scipy 逻辑 eager 执行
                └─► ... 后续张量计算 ... ← 继续追踪
```

**通用启示**：`padding + mask` 是把"控制流"翻译成"张量流"的经典手法。凡是想让 `torch.compile` 收益最大化的模型，都要先把动态 shape 消掉。Transformer 里的 attention mask、DETR 类模型的 GT padding，本质都是这个思路。

**注意**：这套静态打包是**跨文件调用约定变更**（`loss` / `get_targets` / `_get_target_single` / `sampler` 都得跟着改），所以本包只落地了其中的 `@torch._dynamo.disable` 部分；为什么没法用 recipe 表达、参考实现自己漏改了哪个调用方，见 **§4.13**。

---

### 4.8 EfficientNet 注册覆盖

**实际代码**：

```python
# 优化前
@BACKBONES.register_module()
class EfficientNet(BaseModule):
    ...
# 优化后
@BACKBONES.register_module(force=True)
class EfficientNet(BaseModule):
    ...
```

**mmcv 的 Registry 冲突**

```
容器镜像启动顺序（无法完全控制）：
  import mmcls  ──► mmcls.models.backbones 里注册 EfficientNet   ← 版本 A
       ↓
  import mmdet ──► mmdet.models.backbones 里 (可能) 注册 EfficientNet ← 版本 B
       ↓
  from projects.mmdet3d_plugin.models.backbones.efficientnet
                ↓
       @BACKBONES.register_module()           ✗ KeyError: 'EfficientNet' is already registered
       @BACKBONES.register_module(force=True) ✓ 覆盖为 MapTR 自己的实现
```

**副作用**：如果自己的实现和上游签名不兼容，其它模块可能默默出错。所以配套要做的是在模块顶部打一行 log：

```python
logger.info(f"BACKBONES['EfficientNet'] force-overridden by {__file__}")
```

---

### 4.9 强制选择 CUDAExtension 构建

**实际代码**（`setup.py`）：

```python
# 优化前
if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension
    sources += source_cuda
    define_macros += [("WITH_CUDA", None)]

# 优化后
if 1:
# if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension
    ...
```

**问题**：构建镜像或 wheel 时通常没有可见 GPU，`torch.cuda.is_available()` 可能返回
`False`，原逻辑会进入 `else` 并抛出 `NotImplementedError`，导致扩展根本无法构建。

**`if 1:` 的作用**：强制选择 `CUDAExtension` 分支，并把前面通过
`glob(os.path.join(extensions_dir, "*.cu"))` 找到的 `.cu` 文件加入构建，不再依赖构建阶段的
设备可见性。在 ROCm PyTorch 环境中，`CUDAExtension` 会走该 PyTorch 发行版对应的 ROCm
扩展构建流程；在 CUDA PyTorch 环境中则走常规 CUDA 编译流程。

更规范的条件应直接判断 PyTorch 是否带 CUDA/ROCm 编译能力：

```python
if torch.version.cuda is not None or torch.version.hip is not None:
    extension = CUDAExtension
    ...
```

**关于 HIP 源文件**：权威版本没有新增 `geometric_kernel_attn_cuda.hip` 或
`geometric_kernel_attn_hip_kernel.cuh`，setup 仍只收集 `*.cu`。因此 B10 的准确描述是
“强制扩展构建”，不是“新增 HIP kernel”。

**边界**：`if 1:` 只绕过 Python 层的设备可用性判断，不会让纯 CPU 环境自动生成可运行算子。
后续编译器仍需存在可用的 nvcc（CUDA）或 hipcc（ROCm/HCU）工具链。

---

### 4.10 PV 掩码向量化

**实际代码**（`nuscenes_offlinemap_dataset.py`，`gen_pv_semantic_mask` 里逐相机调用）：

```python
# 优化前：每条线 200 次 shapely 采样
distances = np.linspace(0, line_ego.length, 200)
coords = np.array([list(line_ego.interpolate(distance).coords)
                   for distance in distances]).reshape(-1, 2)
pix_coords = perspective(lidar_coords, lidar2feat)

# 优化后：一次 numpy 插值拿到全部采样点
x, y = np.array(line_ego.xy)                              # 折线顶点
seg_len = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)      # 每段长度
cum_len = np.concatenate([[0], np.cumsum(seg_len)])       # 顶点处累积弧长
distances = np.linspace(0, cum_len[-1], 200)              # 等弧距查询点
xs = np.interp(distances, cum_len, x)
ys = np.interp(distances, cum_len, y)
pix_coords = project_points(np.stack([xs, ys], axis=1), lidar2feat, z=z)
```

**为什么两种采样等价**

```
折线：A ────────●────────── B ──────●────── C
                 ↑                    ↑
          距 A 为 d 的采样点      距 A 为 d' 的采样点

shapely:  line.interpolate(d)              → 定位到所在段，段内线性插值
numpy :   np.interp(d, [cum_len], [x])     → 同一段内同一线性插值

弧长轴：cum_len = [0, |AB|, |AB|+|BC|, ...]      ← 顶点处的累积长度
        distances = linspace(0, total, 200)      ← 等弧距的 200 个查询点
                                                    每个点落在某段 [cum_len[i], cum_len[i+1]] 内
```

采样点一致 ⇒ 投影后像素一致 ⇒ `cv2.polylines` 画出的掩码逐像素一致（只差 float64 舍入）。

**为什么贵**

```
每个 nuScenes 样本：
  for 每条中心线（几十 ~ 上百条）:
      for cam_index in range(6):                     # 6 路相机
          line_ego_to_pvmask(...)
              └── 200 × line_ego.interpolate(d)      # 每次构造一个 GEOS Point 对象
  ≈ 线数 × 6 × 200 次 shapely/GEOS 调用，全部落在 dataloader worker 的 CPU 上
```

numpy 版本只剩两个 `np.interp`（x、y 各一次），一次向量化插值覆盖全部 200 点；
shapely 版本是 200 次独立调用的循环，两者在 python 层面的调度开销差两个数量级。

**踩过的坑**

- `perspective()` 里的 `pix_coords[2] > 0` 过滤与 `+ 1e-7` 除法必须原样保留，
  否则落在相机后方的点会被画进掩码；
- `z=-1.6` 是 `line_ego_to_pvmask` 的签名默认值，改写时要从 `self`/参数里透传，
  不要在 `project_points` 里写死 `z=0.0`；
- 这类"换等价实现"的改写，验收标准是**逐像素相等**（本包 `test/optimizations/test_maptrv2_implementations.py`
  就拿官方 loop 当参照），不是"看起来差不多"。

---

### 4.11 Shapely 变换合并

**实际代码**（`nuscenes_offlinemap_dataset.py`）：

```python
# 优化前：两次 GEOS 变换，每条线构造两个新 geometry
line_ego = affinity.scale(line_ego, self.scale_x, self.scale_y, origin=(0, 0))
line_ego = affinity.affine_transform(line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])

# 优化后：一次 affine_transform，offset 手算
def scale_translate_geom(self, geom, scale_x, scale_y, trans_x, trans_y, origin=(0, 0)):
    x0, y0 = origin
    xoff = trans_x + x0 - scale_x * x0
    yoff = trans_y + y0 - scale_y * y0
    return affinity.affine_transform(geom, [scale_x, 0.0, 0.0, scale_y, xoff, yoff])
```

**为什么可以合并**

```
affinity.scale(g, sx, sy, origin=(x0,y0)):
    x'  = sx·(x - x0) + x0 = sx·x + (x0 - sx·x0)
    y'  = sy·(y - y0) + y0 = sy·y + (y0 - sy·y0)

再套一次 affine_transform(..., [1,0,0,1,tx,ty]):
    x'' = x' + tx = sx·x + (tx + x0 - sx·x0)   ← 与 xoff 一致
    y'' = y' + ty = sy·y + (ty + y0 - sy·y0)   ← 与 yoff 一致
```

两次变换的线性部分都是对角阵，复合后仍是仿射，`[a, b, d, e, xoff, yoff]` 六元组能一次
表达，于是 GEOS 侧从「缩放→写回→平移→写回」变成一次坐标变换。

**收益与代价**

```
每条中心线的 BEV 掩码绘制（line_ego_to_mask）
  优化前：affinity.scale（1 次 GEOS 变换 + 1 个新 geometry）
        + affinity.affine_transform（1 次 GEOS 变换 + 1 个新 geometry）
  优化后：affinity.affine_transform（1 次）
配套：np.array(list(line_ego.coords)) → np.asarray(line_ego.coords)
      cv2.polylines(mask, np.int32([coords])) → cv2.polylines(mask, [coords])
      LineString(np.array(instance)) → LineString(instance)（再去掉一次顶点拷贝）

量级：line_ego_to_mask 在 aux_seg.bev_seg 打开时对每条 LineString 调用一次；
      本项省下的是「每条线一次 GEOS 变换 + 一个中间 geometry 分配」。
```

**注意**：`np.asarray(line_ego.coords, dtype=np.int32)` 不再像原实现那样 `[:, :2]`
截断，只有在几何确实是二维（ego 平面中心线）时才等价；三维几何会让 `cv2.polylines`
拿到 3 列而报错。本项目输入来自 nuScenes 矢量地图的 `(x, y)` 顶点，属于二维，安全。

**落地到本包**：`scale_translate_geom` 在包内是模块级函数（`pv_mask.py`），不往模型类里塞
新方法；`maptrv2.pv_mask` 的第二个 `replace` 成员把 `VectorizedLocalMap.line_ego_to_mask`
整体替换为 `pv_mask.line_ego_to_mask`。测试对拍官方「scale + 平移」两步写法：随机线段
300/300 掩码逐像素相同，非零 `origin` 下坐标严格相等。

### 4.12 新增 `compile` 挂点无法直接替换的原因

> **更正（已接入）**：本节分析的是"能不能用 `replace(target=...)` 直接命中这 8 个新符号"，结论是不能（原因见下）。但这不等于这 8 个挂点没法接入——`maptrv2.reference_boundaries` group（见 §B6）改用 `wrap` 包装调用这些 helper 的 5 个基线原生方法，绕开了本节说的限制，已经落地。本节内容作为"为什么直接 `replace` 走不通"的原因分析保留。

§4.5 讲了"参考实现为什么把大方法拆成小函数再挂 `@torch.compile`"。这些挂点为什么不能用 recipe 的方式补进 TurboPhysAI？

- **机制上不行：**PhysAI 的剥离手段是 `replace(target=...)`，要求基线里存在同名同签名的符号；这 8 个 helper（`matmul_1/2/3`、`extract_metas`、`down_sample`、`initialize_queries_and_bev`、`compute_decoder_predictions`、`prepare_transformer_inputs`）在基线里 grep 零命中，`optimization generate` 阶段就会报 target 不存在。
- **语义上也不行**：参考实现不是"抽函数"，而是**重新切了计算边界**。`matmul_2` 把三行合成一个 helper 并搬走了 in-place 语义；`down_sample` 把基线分散在 `bev_pool` 和 `LSSTransform.forward` 两处的逻辑合成一个新方法。这类跨方法/跨文件重排，单点替换表达不了。**同理，assigner 静态打包**（GT 预 pad）属于跨文件调用约定变更，要同步改 `loss`/`get_targets`/`_get_target_single`/`assign`/`sampler.sample` 等多处，等于改模型源码。

**8 处挂点与基线"等价载体"的对照**

| # | 参考实现（带 `@torch.compile`） | 基线里承担同样计算的代码 | 载体性质 |
| --- | --- | --- | --- |
| 1-3 | `BaseTransform.matmul_1/2/3`（`encoder.py:93/106/115`） | `BaseTransform.get_geometry_v1`（`encoder.py:90-151`）的 113-118 / 127-131 / 132-137 三段 | 62 行、纯张量、2 个 `if` |
| 4 | `BaseTransform.extract_metas`（`encoder.py:272-291`） | `BaseTransform.forward`（`encoder.py:243-273`）的 `for img_meta` + 5×（np.asarray + new_tensor） | Python 循环 + numpy↔tensor |
| 5 | `LSSTransform.down_sample`（`encoder.py:1128-1135`） | 拆在两处：`BaseTransform.bev_pool` 尾部 collapse-Z（`encoder.py:225-226`）+ `LSSTransform.forward` 的 `self.downsample`（`encoder.py:1111`） | 自定义 CUDA op |
| 6 | `MapTRPerceptionTransformer.initialize_queries_and_bev`（`transformer.py:317-334`） | 同样代码内联在 `transformer.py:378-388`，宿主 `forward` 约 100 行 | 含 decoder / 注意力 |
| 7 | `MapTRv2Head.compute_decoder_predictions`（`maptrv2_head.py:281-337`） | `MapTRv2Head.forward` 的 364-450 段（`for lvl` + `list.append`） | Python 循环 + list |
| 8 | `MapTRv2Head.prepare_transformer_inputs`（`maptrv2_head.py:339-386`） | `MapTRv2Head.forward` 的 295-329 段（num_vec 选择、query/bev 组装、self_attn_mask） | 依赖配置态的 Python 分支 |

为什么不能按键名换掉。三层原因，越往后越根本：

1. **符号不存在**：包机制是 `replace(target=...)`，`target` 必须在基线 worktree 里解析到定义行。上面那些新名字在基线 `grep` 是零命中，`optimization generate` 阶段就会报「target 不存在」。
2. **参考实现把计算边界重新切过**：不存在同签名同语义的旧函数可以换。以 `matmul_2` 为例：

```python
# 基线 encoder.py:127-131：三步分开写
combine = rots.matmul(torch.inverse(intrins))
points  = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
points += trans.view(B, N, 1, 1, 1, 3)               # 平移
points -= lidar2ego_trans.view(B, 1, 1, 1, 1, 3)     # 再去 lidar 外参

# 参考实现 encoder.py:107-113：合并进一个 helper，且 inverse 被提到 helper 外面
intrins = torch.inverse(intrins)                     # get_geometry_v1:159
points  = self.matmul_2(rots, intrins, trans, lidar2ego_trans, points)
#   def matmul_2(...): combine = x @ y; points = combine @ points
#                      points += trans; points -= lidar2ego_trans
```

​    `+=` / `-=` 这种 **in-place 语义**也被搬进了 helper，这不是换个实现，而是"三行合一 + 改写原地修改的时机"。
3. **是跨方法重排，不是替换**：`down_sample` 最能说明问题。

```
基线                                                     参考实现
BaseTransform.bev_pool:                                  BaseTransform.bev_pool:
  x = bev_pool(x, geom_feats, ...)  ← 自定义 CUDA op
  final = cat(x.unbind(dim=2), 1)  ← collapse Z              （collapse 注释掉，原样返回 x）
  return final
BaseTransform.forward:298  x.permute(0,1,3,2)           （注释掉）
LSSTransform.forward:1111  self.downsample(x)            LSSTransform.down_sample:1129
                                                         permute → unbind/cat → permute → downsample
```

   "collapse Z + 两次 permute + downsample"被合成一个新方法，而基线里这段工作
   分散在**两个文件的两个方法**里，一次 `replace` 表达不了。

**如果改用"编译外层大方法"，会撞上什么**

Dynamo 的编译单位是"连续可追踪的张量段"，遇到不可追踪的东西就 graph break：

```
      ┌─ FX graph #1 ─┐ break  ┌─ FX graph #2 ─┐ break  ┌─ FX graph #3 ─┐
输入 →│  纯张量运算段  │ ─────→ │  再一段张量   │ ─────→ │     ...       │ → 输出
      └───────────────┘        └───────────────┘        └───────────────┘
   每个 break 的代价：同步、把 shape/.item() 取回 host、重建 Python 对象、重新进图
```

逐个载体对号入座：

| 载体（基线） | 里面的断裂点 | 评价 |
| --- | --- | --- |
| `BaseTransform.get_geometry_v1`（`encoder.py:90-151`） | `if self.frustum == None`（写 module 状态，产生 guard）、`if "extra_rots" in kwargs`（配置不用则恒 False） | **8 处里唯一较优的候选**：纯张量 + 3 个 `torch.inverse` |
| `BaseTransform.forward`（`encoder.py:230-300`） | 5 次 `np.asarray` + `new_tensor`（numpy 交互必 break）、Python `for`、`get_mlp_input`/`get_cam_feats`（nn.Module）、`bev_pool` | 相当于编译半个 LSS pipeline |
| `BaseTransform.bev_pool`（`encoder.py:192-228`） | `from mmdet3d.ops import bev_pool`（`encoder.py:8`）是自定义 CUDA op；`cat([torch.full(...) for ix in range(B)])` list 推导；`x = x[kept]` 布尔掩码 → **输出形状依赖数据** | 每个 batch 有效点数不同就重编译 |
| `MapTRPerceptionTransformer.forward`（`transformer.py:315-411`） | `get_bev_features`（BEV 编码器）+ `self.decoder(...)` + `**kwargs` 透传 | 等于编译整个 transformer |
| `MapTRv2Head.forward`（`maptrv2_head.py:279-500`） | `if only_bev:` 提前 return dict；`if self.training` / `query_embed_type == "all_pts"` 配置态 guard；`for lvl` + `list.append` + `if lvl == 0`；夹着 transformer 与 seg_head | 一个函数里 4 类断裂点 |

**结论**：参考实现不是"顺手抽函数"，而是**为编译刻意重构** —— 把纯张量段切成编译单元，
把 CUDA op / scipy / nn.Module / 配置态分支留在 eager 侧。基线没有这些切口，所以要么改模型
源码，要么把大方法整体塞进编译（收益未知）。

**还想试的话，路线是这样的**

1. 在 `maptrv2_optimization/catalog.py` 的 `_COMPILE_TARGETS` 追加目标，例如
   `...maptr.modules.encoder.BaseTransform.get_geometry_v1`；
2. `optimization generate` 重新采 target 证据（否则 `trust` 哈希过期）；
3. 上机观测：
   - `TORCH_LOGS=graph_breaks,recompiles` 或 `torch._dynamo.explain(fn)(...)` → 图数量 / break 原因 / 重编译次数；
   - `TORCH_LOGS=inductor`（或 `tlparse`）→ 生成 kernel 数与融合情况；
   - profiler 对比该段 `self_cuda_time`，以及"冷启动 + 首次迭代"的额外耗时。
4. 风险：本包 `compile_wrapper` 默认 `mode="max-autotune-no-cudagraphs"`，与参考实现的
   18 处挂点相同（参考唯一不同的是 `MapTRAssigner.assign` 的 cudagraphs 选项）；对
   `get_geometry_v1` 这种只有几个 kernel 的小段，autotune 的投入产出比未必划算，
   建议先用默认 mode 对比。

**顺带更正**：`matmul_1/2/3` 的载体是 `BaseTransform.get_geometry_v1`（`encoder.py:90`），
不是 `get_geometry:154` —— 后者走 `lidar2img` + `torch.linalg.solve`，在本配置里是**死代码**
（`encoder.py:275` 的调用是注释，真正生效的是 282 行）。

---

### 4.13 Assigner 静态化的跨文件契约

§4.7 讲了 padding+mask 的收益（把动态 shape 变成静态 shape）。本节讲它的**代价**：
为什么本包只做了 `@torch._dynamo.disable`，而没有把静态打包也剥进来。

**参考实现改的是三件套**

1. 新增 `pad_to_static_list`（`maptrv2_head.py:801-818`）：

```python
def pad_to_static_list(self, tensors, pad_value=0, device=None):
    max_len = 200                              # 写死的固定长度，不是 batch 内 max
    results = []
    for t in tensors:
        out = torch.full((max_len,) + t.shape[1:], pad_value, device=device, dtype=t.dtype)
        mask = torch.zeros(max_len, dtype=torch.bool, device=device)
        length = t.size(0)
        out[:length, ...] = t
        mask[:length] = 1
        results.append((out, mask, length))    # 三元组：padded / valid_mask / real_len
    return results
```

2. `MapTRAssigner.assign` 改成消费三元组（`maptr_assigner.py:136-231`）：

```python
@torch.compile(options={"triton.cudagraphs": True, "triton.cudagraph_trees": False})
def assign(self, bbox_pred, cls_pred, pts_pred, gt_bboxes, gt_labels, gt_pts, ...):
    #  名字没变，类型变了：(num_gt,4) → (padded(200,4), mask(200,), real_len)
    cls_cost = self.cls_cost(cls_pred, gt_labels[0])                       # 200 列全算
    normalized_gt_bboxes = normalize_2d_bbox(gt_bboxes[0], self.pc_range)  # 200 行全算
    pts_cost_ordered = pts_cost_ordered.view(num_bboxes, gt_bboxes[0].size(0), num_orders)
    assign_result = self.hungarian_match(cost[:, gt_bboxes[1]], gt_labels,  # mask 选有效列
                                         assigned_gt_inds, assigned_labels,
                                         gt_bboxes[2], bbox_pred.device)    # 真实 num_gts
```

3. 拆出 `hungarian_match` 并 `@torch._dynamo.disable`（`maptr_assigner.py:117-134`）——
   注意它**和 `@torch.compile` 是配套的**：可编译的纯张量段进图，scipy 段划禁区。

**调用约定是怎么顺着文件传下去的**

```
基线（每个图 × 每个 decoder 层）
  head.loss ─┬─ gt_bboxes_list = [gt.bbox.to(device) for ...]     ← 每图形状不同
             └─ loss_single → get_targets → multi_apply(_get_target_single)
                                   └→ assigner.assign(gt_bboxes: (N,4), gt_labels: (N,))
                                         ├→ sampler.sample(..., gt_bboxes)        (mmdet PseudoSampler)
                                         └→ gt_labels[matched_col_inds] / num_gts = gt_bboxes.size(0)

参考实现
  head.loss:915-917 ─ pad_to_static_list ×3 ─→ [(padded(200,4), mask(200,), len), ...]
             └─ loss_single → get_targets → _get_target_single:604
                                   └→ assigner.assign(gt_bboxes/gt_labels/gt_pts = 三元组)
                                         ├→ sampler.sample(..., gt_bboxes[0])                   ← 实参要改
                                         ├→ get_label_result(..., gt_bboxes[0], gt_labels[0])   ← 新函数
                                         └→ hungarian_match(cost[:, mask], ..., len)            ← 新函数 + disable
```

要同步改的调用点：`loss`（造 pad）、`loss_single`、`get_targets`、`_get_target_single`、
`assigner.assign` / `hungarian_match`、下游 `sampler.sample` —— **少改一处就静默出错**。

**为什么 recipe 表达不了**

- `replace` 的隐含契约是"同名同签名、输入输出语义不变"（像 `line_ego_to_pvmask` 那样）。
  这里输入类型变了、上游构造变了、下游消费变了，属于**接口变更**而非**实现替换**。
- 机制上一次只挂一个符号，且前提是"不改基线源码"。跨文件同步改 6 处 ≈ 改模型源码，
  超出配置化剥离的边界。
- 最硬的证据：**参考实现自己没改全**。两个头共用同一个 `MapTRAssigner`
  （`configs/maptr/*.py` 与 `configs/maptrv2/*.py` 都写 `type="MapTRAssigner"`），
  但 MapTR v1 头的调用点在参考实现里仍是老写法：

```python
# 参考实现 maptr_head.py:385 —— 传的是普通 (num_gt, 4) 张量
assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                                  gt_bboxes, gt_labels, gt_shifts_pts, ...)
#   assign 内部却是 gt_bboxes[0]/[1]/[2]：普通张量会被当成"第一行/第二行/第三行"
#   → MapTR v1 的 config 要么报错，要么算错
```

  这就是"跨文件调用约定变更"的真实代价：改一处，必须审全部调用方。

**收益与残留问题**

```
基线：gt_bboxes = (num_gt, 4)，num_gt ∈ {0,1,2,...,~80} 每样本不同
      + assign 里还有 if num_gts == 0 or num_bboxes == 0 的数据相关分支
              ↓
      @torch.compile(...cudagraphs=True) 要求形状稳定
      → 每个 num_gt 一套图 / 一套 cudagraph buffer，0-GT 样本还要额外特化
      → 重编译风暴，cudagraphs 基本用不起来

参考实现：padded 恒 200 行 → cost 矩阵恒为 (num_query, 200)
      → 图只有一套；0-GT 不再走早退分支，而是被 mask 屏蔽
      残留：cost[:, mask] 仍是数据相关选择（输出列数随 num_gts 变），
            紧接着的 hungarian_match 又是 dynamo.disable 边界
      代价：cost 永远按 200 列算（多数样本只有 10~30 条线，白算 6~20 倍）、
            pad 缓冲常驻显存、max_len=200 写死（大 num_vec 的 config 有溢出风险）
```

**本包的选择**

`maptrv2.assigner` 把 SciPy 段整段隔离在图外：基线没有 `hungarian_match` 这个符号，
`assigner.py` 自己定义了 `_hungarian_match_impl`，再对它做 `torch._dynamo.disable`，
与参考实现的 `@torch._dynamo.disable hungarian_match` 是同一手法。

**本文早先版本认为"无法安全表达完整静态契约、只能做 dynamo 隔离"，这一点已被实际实现推翻**：
`maptrv2.assigner` 现在用 3 个 `replace` 目标（`assign` / `loss` / `_get_target_single`）覆盖了
整条链，`pad_to_static_list` 的逻辑塞在被替换的 `loss` 内部而不是新增 head 方法，因此不需要
改动调用约定。当前配置 `enabled: true`，详见 §3.7.1 U4。

仍未复刻的是参考对 `assign` 的 `@torch.compile(options={"triton.cudagraphs": True, ...})`：
本包由 `TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE` 控制且默认不开，开启时用
`max-autotune-no-cudagraphs`。若要评估收益，先用 profiler 量 `assign` 占比（含 graph break
造成的同步开销）。

## 5. 优化点定位方法

上面的每一项都是"看到症状 → 想到手法"的过程，通用流程：

**1) profile 拿 trace**

```
python -m torch.profiler ...   or   nsys profile / rocprof
              ↓
        chrome trace / TensorBoard
              ↓
        按 self CUDA time 排序，看 top-N
```

**2) 按症状对号入座**

```
症状                                    大概率手法
─────────────────────────────────────────────────────────────────
GPU util 低 + host time 高              DataLoader / pin_memory / fork  (§4.2, §4.3)
tiny kernel 上百个                       fusion → torch.compile           (§4.5)
Conv 时间长但 fp16 没提速                channels_last                     (§4.1)
某个 op 出现在 top-1 且异常长             换等价实现（cdist → 广播）        (§4.6)
dataloader worker CPU 饱和、shapely 占 top  numpy 向量化重采样               (§4.10)
compile 日志刷 "graph break"             _dynamo.disable / padding+mask   (§4.4, §4.7)
镜像里 import 冲突                      registry force=True               (§4.8)
Conv2d channel 不匹配、实际 dim 是 H    ROCm bev_pool 返回 NHWC layout，  (§3.7.1/U3)
  而非 C（如 got 200 channels not 256）     需 maptrv2.bev_pool_fix
torchrun 多卡起 DataLoader worker 时      spawn 下 dict_keys 不可 pickle， (§3.7.1/U5)
  TypeError: cannot pickle 'dict_keys' object   需 DataLoader 级 multiprocessing_context=fork
```

**3) 一次只改一项**

```
    baseline  ─►  开 B1  ─►  开 B1+B2  ─►  开 B1+B2+B3  ─►  ...
       ▲            │           │              │
       │            ▼           ▼              ▼
       └──── 精度 A/B ─── 精度 A/B ──── 精度 A/B ──── ...
```

TurboPhysAI 里每个 `Group` 独立开关就是为了支持这种二分 A/B。

**4) 精度守门**

每加一项跑一个短 schedule（如 MapTRv2 的 1 epoch），对齐 loss 曲线和关键 metric（mAP、chamfer）。掉了就回滚。

**5) 记录 baseline**

固定机器、固定数据、固定 seed，用表格记录：

```
Config                       step time  peak mem  mAP@0.5
─────────────────────────────────────────────────────────
baseline                     820ms      18.2 GB   0.615
+ channels_last              690ms      18.4 GB   0.616
+ cudnn.benchmark            640ms      18.4 GB   0.616
+ pin_memory                 620ms      18.5 GB   0.614  ← IO 影响？
+ cdist → broadcast          610ms      18.6 GB   0.615
+ torch.compile (subset)     540ms      19.1 GB   0.615
...
```

有表格才能理性判断"这一项要不要留下"，否则很容易被感觉误导。
