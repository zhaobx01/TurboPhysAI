# 模型支持清单

本页列出 TurboPhysAI 已支持的模型、基线和使用说明。

优化接入基线是模型优化开发、正确性验证和性能验证所使用的模型 commit。推荐使用表中基线，以保持模型代码与随包优化配置的验证条件一致。

TurboPhysAI 不强制模型仓库停留在优化接入基线（[可以在优化接入基线之外的模型版本上使用吗？](../faq.md#可以在优化接入基线之外的模型版本上使用吗)）。

## 已支持模型

|   模型   |     Method     | Backbone |                 优化接入基线                 |                       使用说明                       |
| :-------: | :------------: | :------: | :------------------------------------------: | :---------------------------------------------------: |
| BEVFormer | BEVFormer-base | R101-DCN | `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` | [BEVFormer](../../../model_examples/BEVFormer/README.md) |
| BEVFusion |   BEVFusion   |    —    | `326653dc06e0938edf1aae7d01efcd158ba83de5` | [BEVFusion](../../../model_examples/BEVFusion/README.md) |
|  OpenVLA  |   OpenVLA-7B   | DINOv2+SigLIP | `c8f03f48af692657d3060c19588038c7220e9af9` | [OpenVLA](../../../model_examples/OpenVLA/README.md) |
| DiT4DiT | DiT4DiT | Cosmos 2.5 + Action DiT | `d8854a11713f846b8b3f768fc3bf2c6cba2bc17f` | [DiT4DiT](../../../model_examples/DiT4DiT/README.md) |
|   模型    |       Method       |   Backbone    |                 优化接入基线                 |                        使用说明                         |
| :-------: | :----------------: | :-----------: | :------------------------------------------: | :-----------------------------------------------------: |
| BEVFormer |   BEVFormer-base   |   R101-DCN    | `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` | [BEVFormer](../../../model_examples/BEVFormer/README.md) |
| BEVFusion |     BEVFusion      |       —       | `326653dc06e0938edf1aae7d01efcd158ba83de5` | [BEVFusion](../../../model_examples/BEVFusion/README.md) |
|  OpenVLA  |     OpenVLA-7B     | DINOv2+SigLIP | `c8f03f48af692657d3060c19588038c7220e9af9` |   [OpenVLA](../../../model_examples/OpenVLA/README.md)  |
|  OpenPI   | PI0.5 LIBERO 训练 |       —       | `215abfb217dbac7d5f1273282331b9b1866c0479` |    [OpenPI](../../../model_examples/OpenPi/README.md)   |
| FlashOCC | FlashOCC-R50 | ResNet-50 | `449dc7947d6549ccb72a99d6f6d8c02f166ec0dd` | [FlashOCC](../../../model_examples/FlashOCC/README.md) |
| PointPillars | PointPillars nuScenes | SECOND | `fe25f7a51d36e3702f961e198894580d83c4387b` | [PointPillars](../../../model_examples/PointPillars/README.md) |
| LingBot-VA | LingBot-VA LIBERO | Wan2.2 video-action | `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb` | [LingBot-VA](../../../model_examples/LingBot-VA/README.md) |
| LingBot-VLA-V2 | LingBot-VLA-V2-6B | Qwen3-VL-4B | `be969b8fd117fb70550c5d4bf4bc328211b5b1b6` | [LingBot-VLA-V2](../../../model_examples/Lingbot-VLA-V2/README.md) |
| GR00T | GR00T N1.7 3B | Cosmos Reason2 / Qwen3-VL | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` | [GR00T](../../../model_examples/Groot/README.md) |
|  DreamZero  |   DreamZero-5B   | Wan2.2 5B TI2V | `ab790c198fbce33503358efbbd4187ce9a89adf3` | [DreamZero](../../../model_examples/DreamZero/README.md) |
|  Sparse4D  |  Sparse4D   | Res50 | `249ffbb695f4e9db628d953e2bf6d36de04bbb69` | [Sparse4D](../../../model_examples/Sparse4D/README.md) |
| LeRobot ACT | ACT | ResNet18 | `3f2179f3b69708b6ad009b2e7685dd9d05269ee1` | [LeRobot ACT](../../../model_examples/LeRobot-ACT/README.md) |

**自动驾驶**、**具身智能**和**世界模型**等方向的模型适配工作持续推进中，请关注后续更新。
