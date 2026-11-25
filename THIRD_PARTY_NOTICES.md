# Third-Party Notices and Provenance

This file records the provenance of source code distributed in TurboPhysAI. The root [BSD-3-Clause License](LICENSE) applies to Hygon-authored material only. A source file derived from an upstream work remains subject to its upstream license and notices.

## Confirmed Apache-2.0 derivations

### OpenWAM / native Wan training

- Source repository: <https://github.com/iMountTai/OpenWAM>
- Source branch and fixed commit: `main` @ `ac8aab4f6ba16b1739c94df823cc444478e9dab0`
- Existing ports retain their original derivation headers; `online_encode.py`, `ports/wan.py` and ZeRO configuration in `__init__.py` include the later post-gather scheduling and HIP BLASLt workspace updates.
- Runtime target baseline: `1204dde1b9e9c232131fc99a3f2c6b78330e7f64`
- License: Apache-2.0, including the source's native Wan license material.
- Local source ports and helper modules are modified; they retain Apache-2.0 identifiers and Hygon modification notices. Original class identities are retained; functions are installed as Turbo wrappers.

| Local files under `turbo_physai/optimizations/models/openwam/` | Direct source path |
| --- | --- |
| Optimization helpers (`attention.py`, `_fa2_varlen.py`, `deferred_gather.py`, `global_compile.py`, `host_async.py`, `hw_queues.py`, `latent_cache.py`, `linear_bias2d.py`, `norms.py`, `online_encode.py`, `pointwise.py`, `runtime.py`, `sac.py`, `text.py`, `text_store.py`, `vae_layout.py`, `zero.py`, `zero_fused.py`, `zero_kernels.py`, `zero_norm.py`) and model preparation in `__init__.py` | Same-named files under `openwam/optimizations/` |
| `ports/data.py` | `openwam/dataloader/robotwin.py` |
| `ports/action_components.py`, `ports/action.py` | `openwam/model/action_backbone/components.py`, `separate_action_dit.py` |
| `ports/architecture.py` | `openwam/model/architectures/base.py` |
| `ports/idm.py`, `ports/joint.py`, `ports/mot.py` | `openwam/model/architectures/dual_system/idm.py`, `joint_self_attn.py`, `mot_driver.py` |
| `ports/encode.py`, `ports/preprocess.py` | `openwam/model/video_backbone/wan/encode.py`, `preprocess.py` |
| `ports/dit.py`, `ports/vae.py` | `openwam/model/video_backbone/wan/models/dit.py`, `vae.py` |
| `ports/wan.py` | `openwam/model/video_backbone/wan_backbone.py` |
| `ports/trainer.py` | `openwam/train/openwam_trainer.py` |
| `cache_vae_latents.py` | `scripts/cache_vae_latents.py` |

The complete source/symbol mapping is packaged in `configs/provenance.json`.
The full license text is retained in [third_party/licenses/OpenWAM-Apache-2.0.txt](third_party/licenses/OpenWAM-Apache-2.0.txt).
No upstream `NOTICE` file is present in the source commit's ported directories.

The tables below record every local derivative file and its direct upstream source. Each source repository is fixed to a 40-character commit. `Modified` means that the upstream notice is retained and a Hygon modification notice is present in the local source. `Unmodified` means that the local file is identical to the upstream file at the fixed commit. No `NOTICE` file was present in the listed upstream baselines.

### BEVFusion / OpenMMLab mmdet3d

- Repository: <https://github.com/mit-han-lab/bevfusion>
- Fixed commit: `326653dc06e0938edf1aae7d01efcd158ba83de5`
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `kernel/bev_pool/src/bev_pool_cpu.cpp` | `mmdet3d/ops/bev_pool/src/bev_pool_cpu.cpp` | Modified |
| `kernel/bev_pool/src/bev_pool_cuda.cu` | `mmdet3d/ops/bev_pool/src/bev_pool_cuda.cu` | Modified |
| `kernel/voxelization/src/scatter_points_cpu.cpp` | `mmdet3d/ops/voxel/src/scatter_points_cpu.cpp` | Modified |
| `kernel/voxelization/src/scatter_points_cuda.cu` | `mmdet3d/ops/voxel/src/scatter_points_cuda.cu` | Modified |
| `kernel/voxelization/src/voxelization.cpp` | `mmdet3d/ops/voxel/src/voxelization.cpp` | Modified |
| `kernel/voxelization/src/voxelization.h` | `mmdet3d/ops/voxel/src/voxelization.h` | Modified |
| `kernel/voxelization/src/voxelization_cpu.cpp` | `mmdet3d/ops/voxel/src/voxelization_cpu.cpp` | Modified |
| `kernel/voxelization/src/voxelization_cuda.cu` | `mmdet3d/ops/voxel/src/voxelization_cuda.cu` | Modified |
| `turbo_physai/optimizations/common/mmdet3d/bev_pool.py` | `mmdet3d/ops/bev_pool/bev_pool.py` | Modified |
| `turbo_physai/optimizations/common/mmdet3d/gaussian.py` | `mmdet3d/core/utils/gaussian.py` | Modified |
| `turbo_physai/optimizations/common/mmdet3d/sparse_conv.py` | `mmdet3d/ops/spconv/ops.py` | Modified |
| `turbo_physai/optimizations/common/mmdet3d/sparse_tensor.py` | `mmdet3d/ops/spconv/structure.py` | Modified |
| `turbo_physai/optimizations/common/mmdet3d/voxelization.py` | `mmdet3d/ops/voxel/voxelize.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/backbone.py` | `mmdet3d/models/fusion_models/bevfusion.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/depth.py` | `mmdet3d/models/vtransforms/base.py`; `mmdet3d/models/vtransforms/depth_lss.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/gaussian.py` | `mmdet3d/core/utils/gaussian.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/training.py` | `mmdet3d/models/fusion_models/base.py`; `mmdet3d/apis/train.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/transfusion.py` | `mmdet3d/models/heads/bbox/transfusion.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/transfusion_bbox_coder.py` | `mmdet3d/core/bbox/coders/transfusion_bbox_coder.py` | Modified |
| `turbo_physai/optimizations/models/bevfusion/transfusion_bbox_coder_runtime.py` | `mmdet3d/core/bbox/coders/transfusion_bbox_coder.py` | Modified |

### BEVFusion bundled spconv / tensorview

- Repository: <https://github.com/mit-han-lab/bevfusion>
- Fixed commit: `326653dc06e0938edf1aae7d01efcd158ba83de5`
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `kernel/sparse_conv/include/paramsgrid.h` | `mmdet3d/ops/spconv/include/paramsgrid.h` | Modified |
| `kernel/sparse_conv/include/pybind11_utils.h` | `mmdet3d/ops/spconv/include/pybind11_utils.h` | Modified |
| `kernel/sparse_conv/include/spconv/fused_spconv_ops.h` | `mmdet3d/ops/spconv/include/spconv/fused_spconv_ops.h` | Modified |
| `kernel/sparse_conv/include/spconv/geometry.h` | `mmdet3d/ops/spconv/include/spconv/geometry.h` | Modified |
| `kernel/sparse_conv/include/spconv/indice.cu.h` | `mmdet3d/ops/spconv/include/spconv/indice.cu.h` | Modified |
| `kernel/sparse_conv/include/spconv/indice.h` | `mmdet3d/ops/spconv/include/spconv/indice.h` | Modified |
| `kernel/sparse_conv/include/spconv/maxpool.h` | `mmdet3d/ops/spconv/include/spconv/maxpool.h` | Modified |
| `kernel/sparse_conv/include/spconv/mp_helper.h` | `mmdet3d/ops/spconv/include/spconv/mp_helper.h` | Unmodified |
| `kernel/sparse_conv/include/spconv/point2voxel.h` | `mmdet3d/ops/spconv/include/spconv/point2voxel.h` | Modified |
| `kernel/sparse_conv/include/spconv/pool_ops.h` | `mmdet3d/ops/spconv/include/spconv/pool_ops.h` | Modified |
| `kernel/sparse_conv/include/spconv/reordering.cu.h` | `mmdet3d/ops/spconv/include/spconv/reordering.cu.h` | Modified |
| `kernel/sparse_conv/include/spconv/reordering.h` | `mmdet3d/ops/spconv/include/spconv/reordering.h` | Modified |
| `kernel/sparse_conv/include/spconv/spconv_ops.h` | `mmdet3d/ops/spconv/include/spconv/spconv_ops.h` | Modified |
| `kernel/sparse_conv/include/tensorview/helper_kernel.cu.h` | `mmdet3d/ops/spconv/include/tensorview/helper_kernel.cu.h` | Unmodified |
| `kernel/sparse_conv/include/tensorview/helper_launch.h` | `mmdet3d/ops/spconv/include/tensorview/helper_launch.h` | Unmodified |
| `kernel/sparse_conv/include/tensorview/tensorview.h` | `mmdet3d/ops/spconv/include/tensorview/tensorview.h` | Modified |
| `kernel/sparse_conv/include/torch_utils.h` | `mmdet3d/ops/spconv/include/torch_utils.h` | Modified |
| `kernel/sparse_conv/include/utility/timer.h` | `mmdet3d/ops/spconv/include/utility/timer.h` | Modified |
| `kernel/sparse_conv/src/all.cc` | `mmdet3d/ops/spconv/src/all.cc` | Modified |
| `kernel/sparse_conv/src/indice_cpu.cc` | `mmdet3d/ops/spconv/src/indice_cpu.cc` | Modified |
| `kernel/sparse_conv/src/indice_cuda.cu` | `mmdet3d/ops/spconv/src/indice_cuda.cu` | Modified |
| `kernel/sparse_conv/src/maxpool_cpu.cc` | `mmdet3d/ops/spconv/src/maxpool_cpu.cc` | Modified |
| `kernel/sparse_conv/src/maxpool_cuda.cu` | `mmdet3d/ops/spconv/src/maxpool_cuda.cu` | Modified |
| `kernel/sparse_conv/src/reordering_cpu.cc` | `mmdet3d/ops/spconv/src/reordering_cpu.cc` | Modified |
| `kernel/sparse_conv/src/reordering_cuda.cu` | `mmdet3d/ops/spconv/src/reordering_cuda.cu` | Modified |

The bundled `kernel/sparse_conv/include/prettyprint.h` file is registered separately under [Boost Software License 1.0 material](#confirmed-boost-software-license-10-material).

### BEVFormer

- Repository: <https://github.com/fundamentalvision/BEVFormer>
- Fixed commit: `66b65f3a1f58caf0507cb2a971b9c0e7f842376c`
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/bevformer/backbone.py` | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/data.py` | `projects/mmdet3d_plugin/datasets/builder.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/geometry_sca.py` | `projects/mmdet3d_plugin/bevformer/modules/transformer.py`; `projects/mmdet3d_plugin/bevformer/modules/encoder.py`; `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/grid_mask.py` | `projects/mmdet3d_plugin/models/utils/grid_mask.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/msda.py` | `projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/training.py` | `projects/mmdet3d_plugin/bevformer/apis/train.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/tsa.py` | `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py` | Modified |

### Diffusers Cosmos transformer

- Repository: <https://github.com/huggingface/diffusers>
- Fixed commit: `275869dcae4ebcfee6a80253fdabc56033335020` (v0.38.0)
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/dit4dit/cosmos_attention.py` | `src/diffusers/models/transformers/transformer_cosmos.py` | Modified |
| `turbo_physai/optimizations/models/dit4dit/cosmos_forward.py` | `src/diffusers/models/transformers/transformer_cosmos.py` (`CosmosTransformer3DModel.forward`) | Modified |

### OpenPI

- Repository: <https://github.com/Physical-Intelligence/openpi>
- Fixed commit: `215abfb217dbac7d5f1273282331b9b1866c0479`
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/openpi/gemma_checkpoint_stride.py` | `src/openpi/models_pytorch/gemma_pytorch.py` | Modified |

### FlashOCC

- Repository: <https://github.com/Yzichen/FlashOCC>
- Fixed commit: `449dc7947d6549ccb72a99d6f6d8c02f166ec0dd`
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/flashocc/geometry_compile.py` | `projects/mmdet3d_plugin/models/necks/view_transformer.py` (`LSSViewTransformer.get_ego_coor`) | Modified |
| `turbo_physai/optimizations/models/flashocc/voxel_pooling_compile.py` | `projects/mmdet3d_plugin/models/necks/view_transformer.py` (`LSSViewTransformer.voxel_pooling_prepare_v2`) | Modified |
| `turbo_physai/optimizations/models/flashocc/compat_sources/dvr/dvr.cpp` | `lib/dvr/dvr.cpp` | Unmodified |
| `turbo_physai/optimizations/models/flashocc/compat_sources/dvr/dvr.cu` | `lib/dvr/dvr.cu` | Modified |

The DVR files retain FlashOCC's attribution to [4d-occ-forecasting](https://github.com/tarashakhurana/4d-occ-forecasting), whose repository includes an [MIT license](https://github.com/tarashakhurana/4d-occ-forecasting/blob/e1ea9434113d303986684395a961aaacac90d470/LICENSE).

### PointPillars / OpenMMLab MMDetection3D

- Repository: <https://github.com/open-mmlab/mmdetection3d>
- Fixed commit: `fe25f7a51d36e3702f961e198894580d83c4387b`
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/pointpillars/unique_free_sampler.py` | `mmdet3d/models/task_modules/samplers/pseudosample.py` | Modified |
| `turbo_physai/optimizations/models/pointpillars/pillar_scatter.py` | `mmdet3d/models/middle_encoders/pillar_scatter.py` | Modified |
| `turbo_physai/optimizations/models/pointpillars/native_limit_period.py` | `mmdet3d/structures/bbox_3d/utils.py` | Modified |
| `turbo_physai/optimizations/models/pointpillars/compiled_nearest_bev_iou.py` | `mmdet3d/structures/ops/iou3d_calculator.py`; `mmdet3d/structures/bbox_3d/base_box3d.py` | Modified |

### PointPillars / OpenMMLab MMDetection

- Repository: <https://github.com/open-mmlab/mmdetection>
- Fixed commit: `fe3f809a0a514189baf889aa358c498d51ee36cd` (v3.2.0)
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/pointpillars/tensorized_assigner.py` | `mmdet/models/task_modules/assigners/max_iou_assigner.py` | Modified |

### LeRobot ACT

- Repository: <https://github.com/huggingface/lerobot>
- Fixed commit: `3f2179f3b69708b6ad009b2e7685dd9d05269ee1`
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/lerobot_act/deferred_metrics.py` | `src/lerobot/policies/act/modeling_act.py` | Modified |

### LeRobot ACT / Hugging Face Accelerate

- Repository: <https://github.com/huggingface/accelerate>
- Fixed commit: `6afc1e5ee217051fde702b23de2813344dc0fd33` (v1.15.0)
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/lerobot_act/h2d_prefetch_stream.py` | `src/accelerate/data_loader.py` | Modified |

### MMCV

- Repository: <https://github.com/open-mmlab/mmcv>
- Fixed commit: `ccdc61c0878d27ac7cccfecd7b474320817f0bbf` (v1.4.3)
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/operators/modulated_deform_conv.py` | `mmcv/ops/modulated_deform_conv.py` | Modified |
| `turbo_physai/operators/multi_scale_deformable_attention.py` | `mmcv/ops/multi_scale_deform_attn.py` | Modified |
| `turbo_physai/optimizations/models/bevformer/mdc.py` | `mmcv/ops/modulated_deform_conv.py` | Modified |
| `test/msda_reference.py` | `mmcv/ops/multi_scale_deform_attn.py` | Modified |

### GR00T optimization sources

| Local file | Direct upstream source | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/groot/qwen_position.py` | [Transformers](https://github.com/huggingface/transformers) v4.57.3, commit `47b0e478f324b54f177ea7998a0791870fdd0324`, `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py` (`Qwen3VLVisionModel.fast_pos_embed_interpolate` and `rot_pos_emb`) | Modified |
| `turbo_physai/optimizations/models/groot/zero2_allgather.py` | [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) v0.18.4, commit `b35d9eb01bc04e774cc05dc43713f2a41423da5c`, `deepspeed/runtime/zero/stage_1_and_2.py` (`DeepSpeedZeroOptimizer.step`) | Modified |
| `turbo_physai/optimizations/models/groot/zero2_overlap.py` | DeepSpeed v0.18.4, commit `b35d9eb01bc04e774cc05dc43713f2a41423da5c`, `deepspeed/runtime/zero/stage_1_and_2.py` (`DeepSpeedZeroOptimizer.reduce_independent_p_g_buckets_and_remove_grads` and `average_tensor`) | Modified |
| `turbo_physai/optimizations/models/groot/vl_padding_dropout.py` | [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), commit `51d4c89f72fda44cbf77285c6a8114b52676b8a1`, `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` and `gr00t/model/modules/dit.py` | Modified |

These sources use Apache-2.0. The DeepSpeed version identifies the public API baseline. The integration is also checked against a DTK vendor build, whose separate source revision is not recorded here.
### DreamZero (Alibaba Wan DiT / VAE / text encoder / policy head)

- Repository: <https://github.com/dreamzero0/dreamzero>
- Fixed commit: `ab790c198fbce33503358efbbd4187ce9a89adf3`
- License: Apache-2.0 (DreamZero's own `LICENSE`, Copyright 2025 NVIDIA Corporation). DreamZero's `groot/vla/model/dreamzero/modules/attention.py` additionally carries an in-file `Copyright 2024-2025 The Alibaba Wan Team Authors.` notice inherited from Wan2.1.

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/dreamzero/attention.py` | `groot/vla/model/dreamzero/modules/attention.py` (`flash_attention`) | Modified |
| `turbo_physai/optimizations/models/dreamzero/CausalWanAttentionBlock.py` | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` (`CausalWanAttentionBlock.__init__`, `CausalWanAttentionBlock.forward`) | Modified |
| `turbo_physai/optimizations/models/dreamzero/CausalWanModel.py` | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` (`CausalWanModel.__init__`, `CausalWanModel.init_weights`, `CausalWanModel._forward_train`) | Modified |
| `turbo_physai/optimizations/models/dreamzero/CausalWanSelfAttention.py` | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` (`CausalWanSelfAttention.forward`, `CausalWanSelfAttention._process_state_blocks`) | Modified |
| `turbo_physai/optimizations/models/dreamzero/wan_video_vae.py` | `groot/vla/model/dreamzero/modules/wan_video_vae.py` (`Resample.forward`, `AttentionBlock.forward`, `WanVideoVAE.encode`, `WanVideoVAE38.__init__`, `Encoder3d.forward`, `Encoder3d_38.forward`) | Modified |
| `turbo_physai/optimizations/models/dreamzero/skip_unused_vae.py` | `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` (`WANPolicyHead.encode_image`, `WANPolicyHead.forward`) | Modified |
| `turbo_physai/optimizations/models/dreamzero/text_encoder.py` | `groot/vla/model/dreamzero/modules/wan_video_text_encoder.py` (`T5Attention.forward`, `WanTextEncoder.forward`) | Modified |
| `turbo_physai/optimizations/models/dreamzero/train_mode_cache.py` | `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` (`WANPolicyHead.train`, `WANPolicyHead.set_frozen_modules_to_eval_mode`) | Modified |

The eight local files listed above currently carry no Hygon modification notice.

The full Apache-2.0 text is in [third_party/licenses/Apache-2.0.txt](third_party/licenses/Apache-2.0.txt).

### LingBot-VLA-V2

- Repository: <https://github.com/Robbyant/lingbot-vla-v2>
- Fixed commit: `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/lingbot_vla_v2/profiling.py` | `lingbotvla/utils/helper.py` | Modified |
| `turbo_physai/optimizations/models/lingbot_vla_v2/fast_hf_query.py` | `lingbotvla/data/vla_data/base_dataset.py` | Modified |
| `turbo_physai/optimizations/models/lingbot_vla_v2/muon.py` | `lingbotvla/optim/muon.py` | Modified |
| `turbo_physai/optimizations/models/lingbot_vla_v2/muon_batched_ns.py` | `lingbotvla/optim/muon.py` | Modified |
| `turbo_physai/optimizations/models/lingbot_vla_v2/muon_gather.py` | `lingbotvla/optim/muon.py` | Modified |
| `turbo_physai/optimizations/models/lingbot_vla_v2/decoder_memory.py` | `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py` (`QwenvlWithExpertV2Model.forward`) | Modified |

### Hugging Face transformers

- Repository: <https://github.com/huggingface/transformers>
- Fixed commit: `8ac2b916b042b1f78b75c9eb941c0f5d2cdd8e10` (v4.57.0)
- License: Apache-2.0

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/lingbot_vla_v2/qwen3vl_rope_index_cpu.py` | `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py` (`Qwen3VLModel.get_rope_index`) | Modified |
| `turbo_physai/optimizations/models/lingbot_vla_v2/qwen3vl_cache_gridTHW.py` | `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py` (`Qwen3VLVisionModel.fast_pos_embed_interpolate`) | Modified |

## Confirmed BSD-3-Clause derivations

| Component | turbo_physai paths | Upstream repository and fixed commit | Upstream paths | License | NOTICE |
| --- | --- | --- | --- | --- | --- |
| PyTorch | `turbo_physai/operators/{grid_sample,upsample_bilinear_2d}.py`; `kernel/grid_sample/{GridSampler.cu,GridSampler.cuh}`; `kernel/upsample_bilinear_2d/UpSampleBilinear2d.cu` | <https://github.com/pytorch/pytorch> @ `e2d141dbde55c2a4370fac5165b0561b6af4798b` (v2.7.1) | `torch/nn/functional.py` (`grid_sample`, `interpolate`); `aten/src/ATen/native/cuda/{GridSampler.cu,GridSampler.cuh,UpSampleBilinear2d.cu}` | BSD-3-Clause | No `NOTICE` file identified at the fixed baseline. |
| PyTorch | `kernel/common/utils.h` | <https://github.com/pytorch/pytorch> @ `49444c3e546bf240bed24a101e747422d1f8a0ee` (v1.13.1) | `aten/src/ATen/native/UpSample.h` | BSD-3-Clause | No `NOTICE` file identified at the fixed baseline. |
| PyTorch | `turbo_physai/optimizations/models/lingbot_va/flex_assets/implementation.py` | <https://github.com/pytorch/pytorch> @ `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` (v2.10.0) | `torch/_inductor/kernel/flex/flex_attention.py` | BSD-3-Clause | [Upstream NOTICE](third_party/notices/PyTorch-2.10-NOTICE.txt) retained. |
| PyTorch | `turbo_physai/optimizations/models/lingbot_va/flex_assets/templates/common.py.jinja` | <https://github.com/pytorch/pytorch> @ `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` (v2.10.0) | `torch/_inductor/kernel/flex/templates/common.py.jinja` | BSD-3-Clause | [Upstream NOTICE](third_party/notices/PyTorch-2.10-NOTICE.txt) retained. |
| PyTorch | `turbo_physai/optimizations/models/lingbot_va/flex_assets/templates/flex_attention.py.jinja` | <https://github.com/pytorch/pytorch> @ `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` (v2.10.0) | `torch/_inductor/kernel/flex/templates/flex_attention.py.jinja` | BSD-3-Clause | [Upstream NOTICE](third_party/notices/PyTorch-2.10-NOTICE.txt) retained. |
| PyTorch | `turbo_physai/optimizations/models/lingbot_va/flex_assets/templates/flex_backwards.py.jinja` | <https://github.com/pytorch/pytorch> @ `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` (v2.10.0) | `torch/_inductor/kernel/flex/templates/flex_backwards.py.jinja` | BSD-3-Clause | [Upstream NOTICE](third_party/notices/PyTorch-2.10-NOTICE.txt) retained. |
| PyTorch | `turbo_physai/optimizations/models/lingbot_va/flex_assets/templates/utilities.py.jinja` | <https://github.com/pytorch/pytorch> @ `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` (v2.10.0) | `torch/_inductor/kernel/flex/templates/utilities.py.jinja` | BSD-3-Clause | [Upstream NOTICE](third_party/notices/PyTorch-2.10-NOTICE.txt) retained. |

The complete PyTorch v2.7.1 license text is in [third_party/licenses/PyTorch-BSD-3-Clause.txt](third_party/licenses/PyTorch-BSD-3-Clause.txt). `UpSampleBilinear2d.cu` also retains its inherited Caffe contributor attribution.

The LingBot-VA Flex assets above are modified derivatives of PyTorch v2.10.0. The five original source files match the official `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` baseline byte-for-byte; their SHA256 values are recorded in `flex_assets/manifest.json`. The HCU package build revision recorded there is used for runtime compatibility checking and is separate from this public upstream source revision. The v2.10.0 [license](third_party/licenses/PyTorch-2.10-BSD-3-Clause.txt) and [NOTICE](third_party/notices/PyTorch-2.10-NOTICE.txt) are retained verbatim.

## Confirmed MIT derivations

### DiT4DiT

- Repository: <https://github.com/Mondo-Robotics/DiT4DiT>
- Fixed commit: `d8854a11713f846b8b3f768fc3bf2c6cba2bc17f`
- License: MIT

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/dit4dit/cosmos_cache.py` | `DiT4DiT/model/modules/vlm/Cosmos25.py` | Modified |
| `turbo_physai/optimizations/models/dit4dit/cosmos_forward.py` | `DiT4DiT/model/modules/vlm/Cosmos25.py` | Modified |
| `turbo_physai/optimizations/models/dit4dit/dataloader.py` | `DiT4DiT/dataloader/__init__.py`; `DiT4DiT/training/train.py` | Modified |
| `turbo_physai/optimizations/models/dit4dit/training.py` | `DiT4DiT/training/train.py` | Modified |
| `turbo_physai/optimizations/models/dit4dit/video_pack.py` | `DiT4DiT/model/modules/vlm/Cosmos25.py` | Modified |

The complete DiT4DiT MIT license text is in [third_party/licenses/DiT4DiT-MIT.txt](third_party/licenses/DiT4DiT-MIT.txt).

| Component | turbo_physai paths | Upstream repository and fixed commit | Upstream paths | License | NOTICE |
| --- | --- | --- | --- | --- | --- |
| MoGe | `turbo_physai/optimizations/models/lingbot_vla_v2/moge_depth_only.py` | <https://github.com/microsoft/moge> @ `b942f00bdc2a2a23ebb474fbe034d487e6dcceec`, vendored in <https://github.com/Robbyant/lingbot-vla-v2> @ `be969b8fd117fb70550c5d4bf4bc328211b5b1b6` | `moge/model/v2.py` (`MoGeModel.forward`) | MIT | No `NOTICE` file identified at the fixed baseline. |
| Sparse4D | `turbo_physai/operators/deformable_aggregation.py`; `kernel/deformable_aggregation/DeformableAggregation.cu` | <https://github.com/HorizonRobotics/Sparse4D> @ `249ffbb695f4e9db628d953e2bf6d36de04bbb69` | `projects/mmdet3d_plugin/ops/deformable_aggregation.py`; `projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu` | MIT | No `NOTICE` file identified at the fixed baseline. |

The complete MoGe license text is in [third_party/licenses/MoGe-MIT.txt](third_party/licenses/MoGe-MIT.txt).

The complete Sparse4D license text is in [third_party/licenses/Sparse4D-MIT.txt](third_party/licenses/Sparse4D-MIT.txt).
### Sparse4D

- Repository: <https://github.com/HorizonRobotics/Sparse4D>
- Fixed commit: `249ffbb695f4e9db628d953e2bf6d36de04bbb69`
- License: MIT

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/operators/deformable_aggregation.py` | `projects/mmdet3d_plugin/ops/deformable_aggregation.py` | Modified |
| `kernel/deformable_aggregation/DeformableAggregation.cu` | `projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu` | Modified |
| `model_examples/Sparse4D/run.sh` | `tools/dist_train.sh` | Modified |
| `model_examples/Sparse4D/sparse4dv4_bs18.py` | `projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py` | Modified |
| `turbo_physai/optimizations/models/sparse4d/blocks.py` | `projects/mmdet3d_plugin/models/blocks.py` | Modified |
| `turbo_physai/optimizations/models/sparse4d/channels_last.py` | `projects/mmdet3d_plugin/models/sparse4d.py`; `projects/mmdet3d_plugin/ops/__init__.py` | Modified |
| `turbo_physai/optimizations/models/sparse4d/detection3d_blocks.py` | `projects/mmdet3d_plugin/models/detection3d/detection3d_blocks.py` | Modified |
| `turbo_physai/optimizations/models/sparse4d/grid_mask.py` | `projects/mmdet3d_plugin/models/grid_mask.py` | Modified |
| `turbo_physai/optimizations/models/sparse4d/instance_bank.py` | `projects/mmdet3d_plugin/models/instance_bank.py` | Modified |
| `turbo_physai/optimizations/models/sparse4d/sparse4d_head.py` | `projects/mmdet3d_plugin/models/sparse4d_head.py`; `projects/mmdet3d_plugin/models/blocks.py` | Modified |
| `turbo_physai/optimizations/models/sparse4d/target.py` | `projects/mmdet3d_plugin/models/detection3d/target.py` | Modified |

No `NOTICE` file was present in the listed upstream baseline. The complete Sparse4D license text is in [third_party/licenses/Sparse4D-MIT.txt](third_party/licenses/Sparse4D-MIT.txt).

### MapTRv2 / MapTR

- Repository: <https://github.com/hustvl/MapTR>
- Fixed commit: `e03f097abef19e1ba3fed5f471a8d80fbfa0a064` (`maptrv2` branch)
- Direct-source repository license: MIT, copyright (c) 2022 Hust Vision Lab

The files below adapt code from the fixed MapTR baseline. All paths in the
upstream column are relative to `projects/mmdet3d_plugin/`. Existing OpenMMLab
and BEVFormer attribution is retained. In particular, `data.py` and
`grid_mask.py` retain their Apache-2.0 notices; the direct repository's MIT
license does not replace inherited notices. No `NOTICE` file was found at
the fixed MapTR baseline.

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/maptrv2/assigner.py` | `maptr/assigners/maptr_assigner.py`; `maptr/dense_heads/maptrv2_head.py` | Modified |
| `turbo_physai/optimizations/models/maptrv2/bev_pool_fix.py` | `maptr/modules/encoder.py` | Modified |
| `turbo_physai/optimizations/models/maptrv2/reference.py` | `maptr/modules/encoder.py`; `maptr/modules/transformer.py`; `maptr/dense_heads/maptrv2_head.py` | Modified |
| `turbo_physai/optimizations/models/maptrv2/pv_mask.py` | `datasets/nuscenes_offlinemap_dataset.py` | Modified |
| `turbo_physai/optimizations/models/maptrv2/eval_compat.py` | `datasets/map_utils/tpfp_chamfer.py` | Modified |
| `turbo_physai/optimizations/models/maptrv2/match_cost.py` | `maptr/losses/map_loss.py` | Modified |
| `turbo_physai/optimizations/models/maptrv2/dataset_vectorization.py` | `datasets/nuscenes_offlinemap_dataset.py` | Modified |
| `turbo_physai/optimizations/models/maptrv2/sync_elimination.py` | `maptr/dense_heads/maptrv2_head.py`; `maptr/dense_heads/maptr_head.py`; `bevformer/dense_heads/bevformer_head.py`; `bevformer/modules/encoder.py` | Modified |
| `turbo_physai/optimizations/models/maptrv2/data.py` | `datasets/builder.py` | Modified |
| `turbo_physai/optimizations/models/maptrv2/grid_mask.py` | `models/utils/grid_mask.py` | Modified |

The complete MapTR license text is in [third_party/licenses/MapTR-MIT.txt](third_party/licenses/MapTR-MIT.txt).

## Confirmed third-party dependencies

The dependency below is installed from its published package; its source code is not copied into this repository.

| Dependency | Version | Upstream project | License | Declared in | Use |
| --- | --- | --- | --- | --- | --- |
| Pillow | `12.3.0` | <https://github.com/python-pillow/Pillow/tree/12.3.0> | MIT-CMU | `requirements.txt`; `docker/ci/Dockerfile` | Image-mask generation in the BEVFormer GridMask optimization. |

The complete Pillow 12.3.0 license text is in [third_party/licenses/Pillow-MIT-CMU.txt](third_party/licenses/Pillow-MIT-CMU.txt).

## Confirmed Boost Software License 1.0 material

| Component | turbo_physai path | Direct distribution source and fixed commit | Upstream path | License | NOTICE |
| --- | --- | --- | --- | --- | --- |
| cxx-prettyprint | `kernel/sparse_conv/include/prettyprint.h` | <https://github.com/mit-han-lab/bevfusion> @ `326653dc06e0938edf1aae7d01efcd158ba83de5` | `mmdet3d/ops/spconv/include/prettyprint.h` | Boost-1.0 | No `NOTICE` file at the fixed baseline. |

The complete Boost Software License 1.0 text is in [third_party/licenses/Boost-1.0.txt](third_party/licenses/Boost-1.0.txt). The source file retains Louis Delacroix's original copyright and license notice.

### LingBot-VA

- Repository: <https://github.com/Robbyant/lingbot-va>
- Fixed commit: `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`
- License: Apache-2.0; see [`third_party/licenses/LingBot-VA-Apache-2.0.txt`](third_party/licenses/LingBot-VA-Apache-2.0.txt).

| Local file | Upstream file | HYGON modification |
| --- | --- | --- |
| `turbo_physai/optimizations/models/lingbot_va/timestep.py` | `wan_va/modules/model.py` (`WanTransformer3DModel._time_embed`) | Modified |
| `turbo_physai/optimizations/models/lingbot_va/rotary.py` | `wan_va/modules/model.py` (`WanAttention.forward`) | Modified |
| `turbo_physai/optimizations/models/lingbot_va/pointwise.py` | `wan_va/modules/model.py` (`WanTransformerBlock.forward`) | Modified |
| `turbo_physai/optimizations/models/lingbot_va/fsdp.py` | `wan_va/distributed/fsdp.py` | Modified |
| `turbo_physai/optimizations/models/lingbot_va/train_sync.py` | `wan_va/train.py` (`Trainer.train`) | Modified |

The Group catalog, compatibility checks, and guarded Flex template adapter are TurboPhysAI integration code. The PyTorch-derived Flex implementation and four templates are registered in the BSD-3-Clause table above and retain their upstream source and license notices.

## Provenance policy for future additions

Before public release, every newly introduced third-party or mixed-source file must be resolved to an upstream path and 40-character commit and added here.

Any newly introduced third-party or mixed source must be added to this file before public release.

## Hygon modification notice

Where Apache-2.0 source code was modified, the source file retains the upstream copyright and license notice, then adds Hygon copyright and `Modified by Hygon.`. The notice does not change the upstream license.

## Cosmos3 (OpenMDW-1.1)

- Repository: <https://github.com/NVIDIA/cosmos-framework>
- Fixed commit: `9726697a83315540c6885baefd2fe353d9c74920`
- License: OpenMDW-1.1

`model_examples/Cosmos3/cosmos-framework.patch` contains modifications derived from NVIDIA Cosmos Framework. Cosmos-derived material retains OpenMDW-1.1 and applicable source-file notices.

| Local patch | Upstream file / Added file | Change type |
| --- | --- | --- |
| `model_examples/Cosmos3/cosmos-framework.patch` | `.gitattributes` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/__init__.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/callbacks/device_monitor.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/callbacks/grad_clip.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/checkpoint/dcp.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/checkpoint/load_only.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/configs/base/defaults/parallelism.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/configs/base/experiment/sft/vision_sft_nano.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/configs/base/reasoner/experiment/videophy2_dataflow_roles.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/configs/base/reasoner/experiment/videophy2_sft_edge.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/configs/base/reasoner/experiment/videophy2_sft_nano.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/configs/toml_config/sft_config.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/configs/toml_config/sft_config_test.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/configs/toml_config/toml_config_helper.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/data/generator/action/datasets/action_sft_dataset.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/data/generator/augmentors/interleaved_video_parsing.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/data/generator/augmentors/video_parsing.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/data/generator/dataflow/loader.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/data/generator/joint_dataloader.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/data/generator/reasoner/video_decoder_qwen.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/data/generator/sequence_packing/runtime.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/inference/args.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/inference/common/args.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/inference/common/config.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/inference/common/inference.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/inference/common/inference_test.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/inference/inference.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/inference/model.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/attention/backends.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/attention/flash2/__init__.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/attention/flash2/checks.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/attention/flash2/functions.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/attention/utils/__init__.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/algorithm/loss/cross_entropy.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/hf_model.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/mot/attention.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/mot/unified_mot.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/omni_mot_model.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/parallelize_vlm.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/reasoner/qwen3_vl/qwen3_vl.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/reasoner/qwen3_vl/utils.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/reasoner/qwen3_vl_moe/qwen3_vl_moe.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/tokenizers/fused_ops/__init__.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/tokenizers/fused_ops/_concat_layout.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/tokenizers/fused_ops/_hipdnn.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/tokenizers/fused_ops/concat_conv_bias.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/tokenizers/fused_ops/conv_bias.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/model/generator/vlm_model.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/scripts/convert_model_to_dcp.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/scripts/train.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/trainer/__init__.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/utils/distributed.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/utils/fsdp_ordered_comm.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/utils/generator/hf_attention_cosmos.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/utils/generator/qwen3_vl_hcu.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/utils/generator/torchcodec_video.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/utils/local_avae_cache.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/utils/memory_format.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/utils/memory_format_test.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/utils/misc.py` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `cosmos_framework/utils/video_decode.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `examples/_sft_launcher_common.sh` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `examples/prepare_droid_perf_subset.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `examples/summarize_sft_perf.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `examples/toml/sft_config/vision_sft_nano.toml` | Modified |
| `model_examples/Cosmos3/cosmos-framework.patch` | `inputs/omni/t2v_n4.json` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `tests/action_hcu/test_config.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `tests/action_hcu/test_dataloader.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `tests/action_hcu/test_decoder_cache.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `tests/action_hcu/test_perf_subset.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `tests/hcu/test_patch_review_regressions.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `tests/hcu/test_review_followups.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `tests/wan_vae_clean_fused_op_test.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `tests/wan_vae_concat_layout_test.py` | Added |
| `model_examples/Cosmos3/cosmos-framework.patch` | `tools/run_framework_inference_local_assets.py` | Added |

The added `cosmos_framework/utils/generator/qwen3_vl_hcu.py` derives its vision
forward methods from [Transformers Qwen3-VL](https://github.com/huggingface/transformers/blob/7dacdf4543ca4fe081f99dedacdb70d8df0fd366/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
at commit `7dacdf4543ca4fe081f99dedacdb70d8df0fd366` (v4.57.6), licensed under Apache-2.0.
It retains the Qwen Team and Hugging Face copyright and Apache-2.0 notice, with
a Hygon modification notice. The license text is in
[Apache-2.0](third_party/licenses/Apache-2.0.txt). This file does not replace
the Transformers model or AutoClass registrations.
