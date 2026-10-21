"""MapTRv2 optimization implementations.

The implementation of each validated optimization lives in the topic module
named after it, mirroring ``optimizations/models/bevformer`` and
``optimizations/models/bevfusion``:

* ``compile.py``     -- ``torch.compile`` / ``torch._dynamo`` wrappers
* ``data.py``        -- ``build_dataloader`` with pinned host memory
* ``grid_mask.py``   -- Dynamo-safe ``GridMask.forward``
* ``match_cost.py``  -- broadcast L1 cost instead of ``torch.cdist``
* ``pv_mask.py``     -- vectorized PV mask resampling, merged BEV transform
* ``training.py``    -- channels-last / cuDNN / start-method runtime recipe

``catalog.py`` binds those implementations to targets in the official
``maptrv2`` branch.
"""
