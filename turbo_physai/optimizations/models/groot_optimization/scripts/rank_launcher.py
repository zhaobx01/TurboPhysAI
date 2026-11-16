"""Rank-local Trainer instrumentation; installed after PhysAI's injection."""
import json
import os
import runpy
import time
from functools import wraps
from pathlib import Path

import torch
from transformers import TrainerCallback


class StepMetricsCallback(TrainerCallback):
    def __init__(self, output):
        self.output = Path(output)
        self.steps = []
        self.last = None

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.last = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        self.steps.append(now - self.last)
        self.last = now

    def on_train_end(self, args, state, control, **kwargs):
        self.output.mkdir(parents=True, exist_ok=True)
        data = {'rank': args.process_index, 'world_size': args.world_size, 'step_seconds': self.steps}
        if torch.cuda.is_available():
            data.update(max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                        max_memory_reserved_bytes=torch.cuda.max_memory_reserved())
        (self.output / f'rank-{args.process_index:02d}.json').write_text(json.dumps(data, indent=2) + '\n')


class TraceCallback(TrainerCallback):
    def __init__(self, output, rank, activities=None):
        self.output = Path(output)
        self.rank = rank
        self.activities = activities or [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        self.profiler = None
        self.stopped = False
        self.exported = False

    def _export(self, profiler):
        profiler.export_chrome_trace(str(self.output / f'trace_rank_{self.rank}_iter_35.json'))
        if self.rank == 0:
            (self.output / 'rank0-key-averages.txt').write_text(profiler.key_averages().table(sort_by='self_cpu_time_total', row_limit=200))
        self.exported = True

    def on_train_begin(self, args, state, control, **kwargs):
        self.output.mkdir(parents=True, exist_ok=True)
        self.profiler = torch.profiler.profile(
            activities=self.activities,
            schedule=torch.profiler.schedule(skip_first=28, wait=1, warmup=5, active=1, repeat=1),
            on_trace_ready=self._export, with_stack=True, record_shapes=False, profile_memory=False)
        self.profiler.start()

    def on_step_end(self, args, state, control, **kwargs):
        if not self.stopped:
            self.profiler.step()
            if state.global_step == 35:
                self.profiler.stop()
                self.stopped = True
                if not self.exported:
                    raise RuntimeError('Profiler did not export step 35')
                (self.output / f'profiler-stopped-rank-{self.rank}.json').write_text(json.dumps(
                    {'rank': self.rank, 'step': 35, 'stopped_after_trace': True}) + '\n')

    def on_train_end(self, args, state, control, **kwargs):
        if not self.stopped:
            raise RuntimeError('Training ended before profiler stopped at step 35')

    def close(self):
        if self.profiler is not None and not self.stopped:
            self.profiler.stop()
            self.stopped = True


def main():
    from transformers.trainer_utils import SaveStrategy
    from gr00t.experiment.trainer import Gr00tTrainer

    original_train = Gr00tTrainer.train

    @wraps(original_train)
    def train(self, *args, **kwargs):
        self.args.save_strategy = SaveStrategy.NO
        root = Path(os.environ['PHYS_BENCH_ATTEMPT_DIR'])
        self.add_callback(StepMetricsCallback(root / 'step_metrics'))
        trace = None
        if os.environ.get('PHYS_BENCH_TRACE') == '1':
            trace = TraceCallback(root / 'profiling', int(os.environ.get('RANK', '0')))
            self.add_callback(trace)
        print('[benchmark] save_strategy=no, rank timing enabled', flush=True)
        try:
            return original_train(self, *args, **kwargs)
        finally:
            if trace is not None:
                trace.close()

    def no_final_save(self, *args, **kwargs):
        print('[benchmark] final save_model skipped', flush=True)

    Gr00tTrainer.train = train
    Gr00tTrainer.save_model = no_final_save
    runpy.run_path(os.path.join(os.environ['GROOT_MODEL'], 'gr00t/experiment/launch_finetune.py'), run_name='__main__')


if __name__ == '__main__':
    main()
