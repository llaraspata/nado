import os
import time
import json
import resource
from contextlib import contextmanager

import torch



def model_stats(model, exclude_prefixes=()):
    """
    Parameter counts of a torch module, plus name/shape/count of each named parameter.
    Parameters whose name starts with one of exclude_prefixes are skipped.
    """
    params = [(name, p) for name, p in model.named_parameters() if not name.startswith(tuple(exclude_prefixes))]
    return {
        'num_params': sum(p.numel() for _, p in params),
        'num_trainable_params': sum(p.numel() for _, p in params if p.requires_grad),
        'parameters': [{'name': name, 'shape': list(p.shape), 'num_params': p.numel()} for name, p in params],
    }


class ResourceTracker(object):
    """
    Measure the resources used by a run: wall-clock and cpu time, peak cpu memory and, on cuda, peak gpu memory.
    Wrap each part of the run in `with tracker.phase(name):` (or start_phase/end_phase) to also get its time and peak gpu memory.
    """
    def __init__(self, device):
        self.use_cuda = str(device).startswith('cuda') and torch.cuda.is_available()
        self.phases = {}
        self.gpu_peak_allocated = 0
        self.gpu_peak_reserved = 0
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats()
        self.start_wall = time.perf_counter()
        self.start_cpu = time.process_time()

    def _sync(self):
        if self.use_cuda:
            torch.cuda.synchronize() # cuda calls are async, wait for them so they're timed in the right phase

    def _update_gpu_peak(self):
        self.gpu_peak_allocated = max(self.gpu_peak_allocated, torch.cuda.max_memory_allocated())
        self.gpu_peak_reserved = max(self.gpu_peak_reserved, torch.cuda.max_memory_reserved())

    def start_phase(self, name):
        self._sync()
        if self.use_cuda:
            self._update_gpu_peak()
            torch.cuda.reset_peak_memory_stats() # so the peak below is this phase's
        self._phase_name = name
        self._phase_start = time.perf_counter()

    def end_phase(self):
        self._sync()
        stats = {'time_s': time.perf_counter() - self._phase_start}
        if self.use_cuda:
            stats['gpu_peak_allocated_mb'] = torch.cuda.max_memory_allocated() / 2**20
            stats['gpu_peak_reserved_mb'] = torch.cuda.max_memory_reserved() / 2**20
            self._update_gpu_peak()
        self.phases[self._phase_name] = stats

    @contextmanager
    def phase(self, name):
        self.start_phase(name)
        try:
            yield
        finally:
            self.end_phase()

    def summary(self, num_examples=None):
        self._sync()
        summary = {
            'total_time_s': time.perf_counter() - self.start_wall,
            'cpu_time_s': time.process_time() - self.start_cpu,
            'cpu_peak_memory_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10, # ru_maxrss is in KB on linux
            'phases': self.phases,
        }
        if num_examples is not None:
            summary['num_examples'] = num_examples
            if num_examples > 0 and 'generation' in self.phases:
                summary['generation_time_per_example_s'] = self.phases['generation']['time_s'] / num_examples
        if self.use_cuda:
            self._update_gpu_peak()
            device = torch.cuda.current_device()
            summary['gpu'] = {
                'name': torch.cuda.get_device_name(device),
                'total_memory_mb': torch.cuda.get_device_properties(device).total_memory / 2**20,
                'peak_allocated_mb': self.gpu_peak_allocated / 2**20,
                'peak_reserved_mb': self.gpu_peak_reserved / 2**20,
            }
        else:
            summary['gpu'] = None
        return summary


def default_stats_file(task, run_name):
    """
    Run-specific default path for save_model_stats: logs/model_stats/<task>_<run_name>_<timestamp>.json
    """
    return os.path.join('logs', 'model_stats', '{}_{}_{}.json'.format(task, run_name, time.strftime('%Y%m%d-%H%M%S')))


def save_model_stats(save_path, model_string, model_path, language_model, conditioning_models, resources=None, lm_exclude_prefixes=()):
    """
    Save a json file describing the models involved in a run:
    the base language model (name, parameter count, skipping parameters under lm_exclude_prefixes)
    and the conditioning model(s), as a dict name -> dict(ckpt=..., task=..., epoch=..., model=torch module).
    resources is optionally a ResourceTracker summary.
    """
    stats = {
        'language_model': {
            'name': model_string,
            'model_path': model_path,
            'class': type(language_model).__name__,
            'num_params': model_stats(language_model, lm_exclude_prefixes)['num_params'],
        },
        'conditioning_models': {},
    }
    for name, info in conditioning_models.items():
        stats['conditioning_models'][name] = {
            'ckpt': info.get('ckpt'),
            'task': info.get('task'),
            'epoch': info.get('epoch'),
            **model_stats(info['model']),
        }
    if resources is not None:
        stats['resources'] = resources
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, 'w') as wf:
        json.dump(stats, wf, indent=2)
