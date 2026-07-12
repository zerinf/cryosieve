from __future__ import annotations

import os
import shlex
from pathlib import Path
from .utility import run_commands


CRYOSIEVE_LOG_NAME = 'cryosieve.log'
DFR_LOG_NAME = 'copra_spa_3d_reconstruction_dfr.log'
POSTPROCESS_LOG_NAME = 'postprocess.log'
COCO_DFR_PARENT_DISTRIBUTED_ENV = 'COCO_DFR_USE_PARENT_DISTRIBUTED'

DISTRIBUTED_ENV_KEYS = (
    'LOCAL_RANK',
    'RANK',
    'WORLD_SIZE',
    'GROUP_RANK',
    'ROLE_RANK',
    'ROLE_WORLD_SIZE',
    'LOCAL_WORLD_SIZE',
    'MASTER_ADDR',
    'MASTER_PORT',
    'TORCHELASTIC_RUN_ID',
    'TORCHELASTIC_RESTART_COUNT',
    'TORCHELASTIC_MAX_RESTARTS',
)

PROGRESS_ENV_KEYS = (
    'COCO_PROGRESS_PATH',
    'COCO_PROGRESS_SOURCE',
    'COCO_PROGRESS_STAGE_ID',
    'COCO_PROGRESS_STEP_KEY',
    'COCO_PROGRESS_STEP_INDEX',
    'COCO_PROGRESS_STEP_TOTAL',
)

COCO_DFR_RECONSTRUCT_MODULES = {
    'copra_spa_3d_reconstruction_dfr.compat.cryosieve_reconstruct',
    'coconut_spa_3d_reconstruction_dfr.compat.cryosieve_reconstruct',
}


def job_log_path(output_dir, log_name):
    job_dir = os.environ.get('COCO_JOB_DIR')
    base = Path(job_dir).absolute() if job_dir else Path(output_dir)
    return base / log_name


def child_env(output_dir, job_log_name=None, cryosieve_log=False):
    env = os.environ.copy()
    for key in DISTRIBUTED_ENV_KEYS:
        env.pop(key, None)
    for key in PROGRESS_ENV_KEYS:
        env.pop(key, None)
    if job_log_name is None:
        env.pop('COCO_JOB_LOG', None)
    else:
        env['COCO_JOB_LOG'] = str(job_log_path(output_dir, job_log_name))
    if cryosieve_log:
        env['COCO_CRYOSIEVE_LOG'] = str(job_log_path(output_dir, CRYOSIEVE_LOG_NAME))
    return env


def is_coco_dfr_reconstruct(command):
    try:
        tokens = shlex.split(command or '')
    except ValueError:
        return False
    for index, token in enumerate(tokens[:-2]):
        executable = Path(token).name
        if executable not in {'python', 'python3'}:
            continue
        if tokens[index + 1] == '-m' and tokens[index + 2] in COCO_DFR_RECONSTRUCT_MODULES:
            return True
    return False


def _positive_env_int(*names):
    for name in names:
        value = os.environ.get(name)
        if value is None or str(value).strip() == '':
            continue
        parsed = int(str(value).strip())
        if parsed < 1:
            raise ValueError(f'{name} must be a positive integer')
        return parsed
    return None


def _child_master_port(half_map, iteration=None):
    base = _positive_env_int('COCO_MASTER_PORT', 'MASTER_PORT') or 29500
    half_map_value = int(half_map or 0)
    iteration_value = int(iteration or 0)
    port = base + 100 + iteration_value * 2 + half_map_value
    if port > 65535:
        port = 20000 + (port % 40000)
    return str(port)


def distributed_child_env(output_dir, job_log_name, half_map, iteration=None):
    env = os.environ.copy()
    for key in PROGRESS_ENV_KEYS:
        env.pop(key, None)
    if job_log_name is None:
        env.pop('COCO_JOB_LOG', None)
    else:
        env['COCO_JOB_LOG'] = str(job_log_path(output_dir, job_log_name))
    env[COCO_DFR_PARENT_DISTRIBUTED_ENV] = '1'
    env['MASTER_ADDR'] = env.get('COCO_MASTER_ADDR') or env.get('MASTER_ADDR') or '127.0.0.1'
    env['MASTER_PORT'] = _child_master_port(half_map, iteration)
    return env


def release_parent_gpu_memory_caches():
    try:
        import torch
        cuda = getattr(torch, 'cuda', None)
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()
            ipc_collect = getattr(cuda, 'ipc_collect', None)
            if ipc_collect is not None:
                ipc_collect()
    except Exception:
        pass
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _as_command_list(commands):
    if isinstance(commands, str):
        return [commands]
    return list(commands or [])


def run_reconstruct_commands(commands, jobname, cwd, output_dir, ctx, reconstruct_software, barrier_func, iteration=None):
    command_list = _as_command_list(commands)
    if ctx is not None and ctx.distributed and is_coco_dfr_reconstruct(reconstruct_software):
        if ctx.is_main:
            from .logger import logger
            logger.info('Run CoCo DFR reconstruction commands with parent distributed allocation')
        for index, command in enumerate(command_list, start=1):
            release_parent_gpu_memory_caches()
            run_commands(
                command,
                f'{jobname} half-map {index}',
                cwd=cwd,
                env=distributed_child_env(output_dir, DFR_LOG_NAME, index, iteration),
            )
            barrier_func(ctx)
        return

    if ctx is None or ctx.is_main:
        run_commands(command_list, jobname, cwd=cwd, env=child_env(output_dir, DFR_LOG_NAME))
    if barrier_func is not None:
        barrier_func(ctx)
