import argparse
import sys
from argparse import Namespace
from pathlib import Path
from .logger import logger
from .reconstruct_runner import (
    DFR_LOG_NAME,
    POSTPROCESS_LOG_NAME,
    child_env,
    is_coco_dfr_reconstruct,
    run_reconstruct_commands,
)


def parse_argument():
    parser = argparse.ArgumentParser(description = 'CryoSieve: a particle sorting and sieving software for single particle analysis in cryo-EM')
    parser.add_argument('--reconstruct_software', type = str,   required = True,  help = 'command for reconstruction')
    parser.add_argument('--postprocess_software', type = str,   required = False, help = 'command for postprocessing')
    parser.add_argument('--i',                    type = str,   required = True,  help = 'input star file path')
    parser.add_argument('--o',                    type = str,   required = True,  help = 'output directory')
    parser.add_argument('--directory',            type = str,   required = False, help = 'directory of particles')
    parser.add_argument('--angpix',               type = float, required = False, help = 'pixelsize in Angstrom')
    parser.add_argument('--sym',                  type = str,   default  = 'C1',  help = 'molecular symmetry, C1 by default')
    parser.add_argument('--num_iters',            type = int,   default  = 10,    help = 'number of iterations for applying CryoSieve, 10 by default')
    parser.add_argument('--frequency_start',      type = float, default  = 50.,   help = 'starting threshold frquency, in Angstrom, 50 by default')
    parser.add_argument('--frequency_end',        type = float, default  = 3.,    help = 'ending threshold frquency, in Angstrom, 3 by default')
    parser.add_argument('--retention_ratio',      type = float, default  = 0.8,   help = 'fraction of retained particles in each iteration, 0.8 by default')
    parser.add_argument('--mask',                 type = str,   required = True,  help = 'mask file path')
    parser.add_argument('--balance',              action = 'store_true',          help = 'randomly drop particles to make all subset into the same size')
    parser.add_argument('--iterative-gridding-correction', dest = 'iterative_gridding_correction', action = 'store_true', default = True, help = 'enable iterative gridding correction during reconstruction')
    parser.add_argument('--no-iterative-gridding-correction', dest = 'iterative_gridding_correction', action = 'store_false', help = 'disable iterative gridding correction during reconstruction')
    parser.add_argument('--num_gpus',             type = int,   default  = 1,     help = 'number of gpus to execute CryoSieve core program, 1 by default')
    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()


def main():
    args = parse_argument()
    if args.postprocess_software is not None:
        logger.warning('Argument `--postprocess_software` will be deprecated')

    from .utility import check_cupy
    check_cupy()

    import numpy as np
    from .ParticleDataset import ParticleDataset
    from .sieve import barrier, destroy_distributed, init_distributed
    from .utility import run_commands

    ctx = init_distributed()
    try:
        src = Path(args.i)
        if not src.exists():
            raise FileNotFoundError(f'{args.i} not found')
        elif src.suffix != '.star':
            raise ValueError(f'{args.i} is not a star file')

        dst = Path(args.o).absolute()
        if ctx.is_main:
            dst.mkdir(parents = True, exist_ok = True)
        barrier(ctx)
        if not dst.is_dir():
            raise ValueError(f'{args.o} is not a directory or cannot be created')

        gridding_correction_flag = '--iterative-gridding-correction' if args.iterative_gridding_correction else '--no-iterative-gridding-correction'
        gridding_correction_arg = gridding_correction_flag if is_coco_dfr_reconstruct(args.reconstruct_software) else ''

        dataset = ParticleDataset(src, args.directory, args.angpix)
        data_dir = dataset.data_dir.absolute()
        if ctx.is_main:
            logger.info(f'Initialize ParticleDataset with given directory {str(data_dir)}')
        if args.balance:
            dataset.balance()
        if ctx.is_main:
            dataset.save(dst / 'iter0.star')
        barrier(ctx)

        # go.
        frequences = 1 / np.linspace(1.0 / args.frequency_start, 1.0 / args.frequency_end, args.num_iters)
        overall_retention_ratio = 1.0
        for i in range(args.num_iters):
            if ctx.is_main:
                logger.info(f'Start iteration {i}, overall retaining ratio {overall_retention_ratio * 100:.2f}%, threshold frequency {frequences[i]:.2f} Angstrom')

            # reconstruct.
            commands = [
                ' '.join([
                    args.reconstruct_software,
                    f'--i "{str(dst / f"iter{i}.star")}"',
                    f'--o "{str(dst / f"iter{i}_half1.mrc")}"',
                    f'--angpix {args.angpix}',
                    f'--sym {args.sym}',
                    '--ctf true',
                    gridding_correction_arg,
                    '--subset 1',
                ]),
                ' '.join([
                    args.reconstruct_software,
                    f'--i "{str(dst / f"iter{i}.star")}"',
                    f'--o "{str(dst / f"iter{i}_half2.mrc")}"',
                    f'--angpix {args.angpix}',
                    f'--sym {args.sym}',
                    '--ctf true',
                    gridding_correction_arg,
                    '--subset 2',
                ])
            ]
            run_reconstruct_commands(
                commands,
                f'3D-reconstruction (iteration {i})',
                cwd = data_dir,
                output_dir = dst,
                ctx = ctx,
                reconstruct_software = args.reconstruct_software,
                barrier_func = barrier,
                iteration = i,
            )

            # postprocess.
            if ctx.is_main and args.postprocess_software is not None:
                pp_dir = dst / f'postprocess_iter{i}'
                pp_dir.mkdir(parents = True, exist_ok = True)
                command = ' '.join([
                    args.postprocess_software,
                    f'--mask "{args.mask}"',
                    f'--i "{str(dst / f"iter{i}_half1.mrc")}"',
                    f'--i2 "{str(dst / f"iter{i}_half2.mrc")}"',
                    f'--o "{str(pp_dir / f"iter{i}")}"',
                    f'--angpix {args.angpix}',
                    '--auto_bfac',
                    '--autob_lowres 10',
                    '--random_seed 0',
                ])
                run_commands(command, f'postprocess (iteration {i})', env = child_env(dst, POSTPROCESS_LOG_NAME))
            barrier(ctx)

            # sieve.
            from .core import process as process_core
            core_args = Namespace(
                i = str(dst / f'iter{i}.star'),
                o = str(dst / f'iter{i + 1}.star'),
                directory = str(data_dir) if args.directory is not None else None,
                angpix = args.angpix,
                volume = [str(dst / f'iter{i}_half1.mrc'), str(dst / f'iter{i}_half2.mrc')],
                mask = args.mask,
                retention_ratio = args.retention_ratio,
                frequency = float(f'{frequences[i]:.3f}'),
                num_gpus = args.num_gpus,
            )
            process_core(core_args, ctx)
            overall_retention_ratio *= args.retention_ratio

        if ctx.is_main:
            logger.info('Execute CryoSieve successfully')
    finally:
        destroy_distributed()


if __name__ == '__main__':
    main()
