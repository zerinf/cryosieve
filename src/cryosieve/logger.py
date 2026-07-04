import logging
import logging.config
import os
from pathlib import Path


def resolve_log_path(default_name):
    cryosieve_log_path = os.environ.get('COCO_CRYOSIEVE_LOG')
    job_dir = os.environ.get('COCO_JOB_DIR')
    job_log_path = os.environ.get('COCO_JOB_LOG')
    if cryosieve_log_path:
        path = Path(cryosieve_log_path)
    elif job_dir:
        path = Path(job_dir) / default_name
    elif job_log_path:
        path = Path(job_log_path)
    else:
        path = Path(default_name)
    if path.parent != Path('.'):
        path.parent.mkdir(parents = True, exist_ok = True)
    return str(path)


logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s][%(levelname)s] %(message)s',
            'datefmt': r'%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': resolve_log_path('cryosieve.log'),
            'mode': 'a',
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        'CryoSieve': {
            'level': 'INFO',
            'handlers': ['file'],
            'propagate': False
        }
    }
})
logger = logging.getLogger('CryoSieve')
