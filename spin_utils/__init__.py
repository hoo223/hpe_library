import os, getpass
user = getpass.getuser()
os.chdir(f'/home/{user}/codes/SPIN')
import SPIN.config as spin_config
import SPIN.constants as spin_constants
from SPIN.models import SMPL
from SPIN.datasets import BaseDataset
group_config = ['spin_config']
group_constants = ['spin_constants']
group_models = ['SMPL']
group_datasets = ['BaseDataset']

__all__ = group_config + group_constants + group_models + group_datasets