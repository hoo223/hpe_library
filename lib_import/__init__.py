## torch library
from sympy import im
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
lib_torch = ['torch', 'nn', 'F', 'optim', 'SummaryWriter', 'Dataset', 'DataLoader', 'DDP']

## standard library
import os, sys, pickle, time, argparse, ast, logging, copy, json, shutil, getpass, csv, math, random, threading, importlib, itertools
from glob import glob
from typing import Optional, Tuple, Dict, Any, List
user = getpass.getuser()
sys.path.append(f"/home/{user}/codes")

lib_standard = ['os', 'sys', 'pickle', 'time', 'argparse', 'ast', 'logging', 'copy', 'json', 'shutil', 'getpass', 'csv', 'math', 'random', 'glob', 'threading', 'importlib', 'itertools', 'Optional', 'Tuple', 'Dict', 'Any', 'List']
from math import radians, degrees, isnan, sin, cos, sqrt, floor
lib_math = ['radians', 'degrees', 'isnan', 'sin', 'cos', 'sqrt', 'floor']
from inspect import getmembers, isfunction, isclass, getsourcefile, isbuiltin
lib_inspect = ['getmembers', 'isfunction', 'isclass', 'getsourcefile', 'isbuiltin']

lib_standard = lib_standard + lib_math + lib_inspect
## third party library
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mpl_toolkits
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.axes._axes import Axes 
lib_matplotlib = ['matplotlib', 'plt', 'gridspec', 'Axes', 'Axes3D', 'mpl_toolkits', 'FigureCanvas', 'patches']

import scipy
from scipy.spatial.transform import Rotation
from scipy.spatial import distance
from scipy.linalg import null_space
lib_scipy = ['scipy', 'Rotation', 'distance', 'null_space']

from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output
from ipywidgets import GridspecLayout
from ipywidgets import TwoByTwoLayout
lib_ipywidgets = ['display', 'widgets', 'interact', 'interactive', 'fixed', 'interact_manual', 'interactive_output', 'GridspecLayout', 'TwoByTwoLayout']

import cv2
import yaml
import imageio
import numpy as np
from tqdm import tqdm
import pandas as pd
from natsort import natsorted
import plotly.graph_objects as go
lib_others = ['cv2', 'yaml', 'imageio', 'np', 'tqdm', 'pd', 'natsorted', 'go']

lib_third_party = lib_matplotlib + lib_scipy + lib_ipywidgets + lib_others 
## custom functions
def import_module_from_path(module_name, file_path):
    # 모듈의 spec 생성
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    # 모듈 객체 생성
    module = importlib.util.module_from_spec(spec)
    # 모듈에 spec 로드
    spec.loader.exec_module(module)
    # sys.modules에 모듈 추가
    sys.modules[module_name] = module
    return module

def generate_init_py(module_list, root, lib_root, utils_root, lib_except_my_utils, verbose=True):
    utils_rel_path = os.path.relpath(utils_root, os.getcwd())
    # set the init file and remove it if exists
    init_py_path = os.path.join(utils_rel_path, '__init__.py')
    print(f"init_py_path: {init_py_path}")
    if os.path.exists(init_py_path): os.remove(init_py_path)
    # check the related path
    print(f"lib_root: {lib_root}")
    lib_rel_root = os.path.relpath(lib_root, root)
    func_dict = {}
    with open(init_py_path, 'w') as f:
        print(f"Gnerating {init_py_path}")
        f.write("# https://passwd.tistory.com/entry/Python-initpy\n")
        f.write(f"# {lib_rel_root} library\n")
        for module in module_list:
            module_file_path = str(getsourcefile(module)) # get the file path of the module
            root_rel_path = os.path.dirname(os.path.relpath(module_file_path, root)) # relative module_file_path from root
            module_name = os.path.basename(module_file_path).split('.py')[0]

            cond0 = lambda x: 'built-in' not in str(sys.modules.get(x[1].__module__)) # do not include the built-in functions
            filtered = [o for o in getmembers(module, isfunction) if cond0(o)] + [o for o in getmembers(module, isclass) if cond0(o)]
            cond1 = lambda x: x[0] not in lib_except_my_utils # do not duplicate the functions in lib_except_my_utils
            cond2 = lambda x: module_name in str(getsourcefile(x[1])) # only the functions in current module
            cond_func = lambda x: cond1(x) and cond2(x)
            func_list = [o[0] for o in filtered if cond_func(o)] # get all the functions in the module
            #func_list += [o[0] for o in filtered if cond_func(o)] # get all the classes in the module

            txt = f"from {root_rel_path.replace('/', '.')}.{module_name} import " + ', '.join([f'{o}' for o in func_list])
            if verbose: print(txt)
            f.write(txt + '\n')
            func_dict[module_name] = func_list
            
        if verbose: print('')
        f.write('\n')
        lib_list = []
        for module in module_list:
            module_file_path = str(getsourcefile(module)) # get the file path of the module
            lib_rel_path = os.path.dirname(os.path.relpath(module_file_path, lib_root)) # relative module_file_path from lib_root
            module_name = os.path.basename(module_file_path).split('.py')[0]
            if len(lib_rel_path) == 0: module_group_name = f"group_{module_name}"
            else: module_group_name = f"group_{lib_rel_path.replace('/', '_')}_{module_name}"
            txt = f"{module_group_name} = {func_dict[module_name]}"
            if verbose: print(txt)
            lib_list.append(module_group_name) 
            f.write(txt + '\n')
        all_txt = f"\n__all__ = {' + '.join(lib_list)}" 
        if verbose: print(all_txt)
        f.write(all_txt)
        print(f"Done: {os.path.join(utils_root, '__init__.py')}")
        
functions = ['import_module_from_path', 'generate_init_py']

## Total library
lib_except_my_utils = lib_torch + lib_standard + lib_third_party + functions #+ lib_adapt_pose
__all__ = lib_except_my_utils # type: ignore