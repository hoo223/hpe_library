# torch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
lib_torch = ['torch', 'nn', 'F', 'optim', 'SummaryWriter', 'Dataset', 'DataLoader', 'DDP']

# standard library
import os, sys, pickle, time, argparse, ast, logging, copy, json, shutil, getpass, csv, math
from math import radians, degrees, isnan, sin, cos, sqrt, floor
from glob import glob
from natsort import natsorted
lib_standard = ['os', 'sys', 'pickle', 'time', 'argparse', 'ast', 'logging', 'copy', 'json', 'shutil', 'radians', 'degrees', 'isnan', 'sin', 'cos', 'sqrt', 'getpass', 'csv', 'glob', 'floor', 'natsorted', 'math']

# third party library
import numpy as np
import cv2
import random
import matplotlib
import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from ipywidgets import interact
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from scipy.spatial import distance
from scipy.linalg import null_space
import pandas as pd
lib_third_party = ['np', 'cv2', 'matplotlib', 'plt', 'gridspec', 'Axes3D', 'interact', 'tqdm', 'Rotation', 'distance', 'go', 'pd', 'random', 'null_space', 'FigureCanvas', 'imageio']

# camera-models library
sys.path.append('/home/hrai/codes/camera-models/')
#os.chdir('/home/hrai/codes/camera-models/')
from camera_models import draw3d_arrow, get_calibration_matrix, get_plane_from_three_points, get_plucker_matrix, get_projection_matrix, get_rotation_matrix, set_xyzlim3d, set_xyzticks, to_homogeneus, to_inhomogeneus, GenericPoint, Image, ImagePlane, Polygon, PrincipalAxis, ReferenceFrame
lib_camera_models = ["GenericPoint", "Image", "ImagePlane", "Polygon", "PrincipalAxis", "ReferenceFrame", "draw3d_arrow", "get_calibration_matrix", "get_plane_from_three_points", "get_plucker_matrix", "get_projection_matrix", "get_rotation_matrix", "to_homogeneus", "to_inhomogeneus", "set_xyzlim3d", "set_xyzticks"]

# AdaptPose library
# sys.path.append('/home/hrai/codes/AdaptPose')
# #os.chdir('/home/hrai/codes/AdaptPose')
# from common.h36m_dataset import Human36mDataset, TEST_SUBJECTS
# from utils.data_utils import read_3d_data, create_2d_data
# lib_adapt_pose = ['Human36mDataset', 'TEST_SUBJECTS', 'read_3d_data', 'create_2d_data']

# PoSynDA library
sys.path.append('/home/hrai/codes/PoSynDA')
from common.h36m_dataset import Human36mDataset
lib_posynda = ['Human36mDataset']

lib_except_my_utils = lib_torch + lib_standard + lib_third_party + lib_camera_models + lib_posynda #+ lib_adapt_pose

os.chdir('/home/hrai/codes/hpe_library')
__all__ = lib_except_my_utils