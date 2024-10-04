# https://passwd.tistory.com/entry/Python-initpy
# MotionBERT/lib library
from MotionBERT.lib.data.datareader_aihub import DataReaderAIHUB
from MotionBERT.lib.data.datareader_fit3d import DataReaderFIT3D
from MotionBERT.lib.data.datareader_h36m import DataReaderH36M
from MotionBERT.lib.data.datareader_poseaug_3dhp import DataReaderPOSEAUG3DHP
from MotionBERT.lib.data.datareader_mesh import DataReaderMesh
from MotionBERT.lib.data.datareader_3dhp import DataReader3DHP
from MotionBERT.lib.data.dataset_motion_2d import posetrack2h36m, InstaVDataset2D, PoseTrackDataset2D
from MotionBERT.lib.data.datareader_total import DataReaderTotal, DataReaderTotalGroup
from MotionBERT.lib.data.dataset_motion_3d import MotionDataset, MotionDataset3D, MotionDataset3DTotal
from MotionBERT.lib.data.datareader_random_limb import DataReaderRandomLimb
from MotionBERT.lib.data.augmentation import Augmenter2D, Augmenter3D
from MotionBERT.lib.data.dataset_mesh import MotionSMPL, SMPLDataset
from MotionBERT.lib.data.datareader_kookmin import DataReaderKOOKMIN
from MotionBERT.lib.data.dataset_action import coco2h36m, get_action_names, human_tracking, make_cam, random_move, ActionDataset, Fit3DAction, KookminAction, NTURGBD, NTURGBD1Shot
from MotionBERT.lib.data.dataset_wild import halpe2h36m, read_input, WildDetDataset
from MotionBERT.lib.utils.utils_data import crop_scale, crop_scale_3d, flip_data, resample, split_clips
from MotionBERT.lib.utils.utils_smpl import get_smpl_faces, SMPL
from MotionBERT.lib.utils.utils_mesh import batch_rodrigues, compute_error, compute_error_frames, estimate_translation, estimate_translation_np, evaluate_mesh, flip_thetas, flip_thetas_batch, quat2mat, quaternion_to_angle_axis, rectify_pose, rigid_align, rigid_transform_3D, rot6d_to_rotmat, rot6d_to_rotmat_spin, rotation_matrix_to_angle_axis, rotation_matrix_to_quaternion
from MotionBERT.lib.utils.tools import construct_include, ensure_dir, get_config, read_pkl, Loader, TextLogger
from MotionBERT.lib.utils.vismo import bounding_box, get_img_from_fig, hex2rgb, joints2image, motion2video, motion2video_3d, motion2video_mesh, pixel2world_vis, pixel2world_vis_motion, render_and_save, rgb2rgba, save_image, vis_data_batch
from MotionBERT.lib.utils.args import check_args, get_opt_args_from_model_name, get_opts_args, list_of_strings, parse_args
from MotionBERT.lib.utils.learning import accuracy, load_backbone, load_pretrained_weights, partial_train_layers, AverageMeter
from MotionBERT.lib.model.drop import drop_path, DropPath
from MotionBERT.lib.model.load_model import load_model
from MotionBERT.lib.model.evaluation import batch_inference_eval, calculate_eval_metric, calculate_eval_metric_canonicalization, evaluate, evaluate_onevec, inference_eval, preprocess_eval
from MotionBERT.lib.model.model_action import ActionHeadClassification, ActionHeadEmbed, ActionNet
from MotionBERT.lib.model.CanonDSTformer import _no_grad_trunc_normal_, trunc_normal_, Attention, Block, CanonDSTformer1, CanonDSTformer2, MLP
from MotionBERT.lib.model.load_dataset import load_dataset
from MotionBERT.lib.model.loss_supcon import SupConLoss
from MotionBERT.lib.model.model_mesh import MeshRegressor, SMPLRegressor
from MotionBERT.lib.model.DHDSTformer import DHDSTformer_limb, DHDSTformer_limb2, DHDSTformer_limb3, DHDSTformer_limb4, DHDSTformer_limb5, DHDSTformer_limb_all_in_one, DHDSTformer_onevec, DHDSTformer_right_arm, DHDSTformer_right_arm2, DHDSTformer_right_arm3, DHDSTformer_torso, DHDSTformer_torso2, DHDSTformer_torso_limb, DHDSTformer_total, DHDSTformer_total2, DHDSTformer_total3, DHDSTformer_total4, DHDSTformer_total5, DHDSTformer_total6, DHDSTformer_total7, DHDSTformer_total8, linear_head
from MotionBERT.lib.model.DSTformer import _no_grad_trunc_normal_, trunc_normal_, Attention, Block, DSTformer, MLP
from MotionBERT.lib.model.loss import get_angles, get_limb_lens, loss_2d_weighted, loss_angle, loss_angle_velocity, loss_joint, loss_limb_gt, loss_limb_var, loss_mpjpe, loss_symmetry, loss_velocity, mpjpe, mpjpe_for_each_joint, n_mpjpe, p_mpjpe, p_mpjpe_for_each_joint, weighted_bonelen_loss, weighted_boneratio_loss, weighted_mpjpe
from MotionBERT.lib.model.loss_mesh import MeshLoss
from MotionBERT.lib.model.training import generate_loss_dict, inference_train, preprocess_train, save_checkpoint, train, train_epoch, update_train_writer
from MotionBERT.lib.model.DHformer import _no_grad_trunc_normal_, trunc_normal_, Attention, Block, DHformer, MLP

group_data_datareader_aihub = ['DataReaderAIHUB']
group_data_datareader_fit3d = ['DataReaderFIT3D']
group_data_datareader_h36m = ['DataReaderH36M']
group_data_datareader_poseaug_3dhp = ['DataReaderPOSEAUG3DHP']
group_data_datareader_mesh = ['DataReaderMesh']
group_data_datareader_3dhp = ['DataReader3DHP']
group_data_dataset_motion_2d = ['posetrack2h36m', 'InstaVDataset2D', 'PoseTrackDataset2D']
group_data_datareader_total = ['DataReaderTotal', 'DataReaderTotalGroup']
group_data_dataset_motion_3d = ['MotionDataset', 'MotionDataset3D', 'MotionDataset3DTotal']
group_data_datareader_random_limb = ['DataReaderRandomLimb']
group_data_augmentation = ['Augmenter2D', 'Augmenter3D']
group_data_dataset_mesh = ['MotionSMPL', 'SMPLDataset']
group_data_datareader_kookmin = ['DataReaderKOOKMIN']
group_data_dataset_action = ['coco2h36m', 'get_action_names', 'human_tracking', 'make_cam', 'random_move', 'ActionDataset', 'Fit3DAction', 'KookminAction', 'NTURGBD', 'NTURGBD1Shot']
group_data_dataset_wild = ['halpe2h36m', 'read_input', 'WildDetDataset']
group_utils_utils_data = ['crop_scale', 'crop_scale_3d', 'flip_data', 'resample', 'split_clips']
group_utils_utils_smpl = ['get_smpl_faces', 'SMPL']
group_utils_utils_mesh = ['batch_rodrigues', 'compute_error', 'compute_error_frames', 'estimate_translation', 'estimate_translation_np', 'evaluate_mesh', 'flip_thetas', 'flip_thetas_batch', 'quat2mat', 'quaternion_to_angle_axis', 'rectify_pose', 'rigid_align', 'rigid_transform_3D', 'rot6d_to_rotmat', 'rot6d_to_rotmat_spin', 'rotation_matrix_to_angle_axis', 'rotation_matrix_to_quaternion']
group_utils_tools = ['construct_include', 'ensure_dir', 'get_config', 'read_pkl', 'Loader', 'TextLogger']
group_utils_vismo = ['bounding_box', 'get_img_from_fig', 'hex2rgb', 'joints2image', 'motion2video', 'motion2video_3d', 'motion2video_mesh', 'pixel2world_vis', 'pixel2world_vis_motion', 'render_and_save', 'rgb2rgba', 'save_image', 'vis_data_batch']
group_utils_args = ['check_args', 'get_opt_args_from_model_name', 'get_opts_args', 'list_of_strings', 'parse_args']
group_utils_learning = ['accuracy', 'load_backbone', 'load_pretrained_weights', 'partial_train_layers', 'AverageMeter']
group_model_drop = ['drop_path', 'DropPath']
group_model_load_model = ['load_model']
group_model_evaluation = ['batch_inference_eval', 'calculate_eval_metric', 'calculate_eval_metric_canonicalization', 'evaluate', 'evaluate_onevec', 'inference_eval', 'preprocess_eval']
group_model_model_action = ['ActionHeadClassification', 'ActionHeadEmbed', 'ActionNet']
group_model_CanonDSTformer = ['_no_grad_trunc_normal_', 'trunc_normal_', 'Attention', 'Block', 'CanonDSTformer1', 'CanonDSTformer2', 'MLP']
group_model_load_dataset = ['load_dataset']
group_model_loss_supcon = ['SupConLoss']
group_model_model_mesh = ['MeshRegressor', 'SMPLRegressor']
group_model_DHDSTformer = ['DHDSTformer_limb', 'DHDSTformer_limb2', 'DHDSTformer_limb3', 'DHDSTformer_limb4', 'DHDSTformer_limb5', 'DHDSTformer_limb_all_in_one', 'DHDSTformer_onevec', 'DHDSTformer_right_arm', 'DHDSTformer_right_arm2', 'DHDSTformer_right_arm3', 'DHDSTformer_torso', 'DHDSTformer_torso2', 'DHDSTformer_torso_limb', 'DHDSTformer_total', 'DHDSTformer_total2', 'DHDSTformer_total3', 'DHDSTformer_total4', 'DHDSTformer_total5', 'DHDSTformer_total6', 'DHDSTformer_total7', 'DHDSTformer_total8', 'linear_head']
group_model_DSTformer = ['_no_grad_trunc_normal_', 'trunc_normal_', 'Attention', 'Block', 'DSTformer', 'MLP']
group_model_loss = ['get_angles', 'get_limb_lens', 'loss_2d_weighted', 'loss_angle', 'loss_angle_velocity', 'loss_joint', 'loss_limb_gt', 'loss_limb_var', 'loss_mpjpe', 'loss_symmetry', 'loss_velocity', 'mpjpe', 'mpjpe_for_each_joint', 'n_mpjpe', 'p_mpjpe', 'p_mpjpe_for_each_joint', 'weighted_bonelen_loss', 'weighted_boneratio_loss', 'weighted_mpjpe']
group_model_loss_mesh = ['MeshLoss']
group_model_training = ['generate_loss_dict', 'inference_train', 'preprocess_train', 'save_checkpoint', 'train', 'train_epoch', 'update_train_writer']
group_model_DHformer = ['_no_grad_trunc_normal_', 'trunc_normal_', 'Attention', 'Block', 'DHformer', 'MLP']

__all__ = group_data_datareader_aihub + group_data_datareader_fit3d + group_data_datareader_h36m + group_data_datareader_poseaug_3dhp + group_data_datareader_mesh + group_data_datareader_3dhp + group_data_dataset_motion_2d + group_data_datareader_total + group_data_dataset_motion_3d + group_data_datareader_random_limb + group_data_augmentation + group_data_dataset_mesh + group_data_datareader_kookmin + group_data_dataset_action + group_data_dataset_wild + group_utils_utils_data + group_utils_utils_smpl + group_utils_utils_mesh + group_utils_tools + group_utils_vismo + group_utils_args + group_utils_learning + group_model_drop + group_model_load_model + group_model_evaluation + group_model_model_action + group_model_CanonDSTformer + group_model_load_dataset + group_model_loss_supcon + group_model_model_mesh + group_model_DHDSTformer + group_model_DSTformer + group_model_loss + group_model_loss_mesh + group_model_training + group_model_DHformer