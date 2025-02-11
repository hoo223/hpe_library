# https://passwd.tistory.com/entry/Python-initpy
# hpe_library/my_utils library
from hpe_library.my_utils.inference import args_dict_to_namespace, construct_torso_from_output, denormalize_input, denormalize_motionbert_result, get_dataset_info_from_segment_folder, get_inference_from_DHDSTformer_limb, get_inference_from_dhdst, get_inference_from_dhdst_torso, get_inference_from_motionbert, get_output_type, get_result, infer_one_segment, load_best_model_for_inference, normalize_input, test_model_by_segment_file
from hpe_library.my_utils.model import init_weights, split_array_by_idxs, BaselineModel, Linear, TorsoModel
from hpe_library.my_utils.dh import DH_matrix, azim_elev_to_vec, batch_azim_elev_to_vec, batch_inverse_tf, batch_projection, batch_rot_x_matrix, batch_rot_y_matrix, batch_rot_z_matrix, build_dh_frame, calculate_azimuth_elevation, calculate_batch_azimuth_elevation, calculate_batch_azimuth_elevation2, calculate_rotation_quaternion, dist_between_points, distance_between_azim_elev, draw_arm, draw_subline, draw_torso_direction, frame_vec_to_matrix, generate_batch_tf_from_batch_origin_R, generate_camera_frame, generate_dh_frame, generate_tf_from_origin_R, generate_two_link, generate_vis_frame, generate_world_frame, get_batch_frame_vec_from_keypoints, get_batch_lower_torso_frame_from_keypoints, get_batch_lower_torso_frame_from_pose, get_batch_reference_frame, get_batch_upper_torso_frame_from_keypoints, get_batch_upper_torso_frame_from_pose, get_frame_from_keypoints, get_lower_torso_frame_from_pose, get_optimal_azimuth_elevation, get_reference_frame, get_torso_direction, get_torso_rotation_matrix, get_torso_shape, get_upper_torso_frame_from_pose, inverse_tf, normalize_vector, project_batch_tensor, projection, rotate_pose_by_R_for_batch, rotate_torso_by_R, rotate_torso_by_R_for_batch_tensor, rotation_distance, rotation_matrix_from_angle, rotation_matrix_to_vector_align, rotation_matrix_torso2torso, Appendage, BatchAppendage, BatchDHModel, DHModel
from hpe_library.my_utils.camera import BatchCamera, Camera
from hpe_library.my_utils.pw3d import find_closest_frame_from_original_3dpw, find_closest_frame_from_poseaug_3dpw, get_3dpw_cam_params, get_3dpw_img_paths, get_3dpw_seq_list, get_3dpw_smpl_cam_3d_hat, get_3dpw_smpl_regressed_joint, get_3dpw_source_list, load_pkl_3dpw, verify_3dpw_seq_datatype
from hpe_library.my_utils.kookmin_dataset import check_available_frame, check_continuity, draw_base_marker_3d, generate_kookmin_pkl_for_each_video, get_cam_param_kookmin, get_lbot, get_video_frame_kookmin, get_video_num_frame_kookmin, load_csv_kookmin, load_pose3d_kookmin
from hpe_library.my_utils.logger import get_logger, log_configs
from hpe_library.my_utils.config import arg_as_list, get_configs
from hpe_library.my_utils.visualization import axes_2d, axes_3d, axes_to_compare_pred_gt, clear_axes, draw_2d_pose, draw_3d_pose, draw_bbox, draw_multiple_2d_pose, draw_multiple_3d_pose, draw_one_segment, draw_rotation_matrix, draw_segment, draw_segments, draw_trajectory, general_plot_func, generate_axes, generate_plot_video, generate_pose_video, get_2d_pose_image, legend_without_duplicate_labels, plot_to_compare_pred_gt, save_h36m_pose_video, show2Dtrajectory, show3Dtrajectory, show_2d_3d, show_whole_segment_trajectories
from hpe_library.my_utils.mpi_inf_3dhp import convert_intrinsic_from_mm_to_pixel, get_3dhp_cam_info, get_img_frame_3dhp, load_3dhp_original, test_3dhp_data_generator, test_3dhp_data_generator_new
from hpe_library.my_utils.data_aug import data_augmentation
from hpe_library.my_utils.notion import create_notion_page, get_all_database_pages, get_notion_dicts, update_mpjpe, update_notion_page
from hpe_library.my_utils.train import check_duplicate_training, get_input_output_candidate, get_num_trial, load_args, load_best_model, load_dataset, load_model, prepare_training, run, run_epoch, save_args, split_array_by_idxs
from hpe_library.my_utils.result import compare_two_checkpoints, load_excel_and_print_pt, load_excels_merge_print_pt, print_and_save_result, seed_summary
from hpe_library.my_utils.canonical import batch_inverse_rotation_matrices, batch_rotation_matrix_from_vectors, batch_rotation_matrix_from_vectors_torch, batch_virtualCameraRotationFromBatchInput, batch_virtualCameraRotationFromPosition, canonicalization_cam_3d, genertate_pcl_img_2d, get_batch_R_orig2virt_from_2d, get_batch_R_orig2virt_from_3d
from hpe_library.my_utils.test_utils import C_to_T, Camera2ImageCoordinate, MPJPE, MPJPE_for_multiple_pose, MPJPE_for_single_pose, T_to_C, World2CameraCoordinate, World2ImageCoordinate, _sqrt_positive_part, _weak_project, aihub2h36m, array2dict, avgErrorForOneAction, avgErrorForOneActor, avgErrorForOneCamera, camera_to_image_frame, change_bbox_convention, check_max_min, coco2h36m, draw_skeleton, draw_skeleton_2d, draw_skeleton_both, euclidean_distance, fit3d2h36m, getAIHubCameraParameter, getNumFromImgName, get_batch_bbox_from_pose2d, get_batch_h36m_keypoints, get_bbox_area, get_bbox_area_from_pose2d, get_bbox_from_pose2d, get_canonical_3d, get_euclidean_norm_from_pose, get_h36m_joint_name, get_h36m_keypoint_index, get_h36m_keypoints, get_h36m_len_ids, get_h36m_limb_lens, get_length_from_pose3d, get_length_ratio_from_pose3d, get_parent_index, get_pose_height, get_root_relative_depth_from_pose, get_rootrel_pose, get_video_frame, get_video_info, get_xy_centered_pose, halpe2h36m, image_coordinates, infer_box, kookmin2h36m, kookmin2h36m_with_nose, loadAIHubCameraParameter, matrix_to_quaternion, mpi_inf_3dhp2h36m, mpi_inf_3dhp2h36m_from_original, normalize, normalize_array, normalize_screen_coordinates, optimize_scaling_factor, plot_cv2_image, procrustes_align, readJSON, readpkl, remove_nose_from_h36m, rotation_matrix_from_vectors, savepkl, skew_symmetric_matrix, skew_symmetric_matrix_tensor, smpl2h36m, standardize_quaternion, undistort_pose2d, update_result_dict
from hpe_library.my_utils.dataset import generate_img_3d, generate_random_segment, generate_random_trajectory, gernerate_dataset_yaml, get_aligned_init_torso, get_ap_pose_2d, get_backbone_line_from_torso, get_bounded_segments_idx, get_cam_param, get_h36m_camera_info, get_input, get_input_gt_for_onevec, get_label, get_limb_angle, get_model_input, get_output, get_pairs, get_part_traj, get_pose_seq_and_cam_param, get_save_paths, get_two_point_parts, h36m_data_processing, load_cam_3d, load_cam_3d_canonical, load_cam_params, load_data, load_data_dict, load_fit3d_one_video, load_h36m, load_image_frame, load_img25d, load_img_2d, load_img_2d_canonical, load_img_3d, load_img_3d_norm, load_img_3d_norm_canonical, load_plot_configs, load_scale_factor, load_scale_factor_norm, load_scale_factor_norm_canonical, load_segment_file_from_parameters, load_source_list, load_total_data, load_world_3d, make_input, make_one_dimension_list, make_output, parse_args_by_model_name, select_dataset_from_checkpoint, select_testset_from_subset, split_continuous_indices, split_source_name, MyCustomDataset

group_inference = ['args_dict_to_namespace', 'construct_torso_from_output', 'denormalize_input', 'denormalize_motionbert_result', 'get_dataset_info_from_segment_folder', 'get_inference_from_DHDSTformer_limb', 'get_inference_from_dhdst', 'get_inference_from_dhdst_torso', 'get_inference_from_motionbert', 'get_output_type', 'get_result', 'infer_one_segment', 'load_best_model_for_inference', 'normalize_input', 'test_model_by_segment_file']
group_model = ['init_weights', 'split_array_by_idxs', 'BaselineModel', 'Linear', 'TorsoModel']
group_dh = ['DH_matrix', 'azim_elev_to_vec', 'batch_azim_elev_to_vec', 'batch_inverse_tf', 'batch_projection', 'batch_rot_x_matrix', 'batch_rot_y_matrix', 'batch_rot_z_matrix', 'build_dh_frame', 'calculate_azimuth_elevation', 'calculate_batch_azimuth_elevation', 'calculate_batch_azimuth_elevation2', 'calculate_rotation_quaternion', 'dist_between_points', 'distance_between_azim_elev', 'draw_arm', 'draw_subline', 'draw_torso_direction', 'frame_vec_to_matrix', 'generate_batch_tf_from_batch_origin_R', 'generate_camera_frame', 'generate_dh_frame', 'generate_tf_from_origin_R', 'generate_two_link', 'generate_vis_frame', 'generate_world_frame', 'get_batch_frame_vec_from_keypoints', 'get_batch_lower_torso_frame_from_keypoints', 'get_batch_lower_torso_frame_from_pose', 'get_batch_reference_frame', 'get_batch_upper_torso_frame_from_keypoints', 'get_batch_upper_torso_frame_from_pose', 'get_frame_from_keypoints', 'get_lower_torso_frame_from_pose', 'get_optimal_azimuth_elevation', 'get_reference_frame', 'get_torso_direction', 'get_torso_rotation_matrix', 'get_torso_shape', 'get_upper_torso_frame_from_pose', 'inverse_tf', 'normalize_vector', 'project_batch_tensor', 'projection', 'rotate_pose_by_R_for_batch', 'rotate_torso_by_R', 'rotate_torso_by_R_for_batch_tensor', 'rotation_distance', 'rotation_matrix_from_angle', 'rotation_matrix_to_vector_align', 'rotation_matrix_torso2torso', 'Appendage', 'BatchAppendage', 'BatchDHModel', 'DHModel']
group_camera = ['BatchCamera', 'Camera']
group_pw3d = ['find_closest_frame_from_original_3dpw', 'find_closest_frame_from_poseaug_3dpw', 'get_3dpw_cam_params', 'get_3dpw_img_paths', 'get_3dpw_seq_list', 'get_3dpw_smpl_cam_3d_hat', 'get_3dpw_smpl_regressed_joint', 'get_3dpw_source_list', 'load_pkl_3dpw', 'verify_3dpw_seq_datatype']
group_kookmin_dataset = ['check_available_frame', 'check_continuity', 'draw_base_marker_3d', 'generate_kookmin_pkl_for_each_video', 'get_cam_param_kookmin', 'get_lbot', 'get_video_frame_kookmin', 'get_video_num_frame_kookmin', 'load_csv_kookmin', 'load_pose3d_kookmin']
group_logger = ['get_logger', 'log_configs']
group_config = ['arg_as_list', 'get_configs']
group_visualization = ['axes_2d', 'axes_3d', 'axes_to_compare_pred_gt', 'clear_axes', 'draw_2d_pose', 'draw_3d_pose', 'draw_bbox', 'draw_multiple_2d_pose', 'draw_multiple_3d_pose', 'draw_one_segment', 'draw_rotation_matrix', 'draw_segment', 'draw_segments', 'draw_trajectory', 'general_plot_func', 'generate_axes', 'generate_plot_video', 'generate_pose_video', 'get_2d_pose_image', 'legend_without_duplicate_labels', 'plot_to_compare_pred_gt', 'save_h36m_pose_video', 'show2Dtrajectory', 'show3Dtrajectory', 'show_2d_3d', 'show_whole_segment_trajectories']
group_mpi_inf_3dhp = ['convert_intrinsic_from_mm_to_pixel', 'get_3dhp_cam_info', 'get_img_frame_3dhp', 'load_3dhp_original', 'test_3dhp_data_generator', 'test_3dhp_data_generator_new']
group_data_aug = ['data_augmentation']
group_notion = ['create_notion_page', 'get_all_database_pages', 'get_notion_dicts', 'update_mpjpe', 'update_notion_page']
group_train = ['check_duplicate_training', 'get_input_output_candidate', 'get_num_trial', 'load_args', 'load_best_model', 'load_dataset', 'load_model', 'prepare_training', 'run', 'run_epoch', 'save_args', 'split_array_by_idxs']
group_result = ['compare_two_checkpoints', 'load_excel_and_print_pt', 'load_excels_merge_print_pt', 'print_and_save_result', 'seed_summary']
group_canonical = ['batch_inverse_rotation_matrices', 'batch_rotation_matrix_from_vectors', 'batch_rotation_matrix_from_vectors_torch', 'batch_virtualCameraRotationFromBatchInput', 'batch_virtualCameraRotationFromPosition', 'canonicalization_cam_3d', 'genertate_pcl_img_2d', 'get_batch_R_orig2virt_from_2d', 'get_batch_R_orig2virt_from_3d']
group_test_utils = ['C_to_T', 'Camera2ImageCoordinate', 'MPJPE', 'MPJPE_for_multiple_pose', 'MPJPE_for_single_pose', 'T_to_C', 'World2CameraCoordinate', 'World2ImageCoordinate', '_sqrt_positive_part', '_weak_project', 'aihub2h36m', 'array2dict', 'avgErrorForOneAction', 'avgErrorForOneActor', 'avgErrorForOneCamera', 'camera_to_image_frame', 'change_bbox_convention', 'check_max_min', 'coco2h36m', 'draw_skeleton', 'draw_skeleton_2d', 'draw_skeleton_both', 'euclidean_distance', 'fit3d2h36m', 'getAIHubCameraParameter', 'getNumFromImgName', 'get_batch_bbox_from_pose2d', 'get_batch_h36m_keypoints', 'get_bbox_area', 'get_bbox_area_from_pose2d', 'get_bbox_from_pose2d', 'get_canonical_3d', 'get_euclidean_norm_from_pose', 'get_h36m_joint_name', 'get_h36m_keypoint_index', 'get_h36m_keypoints', 'get_h36m_len_ids', 'get_h36m_limb_lens', 'get_length_from_pose3d', 'get_length_ratio_from_pose3d', 'get_parent_index', 'get_pose_height', 'get_root_relative_depth_from_pose', 'get_rootrel_pose', 'get_video_frame', 'get_video_info', 'get_xy_centered_pose', 'halpe2h36m', 'image_coordinates', 'infer_box', 'kookmin2h36m', 'kookmin2h36m_with_nose', 'loadAIHubCameraParameter', 'matrix_to_quaternion', 'mpi_inf_3dhp2h36m', 'mpi_inf_3dhp2h36m_from_original', 'normalize', 'normalize_array', 'normalize_screen_coordinates', 'optimize_scaling_factor', 'plot_cv2_image', 'procrustes_align', 'readJSON', 'readpkl', 'remove_nose_from_h36m', 'rotation_matrix_from_vectors', 'savepkl', 'skew_symmetric_matrix', 'skew_symmetric_matrix_tensor', 'smpl2h36m', 'standardize_quaternion', 'undistort_pose2d', 'update_result_dict']
group_dataset = ['generate_img_3d', 'generate_random_segment', 'generate_random_trajectory', 'gernerate_dataset_yaml', 'get_aligned_init_torso', 'get_ap_pose_2d', 'get_backbone_line_from_torso', 'get_bounded_segments_idx', 'get_cam_param', 'get_h36m_camera_info', 'get_input', 'get_input_gt_for_onevec', 'get_label', 'get_limb_angle', 'get_model_input', 'get_output', 'get_pairs', 'get_part_traj', 'get_pose_seq_and_cam_param', 'get_save_paths', 'get_two_point_parts', 'h36m_data_processing', 'load_cam_3d', 'load_cam_3d_canonical', 'load_cam_params', 'load_data', 'load_data_dict', 'load_fit3d_one_video', 'load_h36m', 'load_image_frame', 'load_img25d', 'load_img_2d', 'load_img_2d_canonical', 'load_img_3d', 'load_img_3d_norm', 'load_img_3d_norm_canonical', 'load_plot_configs', 'load_scale_factor', 'load_scale_factor_norm', 'load_scale_factor_norm_canonical', 'load_segment_file_from_parameters', 'load_source_list', 'load_total_data', 'load_world_3d', 'make_input', 'make_one_dimension_list', 'make_output', 'parse_args_by_model_name', 'select_dataset_from_checkpoint', 'select_testset_from_subset', 'split_continuous_indices', 'split_source_name', 'MyCustomDataset']

__all__ = group_inference + group_model + group_dh + group_camera + group_pw3d + group_kookmin_dataset + group_logger + group_config + group_visualization + group_mpi_inf_3dhp + group_data_aug + group_notion + group_train + group_result + group_canonical + group_test_utils + group_dataset