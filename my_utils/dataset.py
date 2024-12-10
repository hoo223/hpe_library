from traitlets import Instance
from hpe_library.lib_import import *
# from .dh import rotate_torso_by_R, get_torso_direction, rotation_matrix_to_vector_align, projection, get_torso_rotation_matrix, calculate_batch_azimuth_elevation
# from .test_utils import readJSON, readpkl, savepkl, halpe2h36m, get_video_info, get_video_frame, get_bbox_area_from_pose2d, get_bbox_from_pose2d, change_bbox_convention, get_bbox_area
# from .test_utils import get_h36m_keypoint_index
# from .test_utils import get_rootrel_pose, optimize_scaling_factor, infer_box, camera_to_image_frame

# from .data_aug import data_augmentation
# from .canonical import canonicalization_cam_3d
# from posynda_utils import Human36mDataset

## for general

def split_source_name(source:str, dataset_name:str) -> Tuple[str, Optional[str], Optional[str]]:
    if dataset_name == 'h36m' or dataset_name == 'kookmin':
        subject, cam_id, action = source.split('_')
        return subject, cam_id, action
    elif dataset_name == 'fit3d':
        splited = source.split('_')
        subject = splited[0]
        cam_id = splited[1]
        action = '_'.join(splited[2:])
        return subject, cam_id, action
    elif dataset_name == '3dhp':
        try:
            subject, cam_id, seq = source.split('_')
            if cam_id == 'None': cam_id = None # only for test data
            if seq == 'None': seq = None # only for test data
            return subject, cam_id, seq
        except: # for test data
            subject = source
            return subject, None, None
    else:
        raise ValueError(f'{dataset_name} not found')

def load_plot_configs(dataset_name:str):
    user = getpass.getuser()
    save_root = f'/home/{user}/codes/MotionBERT/custom_codes/total_process/plot_configs'
    save_path_plot_configs = os.path.join(save_root, f'{dataset_name}_plot_configs.yaml')
    with open(save_path_plot_configs, 'r') as file:
        plot_configs = yaml.safe_load(file)
    return plot_configs

def get_save_paths(save_root:str, dataset_name:str, canonical_type:str, univ:bool,
                   data_aug:dict[str, int]={'step_rot': 0,
                        'sinu_yaw_mag': 0, 'sinu_yaw_period': 273, 'sinu_pitch_mag': 0, 'sinu_pitch_period': 273,
                        'sinu_roll_mag': 0, 'sinu_roll_period': 273,'rand_yaw_mag': 0, 'rand_yaw_period': 0,
                        'rand_pitch_mag': 0, 'rand_pitch_period': 0,'rand_roll_mag': 0, 'rand_roll_period': 0
                    }) -> dict[str, str]:

    def add_data_aug_info(path, data_aug):
        step_rot = data_aug['step_rot']
        sinu_yaw_mag = data_aug['sinu_yaw_mag']
        sinu_yaw_period = data_aug['sinu_yaw_period']
        sinu_pitch_mag = data_aug['sinu_pitch_mag']
        sinu_pitch_period = data_aug['sinu_pitch_period']
        sinu_roll_mag = data_aug['sinu_roll_mag']
        sinu_roll_period = data_aug['sinu_roll_period']
        rand_yaw_mag = data_aug['rand_yaw_mag']
        rand_yaw_period = data_aug['rand_yaw_period']
        rand_pitch_mag = data_aug['rand_pitch_mag']
        rand_pitch_period = data_aug['rand_pitch_period']
        rand_roll_mag = data_aug['rand_roll_mag']
        rand_roll_period = data_aug['rand_roll_period']

        if step_rot != 0: path += f'-steprot_{step_rot}'
        elif sinu_yaw_mag != 0: path += f'-sinu_yaw_m{int(sinu_yaw_mag)}_p{int(sinu_yaw_period)}'
        elif rand_yaw_mag != 0: path += f'-rand_yaw_m{int(rand_yaw_mag)}_p{int(rand_yaw_period)}'
        if sinu_pitch_mag != 0: path += f'-sinu_pitch_m{int(sinu_pitch_mag)}_p{int(sinu_pitch_period)}'
        elif rand_pitch_mag != 0: path += f'-rand_pitch_m{int(rand_pitch_mag)}_p{int(rand_pitch_period)}'
        if sinu_roll_mag != 0:  path += f'-sinu_roll_m{int(sinu_roll_mag)}_p{int(sinu_roll_period)}'
        elif rand_roll_mag != 0: path += f'-rand_roll_m{int(rand_roll_mag)}_p{int(rand_roll_period)}'
        path += '.pkl'
        return path

    # source_list
    save_path_source_list = os.path.join(save_root, f'{dataset_name}-source_list.pkl')
    # cam_params
    save_path_cam_params = os.path.join(save_root, f'{dataset_name}-cam_param.pkl')
    save_path_cam_params_adaptive_focal = os.path.join(save_root, f'{dataset_name}-cam_param-adaptive_focal.pkl')
    # cam_3d
    save_path_cam_3d = os.path.join(save_root, f'{dataset_name}-cam_3d')
    if dataset_name == '3dhp' and univ: save_path_cam_3d += '_univ'
    save_path_cam_3d = add_data_aug_info(save_path_cam_3d, data_aug)
    # if step_rot != 0: save_path_cam_3d += f'-steprot_{step_rot}'
    # elif sinu_yaw_mag != 0: save_path_cam_3d += f'-sinu_yaw_m{int(sinu_yaw_mag)}_p{int(sinu_yaw_period)}'
    # elif rand_yaw_mag != 0: save_path_cam_3d += f'-rand_yaw_m{int(rand_yaw_mag)}_p{int(rand_yaw_period)}'
    # if sinu_pitch_mag != 0: save_path_cam_3d += f'-sinu_pitch_m{int(sinu_pitch_mag)}_p{int(sinu_pitch_period)}'
    # elif rand_pitch_mag != 0: save_path_cam_3d += f'-rand_pitch_m{int(rand_pitch_mag)}_p{int(rand_pitch_period)}'
    # if sinu_roll_mag != 0:  save_path_cam_3d += f'-sinu_roll_m{int(sinu_roll_mag)}_p{int(sinu_roll_period)}'
    # elif rand_roll_mag != 0: save_path_cam_3d += f'-rand_roll_m{int(rand_roll_mag)}_p{int(rand_roll_period)}'
    # save_path_cam_3d += '.pkl'

    # cam_3d_canonical
    save_path_cam_3d_canonical = os.path.join(save_root, f'{dataset_name}-cam_3d')
    if dataset_name == '3dhp' and univ: save_path_cam_3d_canonical += '_univ'
    save_path_cam_3d_canonical += f"-canonical_{canonical_type}"
    save_path_cam_3d_canonical = add_data_aug_info(save_path_cam_3d_canonical, data_aug)
    # if step_rot != 0: save_path_cam_3d_canonical += f'-steprot_{step_rot}'
    # elif sinu_yaw_mag != 0: save_path_cam_3d_canonical += f'-sinu_yaw_m{int(sinu_yaw_mag)}_p{int(sinu_yaw_period)}'
    # elif rand_yaw_mag != 0: save_path_cam_3d_canonical += f'-rand_yaw_m{int(rand_yaw_mag)}_p{int(rand_yaw_period)}'
    # if sinu_pitch_mag != 0: save_path_cam_3d_canonical += f'-sinu_pitch_m{int(sinu_pitch_mag)}_p{int(sinu_pitch_period)}'
    # elif rand_pitch_mag != 0: save_path_cam_3d_canonical += f'-rand_pitch_m{int(rand_pitch_mag)}_p{int(rand_pitch_period)}'
    # if sinu_roll_mag != 0:  save_path_cam_3d_canonical += f'-sinu_roll_m{int(sinu_roll_mag)}_p{int(sinu_roll_period)}'
    # elif rand_roll_mag != 0: save_path_cam_3d_canonical += f'-rand_roll_m{int(rand_roll_mag)}_p{int(rand_roll_period)}'
    # save_path_cam_3d_canonical += '.pkl'

    # img_2d
    save_path_img_2d = os.path.join(save_root, f'{dataset_name}-img_2d')
    save_path_img_2d = add_data_aug_info(save_path_img_2d, data_aug)
    # if step_rot != 0: save_path_img_2d += f'-steprot_{step_rot}'
    # elif sinu_yaw_mag != 0: save_path_img_2d += f'-sinu_yaw_m{int(sinu_yaw_mag)}_p{int(sinu_yaw_period)}'
    # elif rand_yaw_mag != 0: save_path_img_2d += f'-rand_yaw_m{int(rand_yaw_mag)}_p{int(rand_yaw_period)}'
    # if sinu_pitch_mag != 0: save_path_img_2d += f'-sinu_pitch_m{int(sinu_pitch_mag)}_p{int(sinu_pitch_period)}'
    # elif rand_pitch_mag != 0: save_path_img_2d += f'-rand_pitch_m{int(rand_pitch_mag)}_p{int(rand_pitch_period)}'
    # if sinu_roll_mag != 0:  save_path_img_2d += f'-sinu_roll_m{int(sinu_roll_mag)}_p{int(sinu_roll_period)}'
    # elif rand_roll_mag != 0: save_path_img_2d += f'-rand_roll_m{int(rand_roll_mag)}_p{int(rand_roll_period)}'
    # save_path_img_2d += '.pkl'

    # img_2d_canonical
    save_path_img_2d_canonical = os.path.join(save_root, f'{dataset_name}-img_2d')
    save_path_img_2d_canonical += f"-canonical_{canonical_type}"
    save_path_img_2d_canonical = add_data_aug_info(save_path_img_2d_canonical, data_aug)
    # if step_rot != 0: save_path_img_2d_canonical += f'-steprot_{step_rot}'
    # elif sinu_yaw_mag != 0: save_path_img_2d_canonical += f'-sinu_yaw_m{int(sinu_yaw_mag)}_p{int(sinu_yaw_period)}'
    # elif rand_yaw_mag != 0: save_path_img_2d_canonical += f'-rand_yaw_m{int(rand_yaw_mag)}_p{int(rand_yaw_period)}'
    # if sinu_pitch_mag != 0: save_path_img_2d_canonical += f'-sinu_pitch_m{int(sinu_pitch_mag)}_p{int(sinu_pitch_period)}'
    # elif rand_pitch_mag != 0: save_path_img_2d_canonical += f'-rand_pitch_m{int(rand_pitch_mag)}_p{int(rand_pitch_period)}'
    # if sinu_roll_mag != 0:  save_path_img_2d_canonical += f'-sinu_roll_m{int(sinu_roll_mag)}_p{int(sinu_roll_period)}'
    # elif rand_roll_mag != 0: save_path_img_2d_canonical += f'-rand_roll_m{int(rand_roll_mag)}_p{int(rand_roll_period)}'
    # save_path_img_2d_canonical += '.pkl'

    # img_2d_canonical_adaptive_focal
    save_path_img_2d_canonical_adaptive_focal = os.path.join(save_root, f'{dataset_name}-img_2d-canonical_adaptive_focal.pkl')
    # world_3d
    save_path_world_3d = os.path.join(save_root, f'{dataset_name}-world_3d')
    if dataset_name == '3dhp' and univ: save_path_world_3d += '_univ'
    save_path_world_3d += '.pkl'
    # img_3d
    save_path_img_3d = os.path.join(save_root, f'{dataset_name}-img_3d')
    if dataset_name == '3dhp' and univ: save_path_img_3d += '_univ'
    save_path_img_3d += '.pkl'
    # scale_factor
    save_path_scale_factor = os.path.join(save_root, f'{dataset_name}-scale_factor')
    if dataset_name == '3dhp' and univ: save_path_scale_factor += '_univ'
    save_path_scale_factor += '.pkl'
    # img_25d
    save_path_img_25d = os.path.join(save_root, f'{dataset_name}-img_25d')
    if dataset_name == '3dhp' and univ: save_path_img_25d += '_univ'
    save_path_img_25d += '.pkl'

    save_paths = {
        'source_list': save_path_source_list,
        'cam_param': save_path_cam_params,
        'cam_param_adaptive_focal': save_path_cam_params_adaptive_focal,
        'world_3d': save_path_world_3d,
        'cam_3d': save_path_cam_3d,
        'img_2d': save_path_img_2d,
        'cam_3d_canonical': save_path_cam_3d_canonical,
        'img_2d_canonical': save_path_img_2d_canonical,
        'img_3d': save_path_img_3d,
        'img_25d': save_path_img_25d,
        'scale_factor': save_path_scale_factor,
        'img_2d_canonical_adaptive_focal': save_path_img_2d_canonical_adaptive_focal,
    }
    return save_paths

def load_data(dataset_name, data_type, save_folder='data/motion3d', overwrite_list=[], canonical_type=None, only_visible_frame=True, univ=False, no_save=False, adaptive_focal=False, verbose=True,
              data_aug={'step_rot': 0,
                        'sinu_yaw_mag': 0, 'sinu_yaw_period': 273, 'sinu_pitch_mag': 0, 'sinu_pitch_period': 273,
                        'sinu_roll_mag': 0, 'sinu_roll_period': 273,'rand_yaw_mag': 0, 'rand_yaw_period': 0,
                        'rand_pitch_mag': 0, 'rand_pitch_period': 0,'rand_roll_mag': 0, 'rand_roll_period': 0
                        }):
    user = getpass.getuser()
    motionbert_root = f'/home/{user}/codes/MotionBERT/'
    save_root = os.path.join(motionbert_root, save_folder, dataset_name)

    step_rot = data_aug['step_rot']
    sinu_yaw_mag = data_aug['sinu_yaw_mag']
    sinu_yaw_period = data_aug['sinu_yaw_period']
    sinu_pitch_mag = data_aug['sinu_pitch_mag']
    sinu_pitch_period = data_aug['sinu_pitch_period']
    sinu_roll_mag = data_aug['sinu_roll_mag']
    sinu_roll_period = data_aug['sinu_roll_period']
    rand_yaw_mag = data_aug['rand_yaw_mag']
    rand_yaw_period = data_aug['rand_yaw_period']
    rand_pitch_mag = data_aug['rand_pitch_mag']
    rand_pitch_period = data_aug['rand_pitch_period']
    rand_roll_mag = data_aug['rand_roll_mag']
    rand_roll_period = data_aug['rand_roll_period']

    # only_visible_frame -> for 3dhp
    if data_type in overwrite_list:
        overwrite = True
    else:
        if canonical_type != None:
            if '_'.join([data_type, canonical_type]) in overwrite_list: overwrite = True
            else: overwrite = False
        else:
            overwrite = False

    if verbose:
        final_data_type = f'{data_type}'
        if 'canonical' in data_type and canonical_type is not None: final_data_type += f'_{canonical_type}'
        if adaptive_focal:
            if data_type in ['cam_param', 'img_2d_canonical']:
                final_data_type += '-adaptive_focal'
        if data_type in ['cam_3d', 'img_2d', 'cam_3d_canonical', 'img_2d_canonical']:
            if data_aug['step_rot'] != 0: final_data_type += f'-steprot_{step_rot}'
            elif sinu_yaw_mag != 0: final_data_type += f"-sinu_yaw_m{int(sinu_yaw_mag)}_p{int(sinu_yaw_period)}"
            elif rand_yaw_mag != 0: final_data_type += f"-rand_yaw_m{int(rand_yaw_mag)}_p{int(rand_yaw_period)}"
            if sinu_pitch_mag != 0: final_data_type += f"-sinu_pitch_m{int(sinu_pitch_mag)}_p{int(sinu_pitch_period)}"
            elif rand_pitch_mag != 0: final_data_type += f"-rand_pitch_m{int(rand_pitch_mag)}_p{int(rand_pitch_period)}"
            if sinu_roll_mag != 0: final_data_type += f"-sinu_roll_m{int(sinu_roll_mag)}_p{int(sinu_roll_period)}"
            elif rand_roll_mag != 0: final_data_type += f"-rand_roll_m{int(rand_roll_mag)}_p{int(rand_roll_period)}"
        print(f"[overwrite: {overwrite}] ==> Loading {dataset_name.upper()} {final_data_type}...")

    # save path
    save_paths = get_save_paths(save_root, dataset_name, canonical_type, univ, data_aug)

    if data_type   == 'source_list':      return load_source_list(dataset_name, save_paths, overwrite, no_save)
    elif data_type == 'cam_param':        return load_cam_params(dataset_name, save_paths, overwrite, no_save, only_visible_frame, adaptive_focal)
    elif data_type == 'world_3d':         return load_world_3d(dataset_name, save_paths, overwrite, no_save)
    elif data_type == 'cam_3d':           return load_cam_3d(dataset_name, save_paths, overwrite, no_save, univ, only_visible_frame, data_aug)
    elif data_type == 'img_2d':           return load_img_2d(dataset_name, save_paths, overwrite, no_save, only_visible_frame, data_aug)
    elif data_type == 'img_3d':           return load_img_3d(dataset_name, save_paths, overwrite, no_save)
    elif data_type == 'img_25d':          return load_img25d(dataset_name, save_paths, overwrite, no_save)
    elif data_type == 'scale_factor':     return load_scale_factor(dataset_name, save_paths, overwrite, no_save)
    elif data_type == 'cam_3d_canonical': return load_cam_3d_canonical(dataset_name, save_paths, canonical_type, overwrite, no_save, data_aug)
    elif data_type == 'img_2d_canonical': return load_img_2d_canonical(dataset_name, save_paths, canonical_type, overwrite, no_save, adaptive_focal, data_aug)
    else:                                 raise ValueError(f'{data_type} not found')

def load_data_dict(dataset_name, data_type_list=[], overwrite_list=[], verbose=True, univ=False,
                   data_aug={'step_rot': 0,
                        'sinu_yaw_mag': 0, 'sinu_yaw_period': 273, 'sinu_pitch_mag': 0, 'sinu_pitch_period': 273,
                        'sinu_roll_mag': 0, 'sinu_roll_period': 273,'rand_yaw_mag': 0, 'rand_yaw_period': 0,
                        'rand_pitch_mag': 0, 'rand_pitch_period': 0,'rand_roll_mag': 0, 'rand_roll_period': 0
                        }):
    step_rot = data_aug['step_rot']
    sinu_yaw_mag = data_aug['sinu_yaw_mag']
    sinu_yaw_period = data_aug['sinu_yaw_period']
    sinu_pitch_mag = data_aug['sinu_pitch_mag']
    sinu_pitch_period = data_aug['sinu_pitch_period']
    sinu_roll_mag = data_aug['sinu_roll_mag']
    sinu_roll_period = data_aug['sinu_roll_period']
    rand_yaw_mag = data_aug['rand_yaw_mag']
    rand_yaw_period = data_aug['rand_yaw_period']
    rand_pitch_mag = data_aug['rand_pitch_mag']
    rand_pitch_period = data_aug['rand_pitch_period']
    rand_roll_mag = data_aug['rand_roll_mag']
    rand_roll_period = data_aug['rand_roll_period']

    data_dict = {}
    for data_type in data_type_list:
        key = data_type

        if 'adaptive_focal' in data_type:
            data_type = data_type.split('_adaptive_focal')[0]
            adaptive_focal = True
        else: adaptive_focal = False

        if 'cam_3d_canonical' in data_type:
            canonical_type = data_type.split('canonical_')[-1]
            data_type = 'cam_3d_canonical'
        elif 'img_2d_canonical' in data_type:
            canonical_type = data_type.split('canonical_')[-1]
            data_type = 'img_2d_canonical'
        else:
            canonical_type = None

        # if data_type in ['cam_3d', 'img_2d', 'cam_3d_canonical', 'img_2d_canonical']:
        #     if step_rot != 0: key += f'-steprot_{step_rot}'
        #     elif sinu_yaw_mag != 0: key += f'-sinu_yaw_m{sinu_yaw_mag}_p{sinu_yaw_period}'
        #     elif rand_yaw_mag != 0: key += f'-rand_yaw_m{rand_yaw_mag}_p{rand_yaw_period}'
        #     if sinu_pitch_mag != 0: key += f'-sinu_pitch_m{sinu_pitch_mag}_p{sinu_pitch_period}'
        #     elif rand_pitch_mag != 0: key += f'rand_pitch_m{rand_pitch_mag}_p{rand_pitch_period}'
        #     if sinu_roll_mag != 0: key += f'-sinu_roll_m{sinu_roll_mag}_p{sinu_roll_period}'
        #     elif rand_roll_mag != 0: key += f'rand_roll_m{rand_roll_mag}_p{rand_roll_period}'

        data_dict[key] = load_data(dataset_name=dataset_name, data_type=data_type, canonical_type=canonical_type, overwrite_list=overwrite_list, adaptive_focal=adaptive_focal, verbose=verbose, univ=univ, data_aug=data_aug)
    return data_dict

def load_source_list(dataset_name, save_paths, overwrite=False, no_save=False):
    from my_utils import load_3dhp_original, readpkl, savepkl
    from posynda_utils import Human36mDataset
    user = getpass.getuser()
    save_path_source_list = save_paths['source_list']
    if os.path.exists(save_path_source_list) and not overwrite and not no_save:
        source_list = readpkl(save_path_source_list)
    else:
        source_list = []
        if dataset_name == 'h36m':
            if 'h36m_dataset' in globals(): del globals()['h36m_dataset']
            if 'h36m_dataset' in locals(): del locals()['h36m_dataset']
            h36m_dataset = Human36mDataset(f'/home/{user}/codes/hpe_library/data/data_3d_h36m.npz', remove_static_joints=True)._data
            subject_list = h36m_dataset.keys()
            cam_list = ['54138969', '60457274', '55011271', '58860488']
            for subject in subject_list:
                action_list = h36m_dataset[subject].keys()
                for action in action_list:
                    for cam_id in cam_list:
                        source_list.append(f'{subject}_{cam_id}_{action}')
        elif dataset_name == 'fit3d':
            fit3d_root = f'/home/{user}/Datasets/HAAI/Fit3D/train'
            subject_list = os.listdir(fit3d_root)
            subject_list.remove('counts')
            for subject in tqdm(subject_list):
                subject_root = os.path.join(fit3d_root, subject)
                gt_3d_root = os.path.join(subject_root, 'joints3d_25')
                cam_root = os.path.join(subject_root, 'camera_parameters')
                action_list = [action.split('.')[0] for action in os.listdir(gt_3d_root)]
                cam_list = [cam_id for cam_id in os.listdir(cam_root)]
                for action in action_list:
                    for cam_id in cam_list:
                        source_list.append(f'{subject}_{cam_id}_{action}')
        elif dataset_name == '3dhp':
            data_dict_3dhp_train, cam_param_3dhp_train = load_3dhp_original('train')
            data_dict_3dhp_test, cam_param_3dhp_test = load_3dhp_original('test')
            data_dict_3dhp = {**data_dict_3dhp_train, **data_dict_3dhp_test}
            for source in data_dict_3dhp.keys():
                subject, seq, cam_id = split_source_name(source, '3dhp')
                source_list.append(f'{subject}_{cam_id}_{seq}')
        else:
            raise ValueError(f'{dataset_name} not found')
        if not no_save: savepkl(source_list, save_path_source_list)
    return source_list

def load_cam_params(dataset_name, save_paths, overwrite=False, no_save=False, only_visible_frame=False, adaptive_focal=False):
    from my_utils import load_3dhp_original, readpkl, savepkl, readJSON, get_video_info
    from posynda_utils import Human36mDataset
    if adaptive_focal: save_path_cam_params = save_paths['cam_param_adaptive_focal']
    else: save_path_cam_params = save_paths['cam_param']
    if os.path.exists(save_path_cam_params) and not overwrite and not no_save:
        cam_params = readpkl(save_path_cam_params)
    else:
        save_path_source_list = save_paths['source_list']
        assert os.path.exists(save_path_source_list), f'No source_list found for {dataset_name}'
        source_list = readpkl(save_path_source_list)
        cam_params = {}
        if dataset_name == 'h36m':
            if 'h36m_dataset' in globals(): del globals()['h36m_dataset']
            if 'h36m_dataset' in locals(): del locals()['h36m_dataset']
            h36m_dataset = Human36mDataset('/home/hrai/codes/hpe_library/data/data_3d_h36m.npz', remove_static_joints=True)._data
            cam_params_h36m = readJSON('/home/hrai/codes/hpe_library/data/h36m_camera-parameters.json')
            for source in tqdm(source_list):
                subject, cam_id, action = split_source_name(source, dataset_name)
                if subject not in cam_params.keys():         cam_params[subject] = {}
                if action not in cam_params[subject].keys(): cam_params[subject][action] = {}
                intrinsic = np.array(cam_params_h36m['intrinsics'][cam_id]['calibration_matrix']) # (3, 3)
                R = np.array(cam_params_h36m['extrinsics'][subject][cam_id]['R']) # (3, 3)
                t = np.array(cam_params_h36m['extrinsics'][subject][cam_id]['t']).reshape(-1)/1000 # unit: m, (3,)
                extrinsic = np.hstack([R, t.reshape(-1, 1)]) # (3, 4)
                C = -R.T @ t # (3,)
                if cam_id in ['54138969', '60457274']:   W, H = 1000, 1002
                elif cam_id in ['55011271', '58860488']: W, H = 1000, 1000
                else: raise ValueError(f'cam_id {cam_id} not found')
                num_frames = h36m_dataset[subject][action]['positions'].shape[0]
                if adaptive_focal:
                    intrinsic[0][0] = W
                    intrinsic[1][1] = H
                cam_params[subject][action][cam_id] = {
                    'intrinsic': intrinsic, 'extrinsic': extrinsic, 'C': C, 'R': R, 't': t, 'W': W, 'H': H, 'num_frames': num_frames
                }
        elif dataset_name == 'fit3d':
            fit3d_root = f'/home/{user}/Datasets/HAAI/Fit3D/train'
            for source in tqdm(source_list):
                subject, cam_id, action = split_source_name(source, dataset_name)
                assert cam_id != None, f'cam_id is None for {source}'
                assert action != None, f'action is None for {source}'
                if subject not in cam_params.keys():         cam_params[subject] = {}
                if action not in cam_params[subject].keys(): cam_params[subject][action] = {}
                subject_root = os.path.join(fit3d_root, subject)
                gt_3d_path = os.path.join(fit3d_root, subject, 'joints3d_25', action+'.json')
                cam_path = os.path.join(subject_root, 'camera_parameters', cam_id, action+'.json')
                cam_param = readJSON(cam_path)
                R = np.array(cam_param['extrinsics']['R']) # (3, 3)
                C = np.array(cam_param['extrinsics']['T'])[0] # unit: m
                t = -R @ C
                cx, cy = cam_param['intrinsics_wo_distortion']['c']
                fx, fy = cam_param['intrinsics_wo_distortion']['f']
                intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) # (3, 3)
                extrinsic = np.hstack([R, t.reshape(-1, 1)]) # (3, 4)
                video_path = os.path.join(subject_root, 'videos', cam_id, action+'.mp4')
                W, H, _, fps = get_video_info(video_path)
                num_frames = len(np.array(readJSON(gt_3d_path)['joints3d_25'])[:, :17])
                cam_params[subject][action][cam_id] = {
                    'intrinsic': intrinsic, 'extrinsic': extrinsic, 'C': C, 'R': R, 't': t, 'W': W, 'H': H, 'num_frames': num_frames, 'fps': fps
                }
        elif dataset_name == '3dhp':
            data_dict_3dhp_train, cam_param_3dhp_train = load_3dhp_original('train')
            data_dict_3dhp_test, cam_param_3dhp_test = load_3dhp_original('test')
            data_dict_3dhp = {**data_dict_3dhp_train, **data_dict_3dhp_test}
            cam_param_3dhp = {**cam_param_3dhp_train, **cam_param_3dhp_test}
            print(cam_param_3dhp.keys())
            for source in tqdm(source_list):
                subject, cam_id, seq = split_source_name(source, dataset_name)
                if subject not in cam_params.keys():      cam_params[subject] = {}
                if seq not in cam_params[subject].keys(): cam_params[subject][seq] = {}
                if cam_id == None:
                    if only_visible_frame:
                        num_frames = data_dict_3dhp[f'{subject}']['visible_frame'].sum()
                    else:
                        num_frames = data_dict_3dhp[f'{subject}']['annot3'].shape[0] # test data
                    cam_params[subject][seq][cam_id] = cam_param_3dhp[subject].copy() # copy 안하면 원본 데이터가 바뀜
                else:
                    if only_visible_frame:
                        num_frames = data_dict_3dhp[f'{subject}_{seq}_{cam_id}']['visible_frame'].sum()
                    else:
                        num_frames = data_dict_3dhp[f'{subject}_{seq}_{cam_id}']['annot3'].shape[0] # train data
                    cam_params[subject][seq][cam_id] = cam_param_3dhp[cam_id].copy() # copy 안하면 원본 데이터가 바뀜

                cam_params[subject][seq][cam_id]['num_frames'] = num_frames
        else:
            raise ValueError(f'{dataset_name} not found')
        if not no_save: savepkl(cam_params, save_path_cam_params)
    return cam_params

def load_image_frame(dataset_name, source, frame_num):
    from my_utils import split_source_name, get_video_frame, load_cam_params
    user = getpass.getuser()
    subject, cam_id, action = split_source_name(source, dataset_name)
    cam_params = load_cam_params(dataset_name, f'/home/{user}/codes/MotionBERT/data/motion3d')
    num_frames = cam_params[subject][action][cam_id]['num_frames']
    # get video frame
    if dataset_name == '3dhp':
        if frame_num < 0: frame_num = num_frames + frame_num
        if 'TS' in subject: # testset
            img = cv2.imread(f'/home/{user}/Datasets/HAAI/3DHP/original/test/{subject}/imageSequence/img_{frame_num+1:06d}.jpg')
        else: # trainset
            assert cam_id != None, f'cam_id is None for {source}'
            assert action != None, f'action is None for {source}'
            video_path = f'/home/{user}/Datasets/HAAI/3DHP/original/train/{subject}/{action}/imageSequence/video_{cam_id.split("cam")[1]}.avi'
            img = get_video_frame(video_path, frame_num)
    elif dataset_name == 'fit3d':
        video_path = f'/home/{user}/Datasets/HAAI/Fit3D/train/{subject}/videos/{cam_id}/{action}.mp4'
        img = get_video_frame(video_path, frame_num)
    else:
        img = None
    return img

def load_cam_3d(dataset_name, save_paths, overwrite=False, no_save=False, univ=False, only_visible_frame=False,
                data_aug={'step_rot': 0,
                        'sinu_yaw_mag': 0, 'sinu_yaw_period': 273, 'sinu_pitch_mag': 0, 'sinu_pitch_period': 273,
                        'sinu_roll_mag': 0, 'sinu_roll_period': 273,'rand_yaw_mag': 0, 'rand_yaw_period': 0,
                        'rand_pitch_mag': 0, 'rand_pitch_period': 0,'rand_roll_mag': 0, 'rand_roll_period': 0
                        }):
    from my_utils import load_3dhp_original, data_augmentation, readpkl, savepkl, split_source_name, get_video_info
    import random
    random.seed(0)
    # pkl path
    save_path_cam_3d = save_paths['cam_3d']
    # load data
    if os.path.exists(save_path_cam_3d) and not overwrite and not no_save:
        cam_3ds = readpkl(save_path_cam_3d)
    else:
        # prerequisites
        save_path_source_list = save_paths['source_list']
        assert os.path.exists(save_path_source_list), f'No source_list found for {dataset_name}'
        source_list = readpkl(save_path_source_list)
        cam_3ds = {}
        if dataset_name == '3dhp':
            data_dict_3dhp_train, _ = load_3dhp_original('train', overwrite=True)
            data_dict_3dhp_test, _ = load_3dhp_original('test', overwrite=True)
            data_dict_3dhp = {**data_dict_3dhp_train, **data_dict_3dhp_test}
            #num_frames = 0
            for source in tqdm(source_list):
                subject, cam_id, seq = split_source_name(source, '3dhp')
                if seq is not None: source = f'{subject}_{seq}_{cam_id}'
                else: source = subject
                if subject not in cam_3ds.keys():      cam_3ds[subject] = {}
                if seq not in cam_3ds[subject].keys(): cam_3ds[subject][seq] = {}
                annot_type = 'univ_annot3' if univ else 'annot3'
                if only_visible_frame:
                    visible_frame = data_dict_3dhp[source]['visible_frame']
                    cam_3d = data_dict_3dhp[source][annot_type][visible_frame]/1000
                else:
                    cam_3d = data_dict_3dhp[source][annot_type]/1000
                # data augmentation
                cam_3d = data_augmentation(cam_3d, data_aug)
                # store
                cam_3ds[subject][seq][cam_id] = cam_3d
        else:
            # prerequisites
            save_path_world_3d = save_paths['world_3d']
            save_path_cam_params = save_paths['cam_param']
            assert os.path.exists(save_path_world_3d), f'No world_3d found for {dataset_name}'
            assert os.path.exists(save_path_cam_params), f'No cam_params found for {dataset_name}'
            world_3ds = readpkl(save_path_world_3d)
            cam_params = readpkl(save_path_cam_params)
            for source in tqdm(source_list):
                subject, cam_id, action = split_source_name(source, dataset_name)
                if subject not in cam_3ds:          cam_3ds[subject] = {}
                if action  not in cam_3ds[subject]: cam_3ds[subject][action] = {}
                # world_3d
                world_3d = world_3ds[subject][action]
                # cam_param
                cam_param = cam_params[subject][action][cam_id]
                R, t = cam_param['R'], cam_param['t']
                # world to cam
                cam_3d = np.einsum('ijk,kl->ijl', world_3d, R.T) + t # (N, 17, 3)
                # data augmentation
                cam_3d = data_augmentation(cam_3d, data_aug)
                # store
                cam_3ds[subject][action][cam_id] = cam_3d.copy()
        if not no_save: savepkl(cam_3ds, save_path_cam_3d)
    return cam_3ds

def load_img_2d(dataset_name, save_paths, overwrite=False, no_save=False, only_visible_frame=False,
                data_aug={'step_rot': 0,
                        'sinu_yaw_mag': 0, 'sinu_yaw_period': 273, 'sinu_pitch_mag': 0, 'sinu_pitch_period': 273,
                        'sinu_roll_mag': 0, 'sinu_roll_period': 273,'rand_yaw_mag': 0, 'rand_yaw_period': 0,
                        'rand_pitch_mag': 0, 'rand_pitch_period': 0,'rand_roll_mag': 0, 'rand_roll_period': 0
                        }):
    from my_utils import load_3dhp_original, readpkl, savepkl, projection
    # pkl path
    save_path_img_2d = save_paths['img_2d']
    # load data
    if os.path.exists(save_path_img_2d) and not overwrite and not no_save:
        img_2ds = readpkl(save_path_img_2d)
    else:
        # prerequisites
        save_path_source_list = save_paths['source_list']
        assert os.path.exists(save_path_source_list), f'No source_list found for {dataset_name}'
        source_list = readpkl(save_path_source_list)

        img_2ds = {}
        if dataset_name == '3dhp':
            data_dict_3dhp_train, cam_param_3dhp_train = load_3dhp_original('train')
            data_dict_3dhp_test, cam_param_3dhp_test = load_3dhp_original('test')
            data_dict_3dhp = {**data_dict_3dhp_train, **data_dict_3dhp_test}
            for source in tqdm(source_list):
                subject, cam_id, seq = split_source_name(source, '3dhp')
                if seq is not None: source = f'{subject}_{seq}_{cam_id}'
                else: source = subject
                if subject not in img_2ds.keys():      img_2ds[subject] = {}
                if seq not in img_2ds[subject].keys(): img_2ds[subject][seq] = {}
                if only_visible_frame:
                    visible_frame = data_dict_3dhp[source]['visible_frame']
                    img_2ds[subject][seq][cam_id] = data_dict_3dhp[source]['annot2'][visible_frame]
                else:
                    img_2ds[subject][seq][cam_id] = data_dict_3dhp[source]['annot2']
        else:
            save_path_cam_3d = save_paths['cam_3d']
            save_path_cam_params = save_paths['cam_param']
            assert os.path.exists(save_path_cam_3d), f'No cam_3d found for {dataset_name}'
            assert os.path.exists(save_path_cam_params), f'No cam_params found for {dataset_name}'
            cam_3ds = readpkl(save_path_cam_3d)
            cam_params = readpkl(save_path_cam_params)
            for source in tqdm(source_list):
                subject, cam_id, action = split_source_name(source, dataset_name)
                if subject not in img_2ds:          img_2ds[subject] = {}
                if action  not in img_2ds[subject]: img_2ds[subject][action] = {}
                # cam_3d
                cam_3d = cam_3ds[subject][action][cam_id]
                # cam_param
                cam_param = cam_params[subject][action][cam_id]
                intrinsic = np.array(cam_param['intrinsic'])
                # cam to img
                img_2d = projection(cam_3d, intrinsic) # (N, 17, 2)
                # store
                img_2ds[subject][action][cam_id] = img_2d
        if not no_save: savepkl(img_2ds, save_path_img_2d)
    return img_2ds

def load_cam_3d_canonical(dataset_name, save_paths, canonical_type, overwrite=False, no_save=False,
                          data_aug={'step_rot': 0,
                                    'sinu_yaw_mag': 0, 'sinu_yaw_period': 273, 'sinu_pitch_mag': 0, 'sinu_pitch_period': 273,
                                    'sinu_roll_mag': 0, 'sinu_roll_period': 273,'rand_yaw_mag': 0, 'rand_yaw_period': 0,
                                    'rand_pitch_mag': 0, 'rand_pitch_period': 0,'rand_roll_mag': 0, 'rand_roll_period': 0
                                    }):
    from my_utils import canonicalization_cam_3d, readpkl, savepkl
    import random
    random.seed(0)
    assert canonical_type is not None, 'canonical_type is None'
    # prerequisites
    save_path_cam_3d = save_paths['cam_3d']
    save_path_source_list = save_paths['source_list']
    assert os.path.exists(save_path_cam_3d), f'No cam_3d found for {dataset_name}'
    assert os.path.exists(save_path_source_list), f'No source_list found for {dataset_name}'
    cam_3ds = readpkl(save_path_cam_3d)
    source_list = readpkl(save_path_source_list)

    cam_3d_canonicals = {}
    if canonical_type == 'pcl':
        cam_3d_canonicals = cam_3ds
    else:
        # prerequisites
        save_path_cam_3d_canonical = save_paths['cam_3d_canonical'] # pkl path
        if os.path.exists(save_path_cam_3d_canonical) and not overwrite and not no_save:
            cam_3d_canonicals = readpkl(save_path_cam_3d_canonical) # load data
        else:
            for source in tqdm(source_list):
                subject, cam_id, action = split_source_name(source, dataset_name)
                if subject not in cam_3d_canonicals:          cam_3d_canonicals[subject] = {}
                if action  not in cam_3d_canonicals[subject]: cam_3d_canonicals[subject][action] = {}
                # load cam_3d
                cam_3d = cam_3ds[subject][action][cam_id]
                # canonicalization
                cam_3d_canonical = canonicalization_cam_3d(cam_3d, canonical_type)
                # store
                cam_3d_canonicals[subject][action][cam_id] = cam_3d_canonical
            if not no_save: savepkl(cam_3d_canonicals, save_path_cam_3d_canonical)
    return cam_3d_canonicals

def load_img_2d_canonical(dataset_name, save_paths, canonical_type, overwrite=False, no_save=False, adaptive_focal=False,
                          data_aug={'step_rot': 0,
                        'sinu_yaw_mag': 0, 'sinu_yaw_period': 273, 'sinu_pitch_mag': 0, 'sinu_pitch_period': 273,
                        'sinu_roll_mag': 0, 'sinu_roll_period': 273,'rand_yaw_mag': 0, 'rand_yaw_period': 0,
                        'rand_pitch_mag': 0, 'rand_pitch_period': 0,'rand_roll_mag': 0, 'rand_roll_period': 0
                        }):
    from my_utils import readpkl, savepkl, projection, genertate_pcl_img_2d
    # prerequisites
    assert canonical_type is not None, 'canonical_type is None'
    # pkl path
    if adaptive_focal: save_path_img_2d_canonical = save_paths['img_2d_canonical_adaptive_focal']
    else:save_path_img_2d_canonical = save_paths['img_2d_canonical']
    # load data
    if os.path.exists(save_path_img_2d_canonical) and not overwrite and not no_save:
        img_2d_canonicals = readpkl(save_path_img_2d_canonical)
    else:
        # prerequisites
        save_path_source_list = save_paths['source_list']
        if adaptive_focal: save_path_cam_params = save_paths['cam_param_adaptive_focal']
        else: save_path_cam_params = save_paths['cam_param']
        assert os.path.exists(save_path_source_list), f'No source_list found for {dataset_name}'
        assert os.path.exists(save_path_cam_params), f'No cam_params found for {dataset_name}'
        source_list = readpkl(save_path_source_list)
        cam_params = readpkl(save_path_cam_params)

        img_2d_canonicals = {}
        if canonical_type == 'pcl':
            save_path_img_2d = save_paths['img_2d']
            assert os.path.exists(save_path_img_2d), f'No img_2d found for {dataset_name}'
            img_2ds = readpkl(save_path_img_2d)

            for source in tqdm(source_list):
                subject, cam_id, action = split_source_name(source, dataset_name)
                if subject not in img_2d_canonicals:          img_2d_canonicals[subject] = {}
                if action  not in img_2d_canonicals[subject]: img_2d_canonicals[subject][action] = {}
                img_2d = img_2ds[subject][action][cam_id].copy()
                cam_param = cam_params[subject][action][cam_id].copy()
                img_2d_canonical = genertate_pcl_img_2d(img_2d, cam_param)
                img_2d_canonicals[subject][action][cam_id] = img_2d_canonical
            if not no_save: savepkl(img_2d_canonicals, save_path_img_2d_canonical)
        else:
            save_path_cam_3d_canonical = save_paths['cam_3d_canonical']
            assert os.path.exists(save_path_cam_3d_canonical), f'No cam_3d_canonical {canonical_type} found for {dataset_name}'
            cam_3d_canonicals = readpkl(save_path_cam_3d_canonical)

            for source in tqdm(source_list):
                subject, cam_id, action = split_source_name(source, dataset_name)
                if subject not in img_2d_canonicals:          img_2d_canonicals[subject] = {}
                if action  not in img_2d_canonicals[subject]: img_2d_canonicals[subject][action] = {}
                cam_3d_canonical = cam_3d_canonicals[subject][action][cam_id]
                cam_param = cam_params[subject][action][cam_id]
                intrinsic = np.array(cam_param['intrinsic'])
                img_2d_canonical = projection(cam_3d_canonical, intrinsic)
                img_2d_canonicals[subject][action][cam_id] = img_2d_canonical
            if not no_save: savepkl(img_2d_canonicals, save_path_img_2d_canonical)

    return img_2d_canonicals

def load_world_3d(dataset_name, save_paths, overwrite=False, no_save=False):
    from my_utils import readpkl, savepkl, split_source_name, readJSON
    from posynda_utils import Human36mDataset
    user = getpass.getuser()
    # pkl path
    save_path_world_3d = save_paths['world_3d']
    if os.path.exists(save_path_world_3d) and not overwrite and not no_save:
        world_3ds = readpkl(save_path_world_3d)
    else:
        # prerequisites
        save_path_source_list = save_paths['source_list']
        assert os.path.exists(save_path_source_list), f'No source_list found for {dataset_name}'
        source_list = readpkl(save_path_source_list)

        world_3ds = {}
        if dataset_name == 'h36m':
            if 'h36m_dataset' in globals(): del globals()['h36m_dataset']
            if 'h36m_dataset' in locals(): del locals()['h36m_dataset']
            h36m_dataset = Human36mDataset('/home/hrai/codes/hpe_library/data/data_3d_h36m.npz', remove_static_joints=True)._data
            for source in source_list:
                subject, cam_id, action = split_source_name(source, dataset_name)
                if subject not in world_3ds.keys(): world_3ds[subject] = {}
                world_3ds[subject][action] = h36m_dataset[subject][action]['positions'].copy()
        elif dataset_name == 'fit3d':
            fit3d_root = f'/home/{user}/Datasets/HAAI/Fit3D/train'
            for source in tqdm(source_list):
                subject, cam_id, action = split_source_name(source, dataset_name)
                assert cam_id != None, f'cam_id is None for {source}'
                assert action != None, f'action is None for {source}'
                if subject not in world_3ds.keys(): world_3ds[subject] = {}
                gt_3d_path = os.path.join(fit3d_root, subject, 'joints3d_25', action+'.json')
                gt_3d = np.array(readJSON(gt_3d_path)['joints3d_25'])[:, :17] # (F, 17, 3)
                world_3ds[subject][action] = gt_3d
        elif dataset_name == '3dhp':
            # prerequisites
            save_path_cam_3d = save_paths['cam_3d']
            save_path_cam_params = save_paths['cam_param']
            assert os.path.exists(save_path_cam_3d), f'No cam_3d found for {dataset_name} {save_path_cam_3d}'
            assert os.path.exists(save_path_cam_params), f'No cam_params found for {dataset_name} {save_path_cam_params}'
            cam_3ds = readpkl(save_path_cam_3d)
            cam_params = readpkl(save_path_cam_params)
            for source in tqdm(source_list):
                subject, cam_id, action = split_source_name(source, dataset_name)
                if subject not in world_3ds: world_3ds[subject] = {}
                if action in world_3ds[subject]: continue # already generated from other cam_3d
                # cam_3d & cam_param
                cam_3d = cam_3ds[subject][action][cam_id]
                cam_param = cam_params[subject][action][cam_id]
                R, C = cam_param['R'], cam_param['C']
                # cam to world
                world_3d = np.einsum('ijk,kl->ijl', cam_3d, R) + C # (N, 17, 3)
                # store
                world_3ds[subject][action] = world_3d.copy()
        if not no_save: savepkl(world_3ds, save_path_world_3d)
    return world_3ds

def load_img_3d(dataset_name, save_paths, overwrite=False, no_save=False):
    from my_utils import readpkl, savepkl, split_source_name
    save_path_img_3d = save_paths['img_3d']
    if os.path.exists(save_path_img_3d) and not overwrite and not no_save:
        img_3ds = readpkl(save_path_img_3d)
    else:
        # prerequisites
        save_path_cam_3d = save_paths['cam_3d']
        save_path_img_2d = save_paths['img_2d']
        save_path_cam_params = save_paths['cam_param']
        save_path_source_list = save_paths['source_list']
        assert os.path.exists(save_path_cam_3d), f'No cam_3d found for {dataset_name}'
        assert os.path.exists(save_path_img_2d), f'No img_2d found for {dataset_name}'
        assert os.path.exists(save_path_cam_params), f'No cam_params found for {dataset_name}'
        assert os.path.exists(save_path_source_list), f'No source_list found for {dataset_name}'
        cam_3ds = readpkl(save_path_cam_3d)
        img_2ds = readpkl(save_path_img_2d)
        cam_params = readpkl(save_path_cam_params)
        source_list = readpkl(save_path_source_list)

        img_3ds = {}
        for source in tqdm(source_list):
            subject, cam_id, action = split_source_name(source, dataset_name)
            if subject not in img_3ds:          img_3ds[subject] = {}
            if action  not in img_3ds[subject]: img_3ds[subject][action] = {}
            # cam_3d, img_2d
            cam_3d = cam_3ds[subject][action][cam_id] * 1000 # m -> mm (must be in mm)
            img_2d = img_2ds[subject][action][cam_id]
            # cam_param
            cam_param = cam_params[subject][action][cam_id]
            intrinsic = cam_param['intrinsic']
            # img_3d depth
            root_joint = cam_3d[:, 0] # (N, 3)
            tl_joint = root_joint.copy() # top left point of the bounding box
            br_joint = root_joint.copy() # bottom right point of the bounding box
            tl_joint[:, :2] -= 1000.0
            br_joint[:, :2] += 1000.0
            tl_2d = tl_joint @ intrinsic.T # projected top left point
            tl_2d = tl_2d / tl_2d[:, 2:]
            br_2d = br_joint @ intrinsic.T # projected bottom right point
            br_2d = br_2d / br_2d[:, 2:]
            box = np.stack([tl_2d[:, 0], tl_2d[:, 1], br_2d[:, 0], br_2d[:, 1]], axis=1) # (N, 4) - top left x, top left y, bottom right x, bottom right y
            ratio = (box[:, 2] - box[:, 0] + 1) / 2000.0 # (N,)
            img_3d_depth = ratio.reshape(-1, 1)*(cam_3d[...,2] - cam_3d[:,0:1,2]) # (N, 17, 1)
            # img_3d
            img_3d = np.zeros_like(cam_3d)
            img_3d[...,:2] = img_2d.copy()
            img_3d[...,2] = img_3d_depth.copy()
            # store
            img_3ds[subject][action][cam_id] = img_3d.copy()
        if not no_save: savepkl(img_3ds, save_path_img_3d)
        #print(f'{dataset_name} img_3d generated and saved\n')
    return img_3ds

def load_scale_factor(dataset_name, save_paths, overwrite=False, no_save=False):
    from my_utils import readpkl, savepkl, split_source_name, optimize_scaling_factor
    save_path_scale_factor = save_paths['scale_factor']
    if os.path.exists(save_path_scale_factor) and not overwrite and not no_save:
        scale_factors = readpkl(save_path_scale_factor)
    else:
        save_path_cam_3d = save_paths['cam_3d']
        save_path_img_3d = save_paths['img_3d']
        save_path_cam_params = save_paths['cam_param']
        save_path_source_list = save_paths['source_list']
        assert os.path.exists(save_path_cam_3d), f'No cam_3d found for {dataset_name}'
        assert os.path.exists(save_path_img_3d), f'No img_3d found for {dataset_name}'
        assert os.path.exists(save_path_cam_params), f'No cam_params found for {dataset_name}'
        assert os.path.exists(save_path_source_list), f'No source_list found for {dataset_name}'
        source_list = readpkl(save_path_source_list)
        cam_params = readpkl(save_path_cam_params)
        cam_3ds = readpkl(save_path_cam_3d)
        img_3ds = readpkl(save_path_img_3d)

        scale_factors = {}
        for source in tqdm(source_list):
            subject, cam_id, action = split_source_name(source, dataset_name)
            if subject not in scale_factors:          scale_factors[subject] = {}
            if action  not in scale_factors[subject]: scale_factors[subject][action] = {}
            #cam_param = cam_params[subject][action][cam_id]
            # R, t, C, W, H, intrinsic = cam_param['R'], cam_param['t'], cam_param['C'], cam_param['W'], cam_param['H'], cam_param['intrinsic']
            cam_3d = cam_3ds[subject][action][cam_id]*1000
            img_3d = img_3ds[subject][action][cam_id]
            cam_3d_hat = cam_3d - cam_3d[:, 0:1, :]
            img_3d_hat = img_3d - img_3d[:, 0:1, :]

            scale_factor = []
            for frame_num in range(cam_3d.shape[0]):
                pred_lambda, losses = optimize_scaling_factor(img_3d_hat[frame_num], cam_3d_hat[frame_num]) # x,y,z 사용
                scale_factor.append(pred_lambda)
                #pred_lambda, losses = optimize_scaling_factor(cam_3d_hat[frame_num], img_3d_hat[frame_num], learningRate=0.000005) # x,y,z 사용
                #scale_factor.append(1/pred_lambda)
                #pred_lambda, losses3 = optimize_scaling_factor(img_3d_hat[frame_num], cam_3d_hat[frame_num], learningRate=0.00001) # x,y,z 사용

            scale_factors[subject][action][cam_id] = np.array(scale_factor) # (N,)
        if not no_save: savepkl(scale_factors, save_path_scale_factor)

    return scale_factors

def load_img25d(dataset_name, save_paths, overwrite=False, no_save=False):
    from my_utils import readpkl, savepkl, split_source_name
    save_path_img_25d = save_paths['img_25d']
    if os.path.exists(save_path_img_25d) and not overwrite and not no_save:
        img_25ds = readpkl(save_path_img_25d)
    else:
        save_path_img_3d = save_paths['img_3d']
        save_path_scale_factor = save_paths['scale_factor']
        save_path_cam_params = save_paths['cam_param']
        save_path_source_list = save_paths['source_list']
        assert os.path.exists(save_path_img_3d), f'No img_3d found for {dataset_name}'
        assert os.path.exists(save_path_scale_factor), f'No scale_factor found for {dataset_name}'
        assert os.path.exists(save_path_cam_params), f'No cam_params found for {dataset_name}'
        assert os.path.exists(save_path_source_list), f'No source_list found for {dataset_name}'
        source_list = readpkl(save_path_source_list)
        cam_params = readpkl(save_path_cam_params)
        img_3ds = readpkl(save_path_img_3d)
        scale_factors = readpkl(save_path_scale_factor)

        img_25ds = {}
        for source in tqdm(source_list):
            subject, cam_id, action = split_source_name(source, dataset_name)
            if subject not in img_25ds:          img_25ds[subject] = {}
            if action  not in img_25ds[subject]: img_25ds[subject][action] = {}
            cam_3d = img_3ds[subject][action][cam_id]
            scale_factor = scale_factors[subject][action][cam_id]
            img_25d = cam_3d.copy() * scale_factor[:, None, None]
            img_25ds[subject][action][cam_id] = img_25d
        if not no_save: savepkl(img_25ds, save_path_img_25d)

    return img_25ds

def load_h36m():
    from posynda_utils import Human36mDataset
    from my_utils import readJSON
    # camera parameters
    cam_param = readJSON('/home/hrai/codes/hpe_library/data/h36m_camera-parameters.json')
    print('==> Loading 3D data wrt World CS...')
    #if 'h36m_3d_world' not in globals(): del globals()['h36m_3d_world']
    #if 'h36m_3d_world' not in locals(): del locals()['h36m_3d_world']
    h36m_3d_world = Human36mDataset('/home/hrai/codes/hpe_library/data/data_3d_h36m.npz', remove_static_joints=True)._data

    return h36m_3d_world, cam_param

def load_fit3d_one_video(fit3d_root, cam_num, data_type='train', subject='s03', action='burpees'):
    from my_utils import readJSON
    # load gt 3d
    subject_path = os.path.join(fit3d_root, data_type, subject)
    gt_3d_path = os.path.join(subject_path, 'joints3d_25', action + '.json')
    gt_3d = readJSON(gt_3d_path)['joints3d_25']
    # read camera parameter
    if data_type == 'train':
        cam_parameter_path = os.path.join(subject_path, 'camera_parameters', cam_num, action + '.json')
    elif data_type == 'test':
        cam_parameter_path = os.path.join(subject_path, 'camera_parameters', action + '.json')
    cam_param = readJSON(cam_parameter_path)
    R = np.array(cam_param['extrinsics']['R'])
    C = np.array(cam_param['extrinsics']['T']).T
    t = -R @ C
    extrinsic_matrix = np.concatenate([R, t], axis=1)
    cx, cy = cam_param['intrinsics_wo_distortion']['c']
    fx, fy = cam_param['intrinsics_wo_distortion']['f']
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    camera_param = {'extrinsic': extrinsic_matrix, 'intrinsic': intrinsic_matrix}
    return gt_3d, camera_param

def get_cam_param(camera_sub_act, subject, cam_params):
    cam_param_for_sub_act = {}
    for i in range(4):
        cam_info = camera_sub_act[i]
        cam_id = cam_info['id']
        cam_ext = cam_params['extrinsics'][subject][cam_id]
        cam_int = cam_params['intrinsics'][cam_id]
        cam_origin_from_world = - np.array(cam_ext['R']).T @ np.array(cam_ext['t'])*0.001
        cam_proj = np.array(cam_int['calibration_matrix']) @ np.hstack([np.array(cam_ext['R']), np.array(cam_ext['t'])*0.001])
        cam_param_for_sub_act[cam_id] = {'proj': cam_proj, 'ext': cam_ext, 'int': cam_int, 'id': cam_id, 'W': cam_info['res_w'], 'H': cam_info['res_h'], 'C': cam_origin_from_world}
    return cam_param_for_sub_act

def get_pose_seq_and_cam_param(h36m_3d_world, h36m_cam_param, subject, action):
    pose3d = h36m_3d_world[subject][action]['positions'] # 3d skeleton sequence wrt world CS
    cam_info = h36m_3d_world[subject][action]['cameras']
    cam_param = get_cam_param(cam_info, subject, h36m_cam_param)
    return pose3d, cam_param

def get_part_traj(pose_traj, part):
    from my_utils import get_h36m_keypoint_index
    # h36m 기준
    pelvis_idx = get_h36m_keypoint_index('Pelvis')
    l_hip_idx = get_h36m_keypoint_index('L_Hip')
    l_shoulder_idx = get_h36m_keypoint_index('L_Shoulder')
    r_shoulder_idx = get_h36m_keypoint_index('R_Shoulder')
    r_hip_idx = 1
    if part == 'torso':
        part_traj = pose_traj[:, [pelvis_idx, l_hip_idx, l_shoulder_idx, r_shoulder_idx, r_hip_idx], :]
    elif part == 'lower_line':
        part_traj = pose_traj[:, [pelvis_idx, l_hip_idx, r_hip_idx], :]
    elif part == 'all':
        part_traj = {}
        part_traj['torso'] = pose_traj[:, [pelvis_idx, l_hip_idx, l_shoulder_idx, r_shoulder_idx, r_hip_idx], :]
        part_traj['pelvis'] = pose_traj[:, pelvis_idx, :].reshape(-1, 1, 3)
        part_traj['l_hip'] = pose_traj[:, l_hip_idx, :].reshape(-1, 1, 3)
        part_traj['l_shoulder'] = pose_traj[:, l_shoulder_idx, :].reshape(-1, 1, 3)
        part_traj['r_shoulder'] = pose_traj[:, r_shoulder_idx, :].reshape(-1, 1, 3)
        part_traj['r_hip'] = pose_traj[:, r_hip_idx, :].reshape(-1, 1, 3)
        part_traj['lower_line'] = pose_traj[:, [pelvis_idx, l_hip_idx, r_hip_idx], :]
    else:
        try:
            part_idx = get_h36m_keypoint_index(part)
            part_traj = pose_traj[:, part_idx, :].reshape(-1, 1, 3)
        except:
            raise ValueError('part not found')
    return part_traj

def get_aligned_init_torso(torso, forward, height=0.86):
    from my_utils import get_torso_rotation_matrix, rotation_matrix_to_vector_align, rotate_torso_by_R, get_torso_direction
    # align to world origin
    rot_from_world = get_torso_rotation_matrix(torso)
    align_R = rot_from_world.T
    aligned_init_torso = rotate_torso_by_R(torso, align_R)
    # move to world origin
    aligned_init_torso -= aligned_init_torso[0]
    # torso height
    aligned_init_torso[:, 2] += height
    # shoulder height -> same
    aligned_init_torso[2, 2] = aligned_init_torso[3, 2]
    # hip height -> same
    aligned_init_torso[1, 2] = aligned_init_torso[4, 2]
    # shoulder symmetry
    aligned_init_torso[2, 0] = 0
    aligned_init_torso[3, 0] = 0
    aligned_init_torso[2, 1] = -aligned_init_torso[3, 1]
    # align to forward direction
    direction = get_torso_direction(aligned_init_torso)
    rot_to_forward = rotation_matrix_to_vector_align(direction, forward)
    aligned_init_torso = rotate_torso_by_R(aligned_init_torso, rot_to_forward)

    return aligned_init_torso

def generate_random_trajectory(seed_offest, start_torso, cam_proj, num_points=5000, max_deg=5, max_dist=10, max_bias=5, step_size=0.01, rt_type='rxryrztxtytz', seed=None):
    from my_utils import get_torso_rotation_matrix, rotate_torso_by_R, projection
    # max_dist [cm]
    cm_to_m = 0.01
    start_point = start_torso[0]
    rot_from_world = get_torso_rotation_matrix(start_torso)
    #init_pelvis_direction = get_torso_direction(start_torso)

    if seed is None:
        np.random.seed(int(time.time()*1000000)-seed_offest)
    else:
        np.random.seed(seed)
    z = start_point.copy()[2] # fixed height

    prev_point = start_point.copy()
    prev_rot = rot_from_world.copy()
    prev_torso = start_torso.copy()

    points = [prev_point]
    rots = [prev_rot]
    torsos = [prev_torso]
    if isinstance(max_dist, int):
        max_dist_x = max_dist
        max_dist_y = max_dist
        max_dist_z = max_dist
    elif isinstance(max_dist, list):
        assert len(max_dist) == 3, 'max_dist should be [max_dist_x, max_dist_y, max_dist_z]'
        max_dist_x, max_dist_y, max_dist_z = max_dist
    else:
        max_dist_x, max_dist_y, max_dist_z = 5, 5, 5

    for i in range(0, num_points-1):
        # bias is changed at every 20 steps / max_bias should be positive
        bias = np.random.randint(-max_bias, max_bias) if i % int(num_points/20) == 0 and max_bias > 0 else 0

        delta = np.zeros(3)
        if 'tx' in rt_type: delta[0] += np.random.randint(-max_dist_x, max_dist_x+1 + bias)*cm_to_m
        if 'ty' in rt_type: delta[1] += np.random.randint(-max_dist_y, max_dist_y+1 + bias)*cm_to_m
        if 'tz' in rt_type: delta[2] += np.random.randint(-max_dist_z, max_dist_z+1 + bias)*cm_to_m
        delta *= step_size

        roll  = np.random.randint(-max_deg, max_deg+1 + bias) if 'rx' in rt_type and (-max_deg < max_deg+1 + bias) else 0 # high should be bigger than low for random.randint
        pitch = np.random.randint(-max_deg, max_deg+1 + bias) if 'ry' in rt_type and (-max_deg < max_deg+1 + bias) else 0 # high should be bigger than low for random.randint
        yaw   = np.random.randint(-max_deg, max_deg+1 + bias) if 'rz' in rt_type and (-max_deg < max_deg+1 + bias) else 0 # high should be bigger than low for random.randint

        delta_rot = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
        rot = delta_rot @ prev_rot
        if 'fwd' in rt_type:
            direction = rot[:, 0]
            delta = direction*step_size # translation vector

        point = prev_point + delta # new pelvis position
        torso = rotate_torso_by_R(prev_torso, delta_rot) + np.repeat(delta.reshape(1,3), 5, axis=0) # rotation + translation

        points.append(point.copy())
        rots.append(rot.copy())
        torsos.append(torso.copy())
        prev_point = point.copy()
        prev_rot = rot.copy()
        prev_torso = torso.copy()

    points = np.array(points)
    rots = np.array(rots)

    # move traj to start point
    x_mean, y_mean, z_mean = points[:, 0].mean(), points[:, 1].mean(), points[:, 2].mean()
    # x_bias, y_bias = np.random.rand(2)-0.5
    # x_mean += x_bias
    # y_mean += y_bias
    points[:, 0] -= (x_mean-start_point[0])
    points[:, 1] -= (y_mean-start_point[1])
    points[:, 2] -= (z_mean-start_point[2])
    torsos = np.array(torsos)
    torsos[:, :, 0] -= (x_mean-start_point[0])
    torsos[:, :, 1] -= (y_mean-start_point[1])
    torsos[:, :, 2] -= (z_mean-start_point[2])

    # project to 2d
    torsos_projected = projection(torsos, cam_proj)

    return points, rots, torsos, torsos_projected

def generate_random_segment(seed_offest, start_torso, cam_proj, num_points, max_deg, max_bias, max_dist, step_size, rt_type, seed, xyz_range=[0.5, 0.5, 0.2]):

    points, rots, torsos, torsos_projected = generate_random_trajectory(seed_offest, start_torso, cam_proj, num_points=num_points, max_deg=max_deg, max_bias=max_bias, max_dist=max_dist,  step_size=step_size, rt_type=rt_type, seed=seed)

    # exclude points outside of image, -bound<= x,y <=bound
    segments_idx, pass_list, delete_list = get_bounded_segments_idx(points, torsos_projected, bound_center=start_torso[0], range_x=xyz_range[0], range_y=xyz_range[1], range_z=xyz_range[2])

    traj_segments = []
    num_pass_points = 0
    for i in range(len(segments_idx)):
        segment = {
            'points': points[segments_idx[i], :].copy(),
            'rots': rots[segments_idx[i], :].copy(),
            'torsos': torsos[segments_idx[i], :, :].copy(),
            'torsos_projected': torsos_projected[segments_idx[i], :, :].copy()}
        traj_segments.append(segment)
        num_pass_points += len(segments_idx[i])
    return points, rots, torsos, torsos_projected, traj_segments, pass_list, delete_list, num_pass_points

# split conginuous idx groups
def split_continuous_indices(lst):
    result = []
    sub_list = []
    for i, item in enumerate(lst):
        if i > 0 and lst[i-1] != item - 1:
            result.append(sub_list)
            sub_list = []
        sub_list.append(item)
    if sub_list:
        result.append(sub_list)

    return result

# exclude points outside of image, -bound<= x,y <=bound
def get_bounded_segments_idx(points, torsos_projected, bound_center, range_x=1, range_y=1, range_z=1):
    pass_list = []
    delete_list = []

    def in_image_condition(torsos_projected): # check if projected point is in image
        return torsos_projected.max() <= 1000 and torsos_projected.min() >= 0
    def in_bound_condition(points, bound_center, range_x=1, range_y=1, range_z=1): # check if pelvis point is in bound
        return (abs(points[0]-bound_center[0]) <= range_x) and (abs(points[1]-bound_center[1]) <= range_y) and (abs(points[2]-bound_center[2]) <= range_z)
    # classify points satisfying the conditions
    for i in range(0, len(points)):
        if in_image_condition(torsos_projected[i]) and in_bound_condition(points[i], bound_center, range_x, range_y, range_z):
            pass_list.append(i)
        else:
            delete_list.append(i)
    # get segments from pass_list
    segments_idx = split_continuous_indices(pass_list)
    return segments_idx, pass_list, delete_list

# generate torso index pairs for dataset
def get_pairs(center, window_size, pair_stride, N):
    # window range centered at frame
    window_range = [center-window_size//2, center+window_size//2]
    # calculate max distance from center with pair_stride
    max_dist = 0
    while max_dist*pair_stride <= window_size//2:
        max_dist += 1
    max_dist -= 1
    pairs = []
    # generate pairs
    for pair in range(-max_dist*pair_stride, max_dist*pair_stride+1, pair_stride):
        if center < 0 or center+pair < 0:
            continue
        elif center+pair >= N:
            continue
        pairs.append([center, center+pair])
    return pairs

def get_two_point_parts(input, src, point1, point2):
    if   'points' in input: return src[[point1, point2]]
    elif 'center' in input: return src[[point1, point2]].mean(axis=0)
    elif 'length' in input: return np.linalg.norm(src[2]-src[3])
    elif 'angle'  in input: return np.arctan2(src[point2][1] - src[point1][1], src[point2][0] - src[point1][0])

def get_input(input, src_torso, src_rot, src_2d_old, src_2d_new):
    if   'src_torso' in input:   return src_torso
    elif 'src_point' in input: return src_torso[0]
    elif 'src_rot'   in input:   return src_rot
    elif '2d'        in input:
        if   'old'   in input: return src_2d_old
        elif 'new'   in input: return src_2d_new
        elif 'delta' in input: return src_2d_new - src_2d_old
    elif 'twolines'  in input:
        if   'old'   in input: return src_2d_old[1:]
        elif 'new'   in input: return src_2d_new[1:]
        elif 'delta' in input: return src_2d_new[1:] - src_2d_old[1:]
    elif 'upper'     in input:
        l_shoulder, r_shoulder = 2, 3
        if   'old'   in input: src = src_2d_old
        elif 'new'   in input: src = src_2d_new
        elif 'delta' in input: src = src_2d_new - src_2d_old
        return get_two_point_parts(input, src, l_shoulder, r_shoulder)
    elif 'lower'     in input:
        l_hip, r_hip = 1, 4
        if   'old'   in input: src = src_2d_old
        elif 'new'   in input: src = src_2d_new
        elif 'delta' in input: src = src_2d_new - src_2d_old
        return get_two_point_parts(input, src, l_hip, r_hip)

def make_input(input_list, src_torso, src_rot, src_2d_old, src_2d_new, normalize=True, no_confidence=True):
    input_dict = {}
    if normalize:
        src_2d_old[:, :2] /= 1000.0
        src_2d_new[:, :2] /= 1000.0
    if no_confidence:
        src_2d_old = src_2d_old[:, :2]
        src_2d_new = src_2d_new[:, :2]
    for input in input_list:
        input_dict[input] = get_input(input, src_torso, src_rot, src_2d_old, src_2d_new)
    return input_dict

def get_output(output, src_torso, tar_torso, src_rot, tar_rot):
    if   'tar_torso'        in output: return tar_torso
    elif 'tar_rot'          in output: return tar_rot
    elif 'tar_point'        in output: return tar_torso[0]
    elif 'tar_delta_torso'  in output: return tar_torso - src_torso
    elif 'tar_delta_point'  in output: return tar_torso[0] - src_torso[0]
    elif 'tar_delta_rot'    in output: return tar_rot @ src_rot.T
    elif 'tar_delta_rotvec' in output: return Rotation.from_matrix(tar_rot @ src_rot.T).as_rotvec()
    elif 'tar_delta_theta'  in output: return np.linalg.norm(Rotation.from_matrix(tar_rot @ src_rot.T).as_rotvec())
    elif 'tar_delta_quat'   in output: return Rotation.from_matrix(tar_rot @ src_rot.T).as_quat()

def make_output(output_list, src_torso, tar_torso, src_rot, tar_rot, dic=None):
    if dic is None: output_dict = {}
    else: output_dict = dic

    for output in output_list:
        output_dict[output] = get_output(output, src_torso, tar_torso, src_rot, tar_rot)
    return output_dict

def get_model_input(input_list,  device='cuda', src_torso=None, src_rot=None, src_2d_old=None, src_2d_new=None, input_dict=None):
    if input_dict is None:
        assert src_torso is not None, 'src_torso should be given'
        assert src_2d_old is not None, 'src_2d_old should be given'
        assert src_2d_new is not None, 'src_2d_new should be given'
        input_dict = make_input(input_list, src_torso, src_rot, src_2d_old, src_2d_new)
        #print(input_dict)
    model_input = []
    for item in input_list:
        model_input += make_one_dimension_list(input_dict[item])
        #print(item, len(model_input))
    if device == None:
        return model_input
    else:
        return torch.tensor([model_input]).float().to(device)

def get_label(output_list, src_torso=None, tar_torso=None, src_rot=None, tar_rot=None, label_dict=None):
    if label_dict is None:
        assert src_torso is not None, 'src_torso should be given'
        assert tar_torso is not None, 'tar_torso should be given'
        assert src_rot is not None, 'src_rot should be given'
        assert tar_rot is not None, 'tar_rot should be given'
        label_dict = make_output(output_list, src_torso, tar_torso, src_rot, tar_rot)
    label = []
    for item in output_list:
        #label += make_one_dimension_list(label_dict[item])
        if item in []: # 1D data
            label += [label_dict[item]]
        else: # more than 1D data
            label += make_one_dimension_list(label_dict[item])
    return label

def make_one_dimension_list(input):
    if isinstance(input, list):
        return list(np.array(input, dtype=np.float32).reshape(-1))
    elif isinstance(input, np.ndarray):
        return list(input.reshape(-1))
    else:
        return [input]

class MyCustomDataset(Dataset):
    def __init__(self, segment_folder, data_type='train', input_candidate=[], output_candidate=[], input_list=[], output_list=[], auto_load_data=True):
        self.segment_folder = segment_folder # segment folder path
        self.data_type = data_type # train or test
        self.files = [file for file in os.listdir(segment_folder) if data_type in file] # segment file list
        assert len(self.files) > 0, 'No segment file in {}'.format(segment_folder)

        self.input_list = input_list # list of input items
        self.output_list = output_list # list of output items

        self.input_candidate = input_candidate
        self.output_candidate = output_candidate

        self.total_num = 0 # total number of data
        self.input_len = 0 # total length of input
        self.output_len = 0 # total length of output

        self.input_idxs = [] # range of each input
        for key in input_candidate.keys():
            if key in input_list:
                start = self.input_len
                self.input_len += input_candidate[key]
                end = self.input_len
                self.input_idxs.append((start, end))

        self.output_idxs = [] # range of each output
        for key in output_candidate.keys():
            if key in output_list:
                start = self.output_len
                self.output_len += output_candidate[key]
                end =self.output_len
                self.output_idxs.append((start, end))

        self.input_batches = [] # input for train
        self.label_batches = [] # label for train
        self.src_torso_list = [] # input for test
        self.tar_torso_list = [] # label for test

        if auto_load_data:
            self.set_items()


    def set_items(self):
        for seg_idx in tqdm(range(len(self.files))): # for each segment

            # load segment data
            try:
                file_path = os.path.join(self.segment_folder, self.files[seg_idx]) #'segments2/{}_{}_seg{}.pickle'.format(self.data_type, cam, seg_idx)
                with open(file=file_path, mode='rb') as f:
                    temp_seg=pickle.load(f)
            except Exception as e:
                print(e, file_path)
                continue

            # for each pair in segment
            for temp_pair in temp_seg:
                #print(temp_pair.keys())

                # extract input
                # input = []
                # for item in self.input_list:
                #     input += make_one_dimension_list(temp_pair[item]) # 3d data
                input = get_model_input(self.input_list, device='cuda', input_dict=temp_pair)

                # extract label
                # label = []
                # for item in self.output_list:
                #     #print(item, temp_pair[item])
                #     if item in []: # 1D data
                #         label += [temp_pair[item]]
                #     else: # more than 1D data
                #         label += make_one_dimension_list(temp_pair[item])
                label = get_label(self.output_list, label_dict=temp_pair)

                # store data
                self.input_batches.append(np.array(input, dtype=np.float32))
                self.label_batches.append(np.array(label, dtype=np.float32))
                self.src_torso_list.append(temp_pair['src_torso'].astype(np.float32))
                self.tar_torso_list.append(temp_pair['tar_torso'].astype(np.float32))

    def __getitem__(self, index):
        return (self.input_batches[index], self.label_batches[index], self.src_torso_list[index], self.tar_torso_list[index])

    def __len__(self):
        return len(self.input_batches) # of how many examples(images?) you have


def load_segment_file_from_parameters(step_size, max_dist, max_deg, max_bias, rt_type, xyz_range):
    segment_file = 'traj_segment_dataset_{}_{}_{}_{}_{}_{}.pickle'.format(step_size, max_dist, max_deg, max_bias, rt_type, xyz_range)
    segment_file_path = os.path.join('segments',segment_file)
    with open(file=segment_file_path, mode='rb') as f:
        traj_segment_dataset=pickle.load(f)
    return traj_segment_dataset

## for camera parameter estimation

def get_backbone_line_from_torso(torso):
    l_shoulder = torso[2]
    r_shoulder = torso[3]
    neck = (l_shoulder + r_shoulder)/2
    l_hip = torso[1]
    r_hip = torso[4]
    pelvis = (l_hip + r_hip)/2
    line = [pelvis, neck]
    return line

def get_ap_pose_2d(video_path, ap_result_path, dataset='h36m'):
    from my_utils import get_video_info, readJSON, get_bbox_from_pose2d, get_bbox_area, change_bbox_convention, halpe2h36m
    W, H, video_length, _ = get_video_info(video_path)
    ap_result = readJSON(os.path.join(ap_result_path, 'alphapose-results.json'))
    if dataset == 'h36m': num_keypoints = 17
    elif dataset == 'halpe': num_keypoints = 26
    else: raise ValueError('dataset must be h36m or halpe')
    pose_2d_list = np.zeros([video_length, num_keypoints, 3]) # to keep total frames even if there is no pose
    bbox_area_list = np.zeros([video_length])
    for item in ap_result:
        frame_num = int(item['image_id'].split('.')[0])
        if dataset == 'h36m':
            keypoints = halpe2h36m(np.array(item['keypoints']).reshape(-1, 3))[0] # (17, 3)
        elif dataset == 'halpe':
            keypoints = np.array(item['keypoints']).reshape(-1, 3) # (26, 3)
        # find main person that has maximum bbox area in each frame
        bbox = get_bbox_from_pose2d(keypoints, output_type='xyxy')
        bbox_area = get_bbox_area(bbox, input_type='xyxy')
        cx, cy, w, h = change_bbox_convention(bbox, input_type='xyxy', output_type='xywh')
        condition1 = bbox_area > bbox_area_list[frame_num] # bigger bbox area
        condition2 = (cx > W/4) and (cx < W*3/4)
        if condition1 and condition2:
            bbox_area_list[frame_num] = bbox_area
            pose_2d_list[frame_num] = keypoints

    return pose_2d_list

def parse_args_by_model_name(target):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/DHDST_kookmin_baseline.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/MB_ft_h36m/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-g', '--gpu', default='0', type=str, help='GPU id')
    #opts = parser.parse_args([])
    if target == 'MB_release':
        opts = parser.parse_args([
            '--config', 'configs/pretrain/MB_pretrain.yaml',
            '--evaluate', 'checkpoint/pretrain/MB_release/best_epoch.bin'])
    else:
        opts = parser.parse_args([
            '--config', 'configs/pose3d/{}.yaml'.format(target),
            '--evaluate', 'checkpoint/pose3d/{}/best_epoch.bin'.format(target)])
    return opts

def get_limb_angle(batch_pose):
    from my_utils import get_h36m_keypoint_index, calculate_batch_azimuth_elevation
    # batch_gt: (B, T, 17, 3)
    if type(batch_pose) != torch.Tensor:
        batch_pose = torch.tensor(batch_pose).float()
    if len(batch_pose.shape) == 3:
        batch_pose = batch_pose.unsqueeze(0)

    # get the keypoint index
    r_hip = get_h36m_keypoint_index('r_hip')
    r_knee = get_h36m_keypoint_index('r_knee')
    r_ankle = get_h36m_keypoint_index('r_ankle')


    l_hip = get_h36m_keypoint_index('l_hip')
    l_knee = get_h36m_keypoint_index('l_knee')
    l_ankle = get_h36m_keypoint_index('l_ankle')

    r_shoulder = get_h36m_keypoint_index('r_shoulder')
    r_elbow = get_h36m_keypoint_index('r_elbow')
    r_wrist = get_h36m_keypoint_index('r_wrist')

    l_shoulder = get_h36m_keypoint_index('l_shoulder')
    l_elbow = get_h36m_keypoint_index('l_elbow')
    l_wrist = get_h36m_keypoint_index('l_wrist')

    # 3D d1, d2
    d1_r_hip = batch_pose[:, :, r_hip] # torch.Size([B, F, 3])
    d1_l_hip = batch_pose[:, :, l_hip] # torch.Size([B, F, 3])
    d1_r_shoulder = batch_pose[:, :, r_shoulder] # torch.Size([B, F, 3])
    d1_l_shoulder = batch_pose[:, :, l_shoulder] # torch.Size([B, F, 3])
    d2_r_knee = batch_pose[:, :, r_knee] # torch.Size([B, F, 3])
    d2_l_knee = batch_pose[:, :, l_knee] # torch.Size([B, F, 3])
    d2_r_elbow = batch_pose[:, :, r_elbow] # torch.Size([B, F, 3])
    d2_l_elbow = batch_pose[:, :, l_elbow] # torch.Size([B, F, 3])
    d3_r_ankle = batch_pose[:, :, r_ankle] # torch.Size([B, F, 3])
    d3_l_ankle = batch_pose[:, :, l_ankle] # torch.Size([B, F, 3])
    d3_r_wrist = batch_pose[:, :, r_wrist] # torch.Size([B, F, 3])
    d3_l_wrist = batch_pose[:, :, l_wrist] # torch.Size([B, F, 3])

    # 3D vector
    k_r_upper_leg = d2_r_knee - d1_r_hip # torch.Size([B, F, 3])
    k_l_upper_leg = d2_l_knee - d1_l_hip # torch.Size([B, F, 3])
    k_r_upper_arm = d2_r_elbow - d1_r_shoulder # torch.Size([B, F, 3])
    k_l_upper_arm = d2_l_elbow - d1_l_shoulder # torch.Size([B, F, 3])
    k_r_under_leg = d3_r_ankle - d2_r_knee # torch.Size([B, F, 3])
    k_l_under_leg = d3_l_ankle - d2_l_knee # torch.Size([B, F, 3])
    k_r_under_arm = d3_r_wrist - d2_r_elbow # torch.Size([B, F, 3])
    k_l_under_arm = d3_l_wrist - d2_l_elbow # torch.Size([B, F, 3])

    # Azimuth, Elevation angle
    R_azim_r_upper_leg, R_elev_r_upper_leg = calculate_batch_azimuth_elevation(k_r_upper_leg) # torch.Size([B, F])
    R_azim_l_upper_leg, R_elev_l_upper_leg = calculate_batch_azimuth_elevation(k_l_upper_leg) # torch.Size([B, F])
    R_azim_r_upper_arm, R_elev_r_upper_arm = calculate_batch_azimuth_elevation(k_r_upper_arm) # torch.Size([B, F])
    R_azim_l_upper_arm, R_elev_l_upper_arm = calculate_batch_azimuth_elevation(k_l_upper_arm) # torch.Size([B, F])
    R_azim_r_under_leg, R_elev_r_under_leg = calculate_batch_azimuth_elevation(k_r_under_leg) # torch.Size([B, F])
    R_azim_l_under_leg, R_elev_l_under_leg = calculate_batch_azimuth_elevation(k_l_under_leg) # torch.Size([B, F])
    R_azim_r_under_arm, R_elev_r_under_arm = calculate_batch_azimuth_elevation(k_r_under_arm) # torch.Size([B, F])
    R_azim_l_under_arm, R_elev_l_under_arm = calculate_batch_azimuth_elevation(k_l_under_arm) # torch.Size([B, F])

    R_r_upper_leg = torch.cat([R_azim_r_upper_leg.unsqueeze(-1), R_elev_r_upper_leg.unsqueeze(-1)], dim=-1) # torch.Size([B, F, 2])
    R_l_upper_leg = torch.cat([R_azim_l_upper_leg.unsqueeze(-1), R_elev_l_upper_leg.unsqueeze(-1)], dim=-1) # torch.Size([B, F, 2])
    R_r_upper_arm = torch.cat([R_azim_r_upper_arm.unsqueeze(-1), R_elev_r_upper_arm.unsqueeze(-1)], dim=-1) # torch.Size([B, F, 2])
    R_l_upper_arm = torch.cat([R_azim_l_upper_arm.unsqueeze(-1), R_elev_l_upper_arm.unsqueeze(-1)], dim=-1) # torch.Size([B, F, 2])
    R_r_under_leg = torch.cat([R_azim_r_under_leg.unsqueeze(-1), R_elev_r_under_leg.unsqueeze(-1)], dim=-1) # torch.Size([B, F, 2])
    R_l_under_leg = torch.cat([R_azim_l_under_leg.unsqueeze(-1), R_elev_l_under_leg.unsqueeze(-1)], dim=-1) # torch.Size([B, F, 2])
    R_r_under_arm = torch.cat([R_azim_r_under_arm.unsqueeze(-1), R_elev_r_under_arm.unsqueeze(-1)], dim=-1) # torch.Size([B, F, 2])
    R_l_under_arm = torch.cat([R_azim_l_under_arm.unsqueeze(-1), R_elev_l_under_arm.unsqueeze(-1)], dim=-1) # torch.Size([B, F, 2])

    # angle
    angle = torch.cat([R_r_upper_leg, R_l_upper_leg, R_r_upper_arm, R_l_upper_arm, R_r_under_leg, R_l_under_leg, R_r_under_arm, R_l_under_arm], dim=0) # torch.Size([Bx8, F, 2])

    return angle


def get_input_gt_for_onevec(batch_input, batch_gt):
    # batch_input: (B, T, 17, 3)
    # batch_gt: (B, T, 17, 3)
    from my_utils import get_h36m_keypoint_index, calculate_batch_azimuth_elevation

    # get the keypoint index
    r_hip = get_h36m_keypoint_index('r_hip')
    l_hip = get_h36m_keypoint_index('l_hip')
    r_shoulder = get_h36m_keypoint_index('r_shoulder')
    l_shoulder = get_h36m_keypoint_index('l_shoulder')
    r_elbow = get_h36m_keypoint_index('r_elbow')
    l_elbow = get_h36m_keypoint_index('l_elbow')
    r_knee = get_h36m_keypoint_index('r_knee')
    l_knee = get_h36m_keypoint_index('l_knee')

    # 2D p1, p2
    p1_r_hip = batch_input[:, :, r_hip, :2].unsqueeze(2) # torch.Size([1, 243, 1, 2])
    p1_l_hip = batch_input[:, :, l_hip, :2].unsqueeze(2) # torch.Size([1, 243, 1, 2])
    p1_r_shoulder = batch_input[:, :, r_shoulder, :2].unsqueeze(2) # torch.Size([1, 243, 1, 2])
    p1_l_shoulder = batch_input[:, :, l_shoulder, :2].unsqueeze(2) # torch.Size([1, 243, 1, 2])
    p2_r_elbow = batch_input[:, :, r_elbow, :2].unsqueeze(2) # torch.Size([1, 243, 1, 2])
    p2_l_elbow = batch_input[:, :, l_elbow, :2].unsqueeze(2) # torch.Size([1, 243, 1, 2])
    p2_r_knee = batch_input[:, :, r_knee, :2].unsqueeze(2) # torch.Size([1, 243, 1, 2])
    p2_l_knee = batch_input[:, :, l_knee, :2].unsqueeze(2) # torch.Size([1, 243, 1, 2])

    # 2D vector
    v_r_upper_leg = p2_r_knee - p1_r_hip # torch.Size([1, 243, 1, 2])
    v_l_upper_leg = p2_l_knee - p1_l_hip # torch.Size([1, 243, 1, 2])
    v_r_upper_arm = p2_r_elbow - p1_r_shoulder # torch.Size([1, 243, 1, 2])
    v_l_upper_arm = p2_l_elbow - p1_l_shoulder # torch.Size([1, 243, 1, 2])

    # input
    input_r_upper_leg = torch.cat([p1_r_hip, v_r_upper_leg], dim=2)
    input_l_upper_leg = torch.cat([p1_l_hip, v_l_upper_leg], dim=2)
    input_r_upper_arm = torch.cat([p1_r_shoulder, v_r_upper_arm], dim=2)
    input_l_upper_arm = torch.cat([p1_l_shoulder, v_l_upper_arm], dim=2)
    input = torch.cat([input_r_upper_leg, input_l_upper_leg, input_r_upper_arm, input_l_upper_arm], dim=0)

    # 3D d1, d2
    d1_r_hip = batch_gt[:, :, r_hip].unsqueeze(2) # torch.Size([1, F, 1, 3])
    d1_l_hip = batch_gt[:, :, l_hip].unsqueeze(2) # torch.Size([1, F, 1, 3])
    d1_r_shoulder = batch_gt[:, :, r_shoulder].unsqueeze(2) # torch.Size([1, F, 1, 3])
    d1_l_shoulder = batch_gt[:, :, l_shoulder].unsqueeze(2) # torch.Size([1, F, 1, 3])
    d2_r_knee = batch_gt[:, :, r_knee].unsqueeze(2) # torch.Size([1, F, 1, 3])
    d2_l_knee = batch_gt[:, :, l_knee].unsqueeze(2) # torch.Size([1, F, 1, 3])
    d2_r_elbow = batch_gt[:, :, r_elbow].unsqueeze(2) # torch.Size([1, F, 1, 3])
    d2_l_elbow = batch_gt[:, :, l_elbow].unsqueeze(2) # torch.Size([1, F, 1, 3])

    # 3D vector
    k_r_upper_leg = d2_r_knee - d1_r_hip # torch.Size([1, F, 1, 3])
    k_l_upper_leg = d2_l_knee - d1_l_hip # torch.Size([1, F, 1, 3])
    k_r_upper_arm = d2_r_elbow - d1_r_shoulder # torch.Size([1, F, 1, 3])
    k_l_upper_arm = d2_l_elbow - d1_l_shoulder # torch.Size([1, F, 1, 3])

    # Azimuth, Elevation angle
    R_azim_r_upper_leg, R_elev_r_upper_leg = calculate_batch_azimuth_elevation(k_r_upper_leg[:, :, 0]) # torch.Size([1, F])
    R_azim_l_upper_leg, R_elev_l_upper_leg = calculate_batch_azimuth_elevation(k_l_upper_leg[:, :, 0]) # torch.Size([1, F])
    R_azim_r_upper_arm, R_elev_r_upper_arm = calculate_batch_azimuth_elevation(k_r_upper_arm[:, :, 0]) # torch.Size([1, F])
    R_azim_l_upper_arm, R_elev_l_upper_arm = calculate_batch_azimuth_elevation(k_l_upper_arm[:, :, 0]) # torch.Size([1, F])

    R_r_upper_leg = torch.cat([R_azim_r_upper_leg.unsqueeze(-1), R_elev_r_upper_leg.unsqueeze(-1)], dim=-1) # torch.Size([1, F, 2])
    R_l_upper_leg = torch.cat([R_azim_l_upper_leg.unsqueeze(-1), R_elev_l_upper_leg.unsqueeze(-1)], dim=-1) # torch.Size([1, F, 2])
    R_r_upper_arm = torch.cat([R_azim_r_upper_arm.unsqueeze(-1), R_elev_r_upper_arm.unsqueeze(-1)], dim=-1) # torch.Size([1, F, 2])
    R_l_upper_arm = torch.cat([R_azim_l_upper_arm.unsqueeze(-1), R_elev_l_upper_arm.unsqueeze(-1)], dim=-1) # torch.Size([1, F, 2])

    # Bone length
    L_r_upper_leg = torch.mean(torch.norm(k_r_upper_leg, dim=-1), dim=1, keepdim=True) # torch.Size([B, 1, 1])
    L_l_upper_leg = torch.mean(torch.norm(k_l_upper_leg, dim=-1), dim=1, keepdim=True) # torch.Size([B, 1, 1])
    L_r_upper_arm = torch.mean(torch.norm(k_r_upper_arm, dim=-1), dim=1, keepdim=True) # torch.Size([B, 1, 1])
    L_l_upper_arm = torch.mean(torch.norm(k_l_upper_arm, dim=-1), dim=1, keepdim=True) # torch.Size([B, 1, 1])

    # root point gt
    gt_root_point = torch.cat([d1_r_hip, d1_l_hip, d1_r_shoulder, d1_l_shoulder], dim=0)[:, :, 0]

    # length gt
    gt_length = torch.cat([L_r_upper_leg, L_l_upper_leg, L_r_upper_arm, L_l_upper_arm], dim=0)[:, :, 0]

    # angle gt
    gt_angle = torch.cat([R_r_upper_leg, R_l_upper_leg, R_r_upper_arm, R_l_upper_arm], dim=0)

    return input, gt_root_point, gt_length, gt_angle

def get_h36m_camera_info(h36m_3d_world, h36m_cam_param, subject, action, camera_id):
    # h36m_3d_world, h36m_cam_param -> from load_h36m()
    cam_info = h36m_3d_world[subject][action]['cameras']
    cam_param = get_cam_param(cam_info, subject, h36m_cam_param)
    calibration_matrix = np.array(cam_param[camera_id]['int']['calibration_matrix'])
    R = np.array(cam_param[camera_id]['ext']['R'])
    t = np.array(cam_param[camera_id]['ext']['t'])/1000
    H = cam_param[camera_id]['H']
    W = cam_param[camera_id]['W']
    camera_param = {
        'intrinsic': calibration_matrix,
        'extrinsic': np.concatenate([R, t.reshape(3, 1)], axis=1),
    }
    fx = camera_param['intrinsic'][0, 0]
    fy = camera_param['intrinsic'][1, 1]
    cx = camera_param['intrinsic'][0, 2]
    cy = camera_param['intrinsic'][1, 2]
    return calibration_matrix, camera_param, H, W, fx, fy, cx, cy

def h36m_data_processing(pose3d_list, camera_param, fx, fy, cx, cy, length=243):
    from my_utils import World2CameraCoordinate, get_rootrel_pose, infer_box, camera_to_image_frame, optimize_scaling_factor
    pose_2d_list = []
    cam_3d_list = []
    img_3d_list = []
    img_3d_hat_list = []
    img_25d_list = []
    scale_list = []
    for frame_num in tqdm(range(len(pose3d_list[:length]))):
        world_3d = np.array(pose3d_list[frame_num])
        # world to camera
        pos = copy.deepcopy(world_3d)
        cam_3d = World2CameraCoordinate(pos, camera_param['extrinsic']) * 1000 # World coordinate -> Camera coordinate
        cam_3d_hat = get_rootrel_pose(cam_3d)

        # camera to image
        box = infer_box(cam_3d, {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}, 0)
        img_2d, img_3d = camera_to_image_frame(cam_3d, box, {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}, 0)
        img_3d_hat = get_rootrel_pose(img_3d) # (17, 3) # root-relative pose
        # 2.5d factor
        pred_lambda, losses = optimize_scaling_factor(img_3d_hat, cam_3d_hat, stop_tolerance=0.0001) # x,y,z 사용
        # joint 2.5d image
        img_25d = img_3d * pred_lambda

        pose_2d_list.append(img_2d)
        cam_3d_list.append(cam_3d)
        img_3d_list.append(img_3d)
        img_3d_hat_list.append(img_3d_hat)
        img_25d_list.append(img_25d)
        scale_list.append(pred_lambda)

    pose_2d_list = np.array(pose_2d_list)
    pose_2d_list = np.append(pose_2d_list, np.ones((pose_2d_list.shape[0], pose_2d_list.shape[1], 1)), axis=2)
    cam_3d_list = np.array(cam_3d_list)
    img_3d_list = np.array(img_3d_list)
    img_3d_hat_list = np.array(img_3d_hat_list)
    img_25d_list = np.array(img_25d_list)
    scale_list = np.array(scale_list)

    return pose_2d_list, cam_3d_list, img_3d_list, img_3d_hat_list, img_25d_list, scale_list

def select_dataset_from_checkpoint(checkpoint):
    if 'h36m' in checkpoint:
        if 'tr_s1' in checkpoint:
            train_dataset = 'H36M S1'
            if 'ts_s5678' in checkpoint: test_dataset = 'H36M S5 6 7 8'
            else: test_dataset = 'ALL EXCEPT S1'
        elif 'tr_s19_ts_s5678' in checkpoint:
            train_dataset = 'H36M S1 9'
            test_dataset = 'H36M S5 6 7 8'
        elif 's15678_tr_54138969' in checkpoint:
            train_dataset = 'H36M S1 5 6 7 8 CAM 54138969'
            test_dataset = 'H36M S1 5 6 7 8 CAM EXCEPT 54138969'
        else:
            train_dataset = 'H36M S1 5 6 7 8'
            test_dataset = 'H36M S9 11'
    elif 'fit3d' in checkpoint:
        if 'tr_s03' in checkpoint:
            train_dataset = 'FIT3D S3'
            test_dataset = 'FIT3D ALL TRAIN EXCEPT S3'
        elif 'ts_s4710' in checkpoint:
            train_dataset = 'FIT3D S3 5 8 9 11'
            test_dataset = 'FIT3D S4 7 10'
    elif '3dhp' in checkpoint:
        train_dataset = None
        test_dataset = None
    return train_dataset, test_dataset

def select_testset_from_subset(subset):
    if 'H36M' in subset:
        if 'TS_S5678' in subset: test_dataset = 'H36M S5 6 7 8'
        elif 'TR_S1' in subset and 'TS_S5678' not in subset: test_dataset = 'H36M ALL EXCEPT S1'
        elif 'S15678_TR_54138969_TS_OTHERS' in subset: test_dataset = 'H36M S1 5 6 7 8 CAM EXCEPT 54138969'
        else: test_dataset = 'H36M S9 11'
    elif 'FIT3D' in subset:
        if 'ALL_TEST' in subset: test_dataset = 'FIT3D ALL (S3 4 5 7 8 9 10 11)'
        elif 'TS_S4710' in subset: test_dataset = 'FIT3D S4 7 10'
    elif '3DHP' in subset:
        if 'TEST_TS1_6' in subset: test_dataset = '3DHP TESTSET (TS1~6) (original)'
        elif 'TEST_ALL_TRAIN' in subset: test_dataset = '3DHP TRAINSET (S1~8 CAM 0 1 2 4 5 6 7 8)'
    else:
        test_dataset = None
    return test_dataset

def gernerate_dataset_yaml(subset):
    user = getpass.getuser()
    yaml_root=f'/home/{user}/codes/MotionBERT/data/motion3d/yaml_files'
    splited = subset.split('-')
    dataset_name = splited[0].lower()

    input_source = splited[1]
    print(subset)
    train_subject = []
    train_cam = []
    test_cam = []
    cam_list = []
    univ = False
    if dataset_name == 'h36m':
        if 'TR_S1_TS_S5678' in subset:
            train_subject = ['S1']
            test_subject = ['S5', 'S6', 'S7', 'S8']
        elif 'S15678_TR_54138969_TS_OTHERS' in subset:
            train_subject = ['S1', 'S5', 'S6', 'S7', 'S8']
            test_subject = ['S1', 'S5', 'S6', 'S7', 'S8']
            train_cam = ['54138969']
            test_cam = ['60457274', '55011271', '58860488']
        elif 'TR_S19_TS_S5678' in subset:
            train_subject = ['S1', 'S9']
            test_subject = ['S5', 'S6', 'S7', 'S8']
        elif 'TR_S1' in subset:
            train_subject = ['S1']
            test_subject = ['S5', 'S6', 'S7', 'S8', 'S9', 'S11']
        elif 'TEST_ALL' in subset and 'EXCEPT_S1' not in subset:
            test_subject = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
        else:
            train_subject = ['S1', 'S5', 'S6', 'S7', 'S8']
            test_subject = ['S9', 'S11']
    elif dataset_name == 'fit3d':
        if 'TR_S03' in subset:
            train_subject = ['s03']
            test_subject = ['s04', 's05', 's07', 'S8', 'S9', 'S10', 'S11']
        elif 'TS_S4710' in subset:
            train_subject = ['s03', 's05', 'S8', 'S9', 'S11']
            test_subject = ['s04', 's07', 's10']
        elif 'ALL_TEST' in subset:
            test_subject = ['s03', 's04', 's05', 's07', 's08', 's09', 's10', 's11']
        else: raise ValueError(f'Invalid item: {subset}')
    elif dataset_name == '3dhp':
        if 'ALL_TRAIN' in subset:
            test_subject = [f'S{i}' for i in range(1, 9)]
            cam_list = ['cam0', 'cam1', 'cam2', 'cam4', 'cam5', 'cam6', 'cam7', 'cam8']
        elif 'TS1_6' in subset:
            test_subject = [f'TS{i}' for i in range(1, 7)]
        elif 'TS1_4' in subset:
            test_subject = [f'TS{i}' for i in range(1, 5)]
        else: raise ValueError(f'Invalid item: {subset}')
        if 'UNIV' in subset: univ = True
    elif dataset_name == 'kookmin':
        if 'FOLD1' in subset:   test_subject = ['S12', 'S15', 'S18']
        elif 'FOLD2' in subset: test_subject = ['S16', 'S18', 'S19']
        elif 'FOLD3' in subset: test_subject = ['S13', 'S14', 'S19']
        elif 'FOLD4' in subset: test_subject = ['S11', 'S12', 'S20']
        elif 'FOLD5' in subset: test_subject = ['S11', 'S17', 'S18']
        train_subject = list(set([f'S{i}' for i in range(11, 21)]) - set(test_subject))

    #print(train_subject, test_subject)

    # 3d data type
    data_type_list = ['cam_3d']
    if 'CAM_NO_FACTOR' in splited:
        gt_mode = 'cam_3d'
        if 'CANONICAL' in subset:
            data_type_list += ['cam_3d_from_canonical_3d']
            if ('STEP_ROT' in subset) or ('SINU_' in subset) or 'RAND_' in subset: gt_mode = 'cam_3d_from_canonical_3d' # 어차피 rootrel option 아래에서는 cam_3d와 동일
            if 'REVOLUTE' in subset: gt_mode = 'cam_3d_from_canonical_3d' # revolute는 cam_3d_from_canonical_3d와 cam_3d의 root-relative가 다름
    elif 'WORLD_NO_FACTOR' in splited:
        data_type_list += ['world_3d']
        gt_mode = 'world_3d'
    else:
        if 'CANONICAL' in subset:
            data_type_list += ['joint3d_image_from_canonical_3d', '2.5d_factor_from_canonical_3d', 'joints_2.5d_image_from_canonical_3d']
            gt_mode = 'joint3d_image_from_canonical_3d'
        else:
            data_type_list += ['joint3d_image', '2.5d_factor', 'joints_2.5d_image']
            gt_mode = 'joint3d_image'

    # 2d data type
    if 'CANONICAL' in subset:
        data_type_list += ['joint_2d_from_canonical_3d']
        input_mode = 'joint_2d_from_canonical_3d'
    else:
        data_type_list += ['joint_2d']
        input_mode = 'joint_2d'

    # canonical type
    if 'SAME_Z' in subset:         canonical_type = 'same_z'
    elif 'SAME_DIST' in subset:    canonical_type = 'same_dist'
    elif 'FIXED_DIST_5' in subset: canonical_type = 'fixed_dist_5'
    elif 'REVOLUTE' in subset:     canonical_type = 'revolute'
    elif 'PCL' in subset:          canonical_type = 'pcl'
    else: canonical_type = None

    if 'ADAPTIVE_FOCAL' in subset:
        adaptive_focal = True
    else:
        adaptive_focal = False

    # data augmentation
    step_rot = 0
    sinu_yaw_mag = 0
    sinu_pitch_mag = 0
    sinu_roll_mag = 0
    sinu_yaw_period = 273*2
    sinu_pitch_period = 273*2
    sinu_roll_period = 273*2
    rand_yaw_mag = 0
    rand_pitch_mag = 0
    rand_roll_mag = 0
    rand_yaw_period = 0
    rand_pitch_period = 0
    rand_roll_period = 0

    for item in splited:
        if 'STEP_ROT' in item:
            step_rot = float(item.split('_')[-1])
        elif 'SINU_YAW' in item:
            sinu_yaw_mag = float(item.split('_')[2].split('M')[1])
            sinu_yaw_period = float(item.split('_')[3].split('P')[1])
        elif 'RAND_YAW' in item:
            rand_yaw_mag = float(item.split('_')[2].split('M')[1])
            rand_yaw_period = float(item.split('_')[3].split('P')[1])
        if 'SINU_PITCH' in item:
            sinu_pitch_mag = float(item.split('_')[2].split('M')[1])
            sinu_pitch_period = float(item.split('_')[3].split('P')[1])
        elif 'RAND_PITCH' in item:
            rand_pitch_mag = float(item.split('_')[2].split('M')[1])
            rand_pitch_period = float(item.split('_')[3].split('P')[1])
        if 'SINU_ROLL' in item:
            sinu_roll_mag = float(item.split('_')[2].split('M')[1])
            sinu_roll_period = float(item.split('_')[3].split('P')[1])
        elif 'RAND_ROLL' in item:
            rand_roll_mag = float(item.split('_')[2].split('M')[1])
            rand_roll_period = float(item.split('_')[3].split('P')[1])

    #print(dataset_name, input_source, data_type_list, canonical_type, input_mode, gt_mode)

    data = {
        'dataset_name': dataset_name,
        'data_type_list': data_type_list,
        'canonical_type': canonical_type,
        'input_source': input_source,
        'input_mode': input_mode,
        'gt_mode': gt_mode,
        'train_subject': train_subject,
        'test_subject': test_subject,
        'train_cam': train_cam,
        'test_cam': test_cam,
        'cam_list': cam_list,
        'adaptive_focal': adaptive_focal,
        'step_rot': step_rot,
        'sinu_yaw_mag': sinu_yaw_mag,
        'sinu_yaw_period': sinu_yaw_period,
        'sinu_pitch_mag': sinu_pitch_mag,
        'sinu_pitch_period': sinu_pitch_period,
        'sinu_roll_mag': sinu_roll_mag,
        'sinu_roll_period': sinu_roll_period,
        'rand_yaw_mag': rand_yaw_mag,
        'rand_yaw_period': rand_yaw_period,
        'rand_pitch_mag': rand_pitch_mag,
        'rand_pitch_period': rand_pitch_period,
        'rand_roll_mag': rand_roll_mag,
        'rand_roll_period': rand_roll_period,
        'univ': univ
    }

    with open(os.path.join(yaml_root, f'{subset}.yaml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
    print(os.path.join(yaml_root, f'{subset}.yaml'))