from hpe_library.lib_import import *
# from my_utils import readpkl, savepkl, mpi_inf_3dhp2h36m
# from my_utils import normalize_input
# from .inference import normalize_input

def convert_intrinsic_from_mm_to_pixel(sensor_size, focal_length, pixel_aspect, center_offset, sensor_pixels_x, sensor_pixels_y):
    # Calculate pixel sizes
    pixel_size_x = sensor_size[0] / sensor_pixels_x
    pixel_size_y = sensor_size[1] * pixel_aspect / sensor_pixels_y

    # Calculate focal length in pixels
    fx = focal_length / pixel_size_x
    fy = focal_length / pixel_size_y

    # Calculate center offset in pixels
    center_offset_pixels_x = center_offset[0] / pixel_size_x
    center_offset_pixels_y = center_offset[1] / pixel_size_y

    cx, cy = sensor_pixels_x/2 + center_offset_pixels_x, sensor_pixels_y/2 + center_offset_pixels_y
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic

def get_3dhp_cam_info(data_type):
    assert data_type in ['train', 'test'], f"Invalid data_type: {data_type}"
    if data_type == 'train':
        cam_info_3dhp_train = {}
        return cam_info_3dhp_train
    else:
        cam_info_3dhp_test = {
            'test_cam_1_4': {
                'origin': np.array([3427.28, 1387.86, 309.42]),
                'up': np.array([-0.208215, 0.976233, 0.06014]),
                'right': np.array([0.000575281, 0.0616098, -0.9981]),
                'sensor_size': np.array([10, 10]), # in mm
                'focal_length': 7.32506, # in mm
                'pixel_aspect': 1.00044, # y / x
                'center_offset': np.array([-0.0322884, 0.0929296]) # in mm (positive values move right and down)
            },
            'test_cam_5_6': {
                'origin': np.array([-2104.3074, 1038.6707, -4596.6367]),
                'up': np.array([0.025272345, 0.995038509, 0.096227370]),
                'right': np.array([-0.939647257, -0.009210289, 0.342020929]),
                'sensor_size': np.array([10, 5.625]), # in mm
                'focal_length': 8.770747185, # in mm
                'pixel_aspect': 0.993236423, # y / x
                'center_offset': np.array([-0.104908645, 0.104899704]) # in mm (positive values move right and down)
            }
        }

        for key in cam_info_3dhp_test.keys():
            up = cam_info_3dhp_test[key]['up']
            right = cam_info_3dhp_test[key]['right']
            forward = np.cross(right, up)
            R = np.column_stack((right, up, forward))
            print(R)
            t = cam_info_3dhp_test[key]['origin']/1000
            #t = -R@C
            C = -R.T@t
            cam_info_3dhp_test[key]['R'] = R
            cam_info_3dhp_test[key]['t'] = t
            cam_info_3dhp_test[key]['C'] = C
        return cam_info_3dhp_test
    
def get_img_frame_3dhp(data_type, subject, frame_num, seq=None, cam_id=None):
    from my_utils import get_video_frame
    user = getpass.getuser()
    if data_type == 'train':
        cam_id = cam_id.split('cam')[-1]
        video_path = f'/home/{user}/Datasets/HAAI/3DHP/original/train/{subject}/{seq}/imageSequence/video_{cam_id}.avi'
        return get_video_frame(video_path, frame_id=frame_num)
    elif data_type == 'test':
        img_path = f'/home/{user}/Datasets/HAAI/3DHP/original/test/{subject}/imageSequence/img_{frame_num:06d}.jpg'
        return cv2.imread(img_path)
    else:
        raise ValueError(f'Invalid data_type: {data_type}')
    
def load_3dhp_original(data_type='test', overwrite=False, no_save=False):
    from my_utils import readpkl, savepkl, mpi_inf_3dhp2h36m, normalize_input
    #print(f"==> Loading 3DHP {data_type} data...")
    user = getpass.getuser()
    cam_params = readpkl(f'/home/{user}/codes/MotionBERT/custom_codes/dataset_generation/3dhp/3dhp_{data_type}_cam_params.pkl')
    data_dict_path = f'/home/{user}/codes/MotionBERT/custom_codes/dataset_generation/3dhp/3dhp_{data_type}_data_dict.pkl'
    if os.path.exists(data_dict_path) and not overwrite and not no_save:
        data_dict = readpkl(data_dict_path)
    else:
        folder = f'/home/{user}/Datasets/HAAI/3DHP/original/{data_type}'
        data_dict = {}
        if data_type == 'test':
            for subject in natsorted(os.listdir(folder)):
                if 'TS' not in subject: continue
                cam_param = cam_params[subject]
                W, H = cam_param['W'], cam_param['H']
                data_dict[subject] = {}
                data = scipy.io.loadmat(f'/home/{user}/Datasets/HAAI/3DHP/original/test/{subject}/annot_data_modi.mat')
                annot2 = mpi_inf_3dhp2h36m(np.transpose(data['annot2'][:, :, 0, :], (2, 1, 0))).copy()
                annot2_norm = normalize_input(annot2.copy(), W, H)
                annot3 = mpi_inf_3dhp2h36m(np.transpose(data['annot3'][:, :, 0, :], (2, 1, 0))).copy()
                annot3_hat = annot3.copy() - annot3[:, 0, None]
                univ_annot3 = mpi_inf_3dhp2h36m(np.transpose(data['univ_annot3'][:, :, 0, :], (2, 1, 0))).copy()
                valid_frame = np.where(data['valid_frame'] == 1)[1]
                num_valid_frame = len(valid_frame)
                # get visible frame
                w_over_range = (annot2[:, :, 0] > 2048) | (annot2[:, :, 0] < 0)
                h_over_range = (annot2[:, :, 1] > 2048) | (annot2[:, :, 1] < 0)
                over_range = np.logical_or(w_over_range, h_over_range)
                visible = np.logical_not(np.any(over_range, axis=1))
                visible_frame = np.where(visible == True)[0]
                num_visible_frames = len(visible_frame)
                # save data
                data_dict[subject]['annot2'] = annot2
                data_dict[subject]['annot2_norm'] = annot2_norm
                data_dict[subject]['annot3'] = annot3
                data_dict[subject]['annot3_hat'] = annot3_hat
                data_dict[subject]['univ_annot3'] = univ_annot3
                data_dict[subject]['visible_frame'] = visible_frame
                data_dict[subject]['num_visible_frames'] = num_visible_frames
                data_dict[subject]['valid_frame'] = valid_frame
                data_dict[subject]['num_valid_frame'] = num_valid_frame
                data_dict[subject]['num_frames'] = cam_param['num_frames']
        elif data_type == 'train':
            for subject in tqdm(natsorted(os.listdir(folder))):
                for seq in natsorted(os.listdir(os.path.join(folder, subject))):
                    data = scipy.io.loadmat(os.path.join(folder, subject, seq, 'annot.mat'))
                    for cam_num in range(14):
                        cam_id = f'cam{cam_num}'
                        source = '_'.join([subject, seq, cam_id])
                        cam_param = cam_params[cam_id]
                        W, H = cam_param['W'], cam_param['H']
                        data_dict[source] = {}
                        annot2 = mpi_inf_3dhp2h36m(np.array(data['annot2'][cam_num][0].reshape(-1, 28, 2)).copy())
                        annot2_norm = normalize_input(annot2.copy(), W, H)
                        annot3 = mpi_inf_3dhp2h36m(np.array(data['annot3'][cam_num][0].reshape(-1, 28, 3)).copy())
                        annot3_hat = annot3.copy() - annot3[:, 0, None]
                        univ_annot3 = mpi_inf_3dhp2h36m(np.array(data['univ_annot3'][cam_num][0].copy().reshape(-1, 28, 3)).copy())
                        # get valid frame
                        w_over_range = (annot2[:, :, 0] > 2048) | (annot2[:, :, 0] < 0)
                        h_over_range = (annot2[:, :, 1] > 2048) | (annot2[:, :, 1] < 0)
                        over_range = np.logical_or(w_over_range, h_over_range)
                        visible = np.logical_not(np.any(over_range, axis=1))
                        visible_frame = np.where(visible == True)[0]
                        num_visible_frames = len(visible_frame)
                        # save data
                        data_dict[source]['annot2'] = annot2
                        data_dict[source]['annot2_norm'] = annot2_norm
                        data_dict[source]['annot3'] = annot3
                        data_dict[source]['annot3_hat'] = annot3_hat
                        data_dict[source]['univ_annot3'] = univ_annot3
                        data_dict[source]['visible_frame'] = visible_frame
                        data_dict[source]['num_visible_frames'] = num_visible_frames
                        data_dict[source]['num_frames'] = len(annot2)
        if not no_save: savepkl(data_dict, data_dict_path)
        
    return data_dict, cam_params

def test_3dhp_data_generator(pose_type, canonical_type='', fixed_position=[]):
    from my_utils import remove_nose_from_h36m, canonicalization_cam_3d, projection, normalize_input, denormalize_input
    # load original 3dhp data
    mpi_inf_3dhp_test = np.load('data_extra/dataset_extras/mpi_inf_3dhp_test.npz')
    original_test, cam_param_3dhp_test = load_3dhp_original('test', overwrite=False)
    test_3dhp_from_test = np.load('data_extra/test_set/test_3dhp_from_test.npz')
    
    source_list = []
    pose_dict = {}
    for key in pose_type:
        pose_dict[key] = []
    for frame_num in range(len(mpi_inf_3dhp_test['imgname'])):
        source = mpi_inf_3dhp_test['imgname'][frame_num]
        splited = source.split('/')
        subject = splited[1]
        actual_frame = int(splited[-1].split('.')[0].split('_')[-1]) - 1
        source_list.append([subject, actual_frame])
        cam_param = cam_param_3dhp_test[subject]
        W, H, intrinsic = cam_param['W'], cam_param['H'], cam_param['intrinsic']
        #fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        
        # original 3dhp
        annot3 = remove_nose_from_h36m(original_test[subject]['annot3'][actual_frame].copy()/1000)
        annot3_hat = annot3 - annot3[0]
        annot2 = remove_nose_from_h36m(original_test[subject]['annot2'][actual_frame].copy())
        annot2_norm = normalize_input(annot2, W, H)
        annot2_proj = projection(annot3, intrinsic)[..., :2]
        annot2_proj_norm = normalize_input(annot2_proj, W, H)
        univ_annot3 = remove_nose_from_h36m(original_test[subject]['univ_annot3'][actual_frame].copy()/1000)
        univ_annot3_hat = univ_annot3 - univ_annot3[0]
        univ_annot2 = projection(univ_annot3, intrinsic)[..., :2]
        univ_annot2_norm = normalize_input(univ_annot2, W, H)
        # poseaug 3dhp 
        test_pose3d = test_3dhp_from_test['pose3d'][frame_num]
        test_pose2d = test_3dhp_from_test['pose2d'][frame_num]
        test_pose2d_denorm = denormalize_input(test_pose2d, W, H)
        
        if canonical_type != '':
            # canonicalization for annot3
            annot3_canonical = canonicalization_cam_3d(annot3, canonical_type=canonical_type)
            annot2_canonical = projection(annot3_canonical, intrinsic)[..., :2]
            annot2_canonical_norm = normalize_input(annot2_canonical, W, H)
            # canonicalization for univ_annot3
            univ_annot3_canonical = canonicalization_cam_3d(univ_annot3, canonical_type=canonical_type)
            univ_annot2_canonical = projection(univ_annot3_canonical, intrinsic)[..., :2]
            univ_annot2_canonical_norm = normalize_input(univ_annot2_canonical, W, H)
        if len(fixed_position) > 0:
            # fixed position for annot3
            annot3_fixed_pos = annot3 - annot3[0] + fixed_position
            annot2_fixed_pos = projection(annot3_fixed_pos, intrinsic)[..., :2]
            annot2_fixed_pos_norm = normalize_input(annot2_fixed_pos, W, H)
            # fixed position for univ_annot3
            univ_annot3_fixed_pos = univ_annot3 - univ_annot3[0] + fixed_position
            univ_annot2_fixed_pos = projection(univ_annot3_fixed_pos, intrinsic)[..., :2]
            univ_annot2_fixed_pos_norm = normalize_input(univ_annot2_fixed_pos, W, H)
        
        if 'annot3'                     in pose_type: pose_dict['annot3'].append(annot3)
        if 'annot3_hat'                 in pose_type: pose_dict['annot3_hat'].append(annot3_hat)
        if 'univ_annot3'                in pose_type: pose_dict['univ_annot3'].append(univ_annot3)
        if 'univ_annot3_hat'            in pose_type: pose_dict['univ_annot3_hat'].append(univ_annot3_hat)
        if 'annot3_fixed_pos'           in pose_type: pose_dict['annot3_fixed_pos'].append(annot3_fixed_pos)
        if 'univ_annot3_fixed_pos'      in pose_type: pose_dict['univ_annot3_fixed_pos'].append(univ_annot3_fixed_pos)
        if 'annot3_canonical'           in pose_type: pose_dict['annot3_canonical'].append(annot3_canonical)
        if 'univ_annot3_canonical'      in pose_type: pose_dict['univ_annot3_canonical'].append(univ_annot3_canonical)
        if 'test_pose3d'                in pose_type: pose_dict['test_pose3d'].append(test_pose3d)
        if 'annot2'                     in pose_type: pose_dict['annot2'].append(annot2)
        if 'annot2_norm'                in pose_type: pose_dict['annot2_norm'].append(annot2_norm)
        if 'annot2_proj'                in pose_type: pose_dict['annot2_proj'].append(annot2_proj)
        if 'annot2_proj_norm'           in pose_type: pose_dict['annot2_proj_norm'].append(annot2_proj_norm)
        if 'annot2_fixed_pos'           in pose_type: pose_dict['annot2_fixed_pos'].append(annot2_fixed_pos)
        if 'annot2_fixed_pos_norm'      in pose_type: pose_dict['annot2_fixed_pos_norm'].append(annot2_fixed_pos_norm)
        if 'annot2_canonical'           in pose_type: pose_dict['annot2_canonical'].append(annot2_canonical)
        if 'annot2_canonical_norm'      in pose_type: pose_dict['annot2_canonical_norm'].append(annot2_canonical_norm)
        if 'univ_annot2'                in pose_type: pose_dict['univ_annot2'].append(univ_annot2)
        if 'univ_annot2_norm'           in pose_type: pose_dict['univ_annot2_norm'].append(univ_annot2_norm)
        if 'univ_annot2_fixed_pos'      in pose_type: pose_dict['univ_annot2_fixed_pos'].append(univ_annot2_fixed_pos)
        if 'univ_annot2_fixed_pos_norm' in pose_type: pose_dict['univ_annot2_fixed_pos_norm'].append(univ_annot2_fixed_pos_norm)
        if 'univ_annot2_canonical'      in pose_type: pose_dict['univ_annot2_canonical'].append(univ_annot2_canonical)
        if 'univ_annot2_canonical_norm' in pose_type: pose_dict['univ_annot2_canonical_norm'].append(univ_annot2_canonical_norm)
        if 'test_pose2d'                in pose_type: pose_dict['test_pose2d'].append(test_pose2d)
        if 'test_pose2d_denorm'         in pose_type: pose_dict['test_pose2d_denorm'].append(test_pose2d_denorm)
        
    for key in pose_dict.keys():
        pose_dict[key] = np.array(pose_dict[key])
    return pose_dict, source_list, cam_param_3dhp_test