from hpe_library.lib_import import *

def get_3dpw_seq_list():
    user = getpass.getuser()
    root_3dpw_original = f'/home/{user}/Datasets/HAAI/3DPW/original'
    train_root = root_3dpw_original + '/sequenceFiles/sequenceFiles/train'
    test_root = root_3dpw_original + '/sequenceFiles/sequenceFiles/test'
    valid_root = root_3dpw_original + '/sequenceFiles/sequenceFiles/validation'
    train_seq_list = [item.split('.pkl')[0] for item in os.listdir(train_root)]
    test_seq_list = [item.split('.pkl')[0] for item in os.listdir(test_root)]
    valid_seq_list = [item.split('.pkl')[0] for item in os.listdir(valid_root)]
    return train_seq_list, test_seq_list, valid_seq_list

def get_3dpw_source_list():
    from hpe_library.my_utils import savepkl, readpkl
    user = getpass.getuser()
    save_path_source_list = f'/home/{user}/codes/MotionBERT/custom_codes/dataset_generation/3dpw/3dpw-source_list.pkl'
    if os.path.exists(save_path_source_list):
        print(f"load 3dpw source_list from {save_path_source_list}")
        source_list = readpkl(save_path_source_list)
    else:
        root_3dpw_original = f'/home/{user}/Datasets/HAAI/3DPW/original'
        source_list = []
        for pkl_path in glob(root_3dpw_original+'/sequenceFiles/sequenceFiles/*/*.pkl'):
            data_type = pkl_path.split('/')[-2]
            with open(pkl_path, 'rb') as f:
                pkl_data = pickle.load(f, encoding='latin1')
                num_people = len(pkl_data['poses'])
                seq_name = str(pkl_data['sequence'])
                for subject_id in range(num_people):
                    source_list.append(f"{data_type}_{seq_name}_{subject_id}")
        savepkl(source_list, save_path_source_list)
    return source_list

def get_3dpw_img_paths(only_valid=True):
    from hpe_library.my_utils import savepkl, readpkl
    user = getpass.getuser()
    save_path_img_paths = f'/home/{user}/codes/MotionBERT/custom_codes/dataset_generation/3dpw/3dpw-img_paths.pkl'
    if os.path.exists(save_path_img_paths):
        print(f"load 3dpw img_paths from {save_path_img_paths}")
        img_paths = readpkl(save_path_img_paths)
    else:
        source_list = get_3dpw_source_list()
        img_paths = {}
        for source in tqdm(source_list, desc='generate 3dpw img_paths'):
            data_type = source.split('_')[0]
            seq_name = source.removeprefix(f"{data_type}_")[:-2]
            sub_id = int(source.split('_')[-1])
            if data_type not in img_paths: img_paths[data_type] = {}
            if seq_name not in img_paths: img_paths[data_type][seq_name] = {}
            data = load_pkl_3dpw(data_type, seq_name)
            num_frames = len(data['poses'][0])
            img_names = np.array(['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
            if only_valid:
                valid = np.array(data['campose_valid']).astype(bool)
                img_paths[data_type][seq_name][sub_id] = img_names[valid[sub_id]]
            else:
                img_paths[data_type][seq_name][sub_id] = img_names
        savepkl(img_paths, save_path_img_paths)
    return img_paths

def get_3dpw_cam_params(overwrite=False):
    from hpe_library.my_utils import savepkl, readpkl
    user = getpass.getuser()
    save_path_cam_params = f'/home/{user}/codes/MotionBERT/custom_codes/dataset_generation/3dpw/3dpw-cam_params.pkl'
    if os.path.exists(save_path_cam_params) and not overwrite:
        print(f"load 3dpw cam_params from {save_path_cam_params}")
        cam_params = readpkl(save_path_cam_params)
    else:
        root_3dpw_original = f'/home/{user}/Datasets/HAAI/3DPW/original'
        img_root = root_3dpw_original + '/imageFiles'
        source_list = get_3dpw_source_list()
        cam_params = {}
        for source in tqdm(source_list, desc='generate 3dpw cam_params'):
            data_type = source.split('_')[0]
            seq_name = source.removeprefix(f"{data_type}_")[:-2]
            sub_id = int(source.split('_')[-1])
            if data_type not in cam_params: cam_params[data_type] = {}
            if seq_name not in cam_params: cam_params[data_type][seq_name] = {}
            # pkl_path = f"{root_3dpw_original}/sequenceFiles/sequenceFiles/{data_type}/{seq_name}.pkl"
            # assert os.path.exists(pkl_path), f"{pkl_path} not found"
            # with open(pkl_path, 'rb') as f:
            #     pkl_data = pickle.load(f, encoding='latin1')
            data = load_pkl_3dpw(data_type, seq_name)
            valid = np.array(data['campose_valid']).astype(bool)[sub_id]
            intrinsic = data['cam_intrinsics']
            extrinsic = data['cam_poses'][valid]
            R = extrinsic[:, :3, :3] # (N, 3, 3)
            t = extrinsic[:, :3, 3] # (N, 3)
            R_t = np.transpose(R, (0, 2, 1))
            C = -np.matmul(R_t, t[...,None]).squeeze() # (N, 3)
            img = cv2.imread(f'{img_root}/{seq_name}/image_{0:05d}.jpg')
            W, H = img.shape[1], img.shape[0]
            num_frames = extrinsic.shape[0]
            cam_params[data_type][seq_name] = {
                'R': R,
                't': t,
                'C': C,
                'W': W,
                'H': H,
                'intrinsic': intrinsic,
                'extrinsic': extrinsic,
                'num_frames': num_frames,
                'fps': 30.0
            }
        savepkl(cam_params, save_path_cam_params)
    return cam_params

def load_pkl_3dpw(data_type, seq_name):
    user = getpass.getuser()
    pkl_path = f"/home/{user}/Datasets/HAAI/3DPW/original/sequenceFiles/sequenceFiles/{data_type}/{seq_name}.pkl"
    assert os.path.exists(pkl_path), f"{pkl_path} not found"
    with open(pkl_path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def get_3dpw_smpl_regressed_joint(data_type, seq_name, sub_id, J_regressor, smpl_male, smpl_female, device=torch.device('cuda')):
    data = load_pkl_3dpw(data_type, seq_name)
    smpl_pose = data['poses']
    smpl_betas = data['betas']
    global_poses = data['cam_poses'] # extrinsic
    genders = data['genders']
    valid = np.array(data['campose_valid']).astype(bool)
    num_frames = len(smpl_pose[0])
    seq_name = str(data['sequence'])
    poses_, shapes_, genders_ = [], [], []
    valid_pose = smpl_pose[sub_id][valid[sub_id]]
    valid_betas = np.tile(smpl_betas[sub_id][:10].reshape(1,-1), (num_frames, 1))
    valid_betas = valid_betas[valid[sub_id]]
    valid_global_poses = global_poses[valid[sub_id]]
    gender = genders[sub_id]
    # consider only valid frames
    for valid_i in range(valid_pose.shape[0]):
        pose = valid_pose[valid_i]
        extrinsics = valid_global_poses[valid_i][:3,:3]
        pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]
        poses_.append(pose)
        shapes_.append(valid_betas[valid_i])
        genders_.append(gender)

    poses = np.array(poses_)
    shapes = np.array(shapes_)
    gender = np.array([0 if str(g) == 'm' else 1 for g in genders_]).astype(np.int32)
    # to tensor
    poses = torch.from_numpy(poses).to(device).float()
    betas = torch.from_numpy(shapes).to(device).float()

    J_regressor_batch = J_regressor[None, :].expand(len(poses), -1, -1).to(device)
    gt_vertices = smpl_male(global_orient=poses[:,:3], body_pose=poses[:,3:], betas=betas).vertices # (B, 6890, 3)
    gt_vertices_female = smpl_female(global_orient=poses[:,:3], body_pose=poses[:,3:], betas=betas).vertices
    gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
    cam_3d = torch.matmul(J_regressor_batch, gt_vertices).detach().cpu().numpy() # (B, 17, 3)
    return cam_3d

def get_3dpw_smpl_cam_3d_hat():
    from hpe_library.my_utils import savepkl, readpkl
    user = getpass.getuser()
    save_path_smpl_cam_3d_hat = f'/home/{user}/codes/MotionBERT/custom_codes/dataset_generation/3dpw/3dpw-smpl_cam_3d_hat.pkl'
    if os.path.exists(save_path_smpl_cam_3d_hat):
        print(f"load 3dpw smpl_cam_3d_hat from {save_path_smpl_cam_3d_hat}")
        smpl_cam_3ds_hat = readpkl(save_path_smpl_cam_3d_hat)
    else:
        from hpe_library.spin_utils import SMPL, spin_config
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # craete SMPL model
        smpl_male = SMPL(spin_config.SMPL_MODEL_DIR, gender='male', create_transl=False).to(device)
        smpl_female = SMPL(spin_config.SMPL_MODEL_DIR, gender='female', create_transl=False).to(device)
        # load SMPL regressor
        J_regressor = torch.from_numpy(np.load(spin_config.JOINT_REGRESSOR_H36M)).float() # torch.Size([17, 6890])
        smpl_cam_3ds_hat = {}
        for source in tqdm(source_list, desc='generate cam_3d'):
            data_type = source.split('_')[0]
            seq_name = source.removeprefix(f"{data_type}_")[:-2]
            sub_id = int(source.split('_')[-1])
            #print(data_type, seq_name, sub_id)
            if data_type not in smpl_cam_3ds_hat: smpl_cam_3ds_hat[data_type] = {}
            if seq_name not in smpl_cam_3ds_hat: smpl_cam_3ds_hat[data_type][seq_name] = {}
            regressed_joint = get_3dpw_smpl_regressed_joint(data_type, seq_name, sub_id, J_regressor, smpl_male, smpl_female)
            regressed_joint_hat = regressed_joint - regressed_joint[:, [0], :]
            smpl_cam_3ds_hat[data_type][seq_name][sub_id] = regressed_joint_hat
        savepkl(smpl_cam_3ds_hat, save_path_smpl_cam_3d_hat)
    return smpl_cam_3ds_hat


def verify_3dpw_seq_datatype(seq_name):
    train_root = '/home/hrai/Datasets/HAAI/3DPW/original/sequenceFiles/sequenceFiles/train'
    test_root = '/home/hrai/Datasets/HAAI/3DPW/original/sequenceFiles/sequenceFiles/test'
    valid_root = '/home/hrai/Datasets/HAAI/3DPW/original/sequenceFiles/sequenceFiles/validation'
    if seq_name in [item.removesuffix('.pkl') for item in os.listdir(train_root)]:   return 'train'
    elif seq_name in [item.removesuffix('.pkl') for item in os.listdir(test_root)]: return 'test'
    elif seq_name in [item.removesuffix('.pkl') for item in os.listdir(valid_root)]: return 'valid'
    else:
        print(f"seq_name: {seq_name} not found in train, test, valid")
        return None

def find_closest_frame_from_poseaug_3dpw(target):
    target = target.copy() - target[0] # root-relative
    data_poseaug = np.load('/home/hrai/Datasets/HAAI/3DPW/poseaug/test_3dpw_gt.npz', allow_pickle=True)
    pose3d_poseaug = data_poseaug['pose3d']
    closest_frame = -1
    closest_dist = np.inf
    closest_pose = None
    for i in range(len(pose3d_poseaug)):
        pose3d = pose3d_poseaug[i]
        pose3d = pose3d - pose3d[0] # root-relative
        dist = np.mean(np.linalg.norm(target-pose3d, axis=0))
        if dist < closest_dist:
            closest_dist = dist
            closest_frame = i
            closest_pose = pose3d
    return closest_frame, closest_pose, closest_dist

def find_closest_frame_from_original_3dpw(target, mode='smpl', group=['train', 'valid', 'test']):
    assert mode in ['smpl', 'joint'], f"mode should be 'smpl' or 'joint', but got {mode}"
    roots = []
    if 'train' in group: roots.append('sequenceFiles/sequenceFiles/train')
    if 'valid' in group: roots.append('sequenceFiles/sequenceFiles/validation')
    if 'test' in group: roots.append('sequenceFiles/sequenceFiles/test')
    assert len(roots) > 0, f"no group found in {group}"

    closest_frame = -1
    closest_dist = np.inf
    closest_pose = None
    closest_seq = None
    closest_data_type = None

    target_pose = target.copy() - target[0] # root-relative
    for root in roots:
        for item in glob(root+'/*'):
            with open(item, 'rb') as f:
                test_data = pickle.load(f, encoding='latin1')
                seq_name = test_data['sequence'].copy()
                jointPositions = test_data['jointPositions'].copy()
                num_subjects = len(jointPositions)
                num_frames = len(test_data['img_frame_ids'])
                campose_valid = np.array(test_data['campose_valid'].copy()).astype(bool)[0]
                cam_extrinsics = test_data['cam_poses'].copy()
                for subject_id in range(num_subjects):
                    smpl_world_3d = jointPositions[subject_id].reshape(-1, 24, 3) # N, 24, 3
                    for frame_num in range(num_frames):
                        t = cam_extrinsics[frame_num][:3, 3]
                        R = cam_extrinsics[frame_num][:3, :3]
                        smpl_cam_3d = np.einsum('ij,jk->ik', smpl_world_3d[frame_num], R.T) + t
                        cam_3d = smpl2h36m(smpl_cam_3d, with_nose=False)[0]
                        cam_3d_hat = cam_3d.copy() - cam_3d[0]
                        # mpjpe
                        dist = np.linalg.norm(cam_3d_hat - target_pose, axis=0).mean()
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_frame = frame_num
                            closest_pose = cam_3d_hat
                            closest_seq = seq_name
                            closest_data_type = root

    return closest_frame, closest_pose, closest_dist, closest_seq, closest_data_type

