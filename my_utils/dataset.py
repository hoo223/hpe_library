from lib_import import *
from .dh import rotate_torso_by_R, get_torso_direction, rotation_matrix_to_vector_align, projection, get_torso_rotation_matrix
from .test_utils import readJSON, halpe2h36m, get_video_info, get_bbox_area_from_pose2d, get_bbox_from_pose2d, change_bbox_convention, get_bbox_area
from .test_utils import get_h36m_keypoint_index

## for general

def load_h36m():
    # camera parameters
    cam_param = readJSON('/home/hrai/codes/hpe_library/data/h36m_camera-parameters.json')

    print('==> Loading 3D data wrt World CS...')
    h36m_3d_world = Human36mDataset('/home/hrai/codes/hpe_library/data/data_3d_h36m.npz', remove_static_joints=True)

    return h36m_3d_world, cam_param

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
    pose3d = h36m_3d_world._data[subject][action]['positions'] # 3d skeleton sequence wrt world CS
    cam_info = h36m_3d_world._data[subject][action]['cameras']
    cam_param = get_cam_param(cam_info, subject, h36m_cam_param)
    return pose3d, cam_param

def get_part_traj(pose_traj, part):
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
    from my_utils import get_torso_rotation_matrix
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
                input = get_model_input(self.input_list, device=None, input_dict=temp_pair)

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