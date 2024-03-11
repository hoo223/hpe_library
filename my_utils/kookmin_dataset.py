from lib_import import *
from .test_utils import kookmin2h36m, World2CameraCoordinate, get_rootrel_pose, infer_box, camera_to_image_frame, optimize_scaling_factor, savepkl, get_video_info, get_video_frame, kookmin2h36m_with_nose
from .dh import rotate_torso_by_R


LBOT = np.array([-0.53090762,  1.47796592,  0.00880748])

def get_lbot(collection_phase):
    if collection_phase == 1:
        return np.array([-0.53090762,  1.47796592,  0.00880748])
    elif collection_phase == 2:
        return np.array([-0.69327857,  0.53428134,  0.00267539])
    else:
        assert False, 'Invalid collection phase'

def get_cam_param_kookmin(cam_param, subject, camera_id, action=None):
    if action == None:
        H, W = cam_param[subject][camera_id]['H'], cam_param[subject][camera_id]['W']
        R = cam_param[subject][camera_id]['R']
        t = cam_param[subject][camera_id]['t'].reshape(3, 1) 
        C = cam_param[subject][camera_id]['C']
        calibration_matrix = cam_param[subject][camera_id]['intrinsic']
        image = cam_param['for_pnp_data'][subject][camera_id]['image']
    else:
        H, W = cam_param[subject][action][camera_id]['H'], cam_param[subject][action][camera_id]['W']
        R = cam_param[subject][action][camera_id]['R']
        t = cam_param[subject][action][camera_id]['t'].reshape(3, 1) 
        C = cam_param[subject][action][camera_id]['C']
        calibration_matrix = cam_param[subject][action][camera_id]['intrinsic']
        image = cam_param['for_pnp_data'][subject][action][camera_id]['image']
    return H, W, R, t, C, calibration_matrix, image


def load_csv_kookmin(subject, action, phase, root='/home/hrai/Datasets/HAAI/국민대데이터/data/gist data/'):
    if phase == '':
        if os.path.exists(root + '{}/{}_{}_interpolated.csv'.format(subject, subject, action)):
            file_path = root + '{}/{}_{}_interpolated.csv'.format(subject, subject, action)
        else:
            file_path = root + '{}/{}_{}.csv'.format(subject, subject, action)
    else:
        if os.path.exists(root + '{}/{}_{}_{}_interpolated.csv'.format(subject, subject, action, phase)):
            file_path = root + '{}/{}_{}_{}_interpolated.csv'.format(subject, subject, action, phase)
        else:
            file_path = root + '{}/{}_{}_{}.csv'.format(subject, subject, action, phase)
    f = open(file_path, 'r', encoding='utf-8')
    rdr = list(csv.reader(f))
    f.close() 
    return rdr

def load_pose3d_kookmin(subject, action, phase, origin=np.zeros(3), root='/home/hrai/Datasets/HAAI/국민대데이터/data/gist data/', h36m=False, with_nose=True):
    rdr = load_csv_kookmin(subject, action, phase, root)
    pose_3d_list = []
    attribute_list = []
    for i, line in enumerate(rdr):
        if i == 0:
            first_line = line
            for item in first_line[2:]:
                attribute_list.append(item[:-2])
            attribute_list = list(dict.fromkeys(attribute_list))
            continue
        num_joint = int(len(line[2:])/3)
        pose_3d = np.zeros((num_joint, 3))
        for j in range(num_joint):
            try:
                pose_3d[j, 0] = float(line[2+3*j])
                pose_3d[j, 1] = float(line[2+3*j+1])
                pose_3d[j, 2] = float(line[2+3*j+2])
            except: # if there is no data
                pose_3d[j] = origin
        pose_3d = pose_3d - origin
        pose_3d_list.append(pose_3d)
    
    pose_3d_list = np.array(pose_3d_list)
    if h36m:
        if with_nose:
            pose_3d_list = kookmin2h36m_with_nose(pose_3d_list)
        else:
            pose_3d_list = kookmin2h36m(pose_3d_list)
    
    return pose_3d_list, attribute_list

def draw_base_marker_3d(ax, lbot, ltop, rbot, rtop, center1, center2):
    ax.plot(lbot[0]-lbot[0], lbot[1]-lbot[1], lbot[2]-lbot[2], 'rx') # origin
    ax.plot(ltop[0], ltop[1], ltop[2], 'ro')
    ax.plot(rbot[0], rbot[1], rbot[2], 'ro')
    ax.plot(rtop[0], rtop[1], rtop[2], 'ro')
    ax.plot(center1[0], center1[1], center1[2], 'ro')
    ax.plot(center2[0], center2[1], center2[2], 'ro')
    
def get_video_frame_kookmin(subject, action, camera_id, frame_id=None):
    video_path = '/home/hrai/Datasets/HAAI/국민대데이터/data/videos/{}/{}_{}_{}.mp4'.format(subject, subject, action, camera_id)
    #video_path = glob('/home/hrai/Datasets/HAAI/국민대데이터/data/videos/{}/{}_{}_{}.*'.format(subject, subject, action, camera_id))[0]
    return get_video_frame(video_path, frame_id)
    
def get_video_num_frame_kookmin(subject, action, camera_id):
    video_path = '/home/hrai/Datasets/HAAI/국민대데이터/data/videos/{}/{}_{}_{}.mp4'.format(subject, subject, action, camera_id)
    width, height, num_frame, fps = get_video_info(video_path)
    return num_frame

def check_continuity(l):
    for i in range(len(l)-1):
        if l[i] != l[i+1] - 1:
            return False
    return True

# check available frames
def check_available_frame(pose3d_list, necessary_joints, interpolate_spine=False, num_joints=25, verbose=False):
    # num_joints: 25 for first kookmin collection
    #             26 for second kookmin collection
    available_frames = []
    missing_frames = []
    for frame in range(len(pose3d_list)):
        available_joint = np.where(np.linalg.norm(pose3d_list[frame], axis=1)!=0)[0]
        if len(available_joint) == num_joints: # all joints are available
            available_frames.append(frame)
        else: # some joints are missing
            if all(x in available_joint for x in necessary_joints): # if all necessary joints are available
                available_frames.append(frame)
                # interpolate spine if necessary
                if interpolate_spine and (2 not in available_joint):
                    l_hip = pose3d_list[frame, 13, :]
                    r_hip = pose3d_list[frame, 14, :]
                    pelvis = (l_hip + r_hip)/2
                    neck = pose3d_list[frame, 1, :]
                    pose3d_list[frame, 2, :] = (pelvis + neck) / 2 
            else:
                missing_frames.append(frame)
    
    num_frames = len(available_frames)
    if abs(num_frames - len(pose3d_list)) > len(pose3d_list)/2:
        print('Not enough available frames ({} frames)'.format(num_frames))
        return None, 0
    else:
        if check_continuity(available_frames):
            if verbose:
                print('Available frames: {} ~ {}'.format(available_frames[0], available_frames[-1]))
            return available_frames, num_frames
        else:
            print('Not continuous')
            return None, 0
        
def generate_kookmin_pkl_for_each_video(pose3d_list, available_frames, subject, camera_id, action, phase, camera_param, save_folder, trans=None, rot=None, centered_root=False, overwrite=False):
    '''
    pose3d_list: (num_frames, num_joints, 3)
    available_frames: list of available frames
    subject: subject id
    camera_id: camera id
    action: action type
    phase: 001 or 002 or 003
    camera_param: camera parameters - intrinsic, extrinsic
    save_folder: folder that pkl files are saved
    trans, rot: translation and rotation to match fit3d setting
    overwrite: overwrite to existing pkl files
    '''
    if phase == '':
        source = '{}_{}_{}'.format(subject, camera_id, action)
    else:
        source = '{}_{}_{}_{}'.format(subject, camera_id, action, phase)
    
    data = {}
    for key in ['joint_2d', 'confidence', 'joint3d_image', 'joints_2.5d_image', '2.5d_factor', 'camera_name', 'action', 'source', 'frame', 'world_3d', 'cam_3d', 'cam_param']:
        data[key] = []
    
    file_name = source + '.pkl'
    save_path = os.path.join(save_folder, file_name) 
    if os.path.exists(save_path):
        if not overwrite:
            print(save_path, 'exists')
            return 0
    
    fx = camera_param['intrinsic'][0, 0]  
    fy = camera_param['intrinsic'][1, 1]  
    cx = camera_param['intrinsic'][0, 2]  
    cy = camera_param['intrinsic'][1, 2] 
    
    if centered_root:
        trans = np.zeros(3)
    
    for i in tqdm(range(len(available_frames))):
        frame_num = available_frames[i]
        # joint3d_image
        world_3d = np.array(pose3d_list[frame_num])
        modified = world_3d.copy()
        if trans is not None:
            modified -= np.array([modified[0][0], modified[0][1], 0])
            modified += np.array([trans[0], trans[1], 0])
            #modified += trans
        if rot is not None:
            modified = rotate_torso_by_R(modified, rot)
        world_3d = modified

        # world to camera
        pos = copy.deepcopy(world_3d)
        cam_3d = World2CameraCoordinate(pos, camera_param['extrinsic']) * 1000 # World coordinate -> Camera coordinate
        cam_3d_hat = get_rootrel_pose(cam_3d)

        # camera to image
        box = infer_box(cam_3d, {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}, 0)
        img_2d, img_3d = camera_to_image_frame(cam_3d, box, {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}, 0) 
        #img_2d_hat = get_rootrel_pose(img_2d) # (17, 2) # root-relative pose 
        img_3d_hat = get_rootrel_pose(img_3d) # (17, 3) # root-relative pose 

        # 2.5d factor
        pred_lambda, losses = optimize_scaling_factor(img_3d_hat, cam_3d_hat, stop_tolerance=0.0001) # x,y,z 사용
        #print('losses:', losses)

        # joint 2.5d image
        img_25d = img_3d * pred_lambda
        #img_25d_hat = get_rootrel_pose(img_25d)

        # store
        data['joint_2d'].append(np.array(img_2d).copy()) 
        data['confidence'].append(np.ones(17)) 
        data['joint3d_image'].append(np.array(img_3d).copy()) 
        data['joints_2.5d_image'].append(np.array(img_25d).copy()) 
        data['2.5d_factor'].append(np.array(pred_lambda).copy()) 
        data['camera_name'].append(np.array(camera_id).copy()) 
        data['action'].append(np.array(action).copy()) 
        data['source'].append(np.array(source).copy()) 
        data['frame'].append(np.array(frame_num).copy()) 
        data['world_3d'].append(np.array(world_3d).copy()) 
        data['cam_3d'].append(np.array(cam_3d).copy()) 
        data['cam_param'].append(np.array(camera_param).copy()) 
        #break
    # save
    savepkl(data, save_path)
    
    return 1

# LEGACY 
# def generate_kookmin_pkl_for_each_video(pose3d_list, available_frames, subject, camera_id, action, phase, camera_param, save_folder, trans=None, rot=None, overwrite=False):
#     '''
#     pose3d_list: (num_frames, num_joints, 3)
#     available_frames: list of available frames
#     subject: subject id
#     camera_id: camera id
#     action: action type
#     phase: 001 or 002 or 003
#     camera_param: camera parameters - intrinsic, extrinsic
#     save_folder: folder that pkl files are saved
#     trans, rot: translation and rotation to match fit3d setting
#     overwrite: overwrite to existing pkl files
#     '''
#     if '{}_{}_{}_{}.pkl'.format(subject, camera_id, action, phase) in os.listdir(save_folder): 
#         print('{}_{}_{}.pkl'.format(subject, camera_id, action), 'exists')
#         if not overwrite:
#             return 0
    
#     data = {}
#     for key in ['joint_2d', 'confidence', 'joint3d_image', 'joints_2.5d_image', '2.5d_factor', 'camera_name', 'action', 'source', 'frame', 'world_3d', 'cam_3d', 'cam_param']:
#         data[key] = []
    
#     source = '{}_{}_{}_{}'.format(subject, camera_id, action, phase) 
#     file_name = source + '.pkl'
#     save_path = os.path.join(save_folder, file_name) 
    
#     fx = camera_param['intrinsic'][0, 0]  
#     fy = camera_param['intrinsic'][1, 1]  
#     cx = camera_param['intrinsic'][0, 2]  
#     cy = camera_param['intrinsic'][1, 2] 
    
#     if trans is None: trans = np.zeros(3)
#     if rot is None: rot = np.eye(3)
    
#     for i in tqdm(range(len(available_frames))):
#         frame_num = available_frames[i]
#         # joint3d_image
#         world_3d = np.array(pose3d_list[frame_num])
#         modified = world_3d.copy()
#         modified -= np.array([modified[0][0], modified[0][1], 0])
#         modified += np.array([trans[0], trans[1], 0])
#         world_3d = rotate_torso_by_R(modified, rot)

#         # world to camera
#         pos = copy.deepcopy(world_3d)
#         cam_3d = World2CameraCoordinate(pos, camera_param['extrinsic']) * 1000 # World coordinate -> Camera coordinate
#         cam_3d_hat = get_rootrel_pose(cam_3d)

#         # camera to image
#         box = infer_box(cam_3d, {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}, 0)
#         img_2d, img_3d = camera_to_image_frame(cam_3d, box, {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}, 0) 
#         #img_2d_hat = get_rootrel_pose(img_2d) # (17, 2) # root-relative pose 
#         img_3d_hat = get_rootrel_pose(img_3d) # (17, 3) # root-relative pose 

#         # 2.5d factor
#         pred_lambda = optimize_scaling_factor(img_3d_hat, cam_3d_hat) # x,y,z 사용

#         # joint 2.5d image
#         img_25d = img_3d * pred_lambda
#         #img_25d_hat = get_rootrel_pose(img_25d)

#         # store
#         data['joint_2d'].append(np.array(img_2d).copy()) 
#         data['confidence'].append(np.ones(17)) 
#         data['joint3d_image'].append(np.array(img_3d).copy()) 
#         data['joints_2.5d_image'].append(np.array(img_25d).copy()) 
#         data['2.5d_factor'].append(np.array(pred_lambda).copy()) 
#         data['camera_name'].append(np.array(camera_id).copy()) 
#         data['action'].append(np.array(action).copy()) 
#         data['source'].append(np.array(source).copy()) 
#         data['frame'].append(np.array(frame_num).copy()) 
#         data['world_3d'].append(np.array(world_3d).copy()) 
#         data['cam_3d'].append(np.array(cam_3d).copy()) 
#         data['cam_param'].append(np.array(camera_param).copy()) 

#     # save
#     savepkl(data, save_path)
    
#     return 1