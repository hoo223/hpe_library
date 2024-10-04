from hpe_library.lib_import import *
# from .config import get_configs
# from .dataset import get_model_input
# from .dh import get_torso_direction, rotate_torso_by_R, projection
# from .train import load_args, load_best_model, load_dataset

def args_dict_to_namespace(args_dict, arg_blacklist=['num_trial', 'save_model_path', 'test_loss_lowest', 'best_epoch', 'segment_file', 'input_candidate', 'output_candidate', 'device'], with_result=True):
    from hpe_library.my_utils import get_configs
    arg_list = []
    for arg in args_dict.keys():
        # blacklist
        if arg in arg_blacklist:
            continue
        if args_dict[arg] == None: # use default value if argument value is None
            continue
        arg_list.append('--'+arg)
        # if args_dict[arg] in [True, False]: # boolean은 str로 변환하면 parser에서 무조건 True로 인식함
        #     arg_list.append(args_dict[arg]) 
        # else:                        
        arg_list.append(str(args_dict[arg]))
        #print(arg, args_dict[arg])
    #print(arg_list)

    args = get_configs(arg_list)
    if with_result:
        args.num_trial = args_dict['num_trial']
        args.save_model_path = args_dict['save_model_path']
        args.test_loss_lowest = args_dict['test_loss_lowest']
        args.best_epoch = args_dict['best_epoch']
        args.segment_file = args_dict['segment_file']
    return args

def get_result(model, input, src_torso, tar_torso=None, label=None, output_type='dp_rotvec_th'):
    from hpe_library.my_utils import get_torso_direction, rotate_torso_by_R
    output = model(input)
    #src_torso = src_torso.cpu().detach().numpy().reshape(5, 3)
    gt_torso = tar_torso 
    if output_type == 'torso':
        pred_torso = output.cpu().detach().numpy().reshape(5, 3)
        if label is not None:
            gt_torso = label.cpu().detach().numpy().reshape(5, 3)
    elif output_type == 'delta_torso':
        pred_torso = output[0].cpu().detach().numpy().reshape(5, 3) + src_torso
        if label is not None:
            gt_torso = label.cpu().detach().numpy().reshape(5, 3) + src_torso
        elif tar_torso is not None:
            gt_torso = tar_torso
    elif output_type == 'dp':
        pred_delta_point = output
        return pred_delta_point
    elif output_type == 'dp_rotvec_th':
        pred_delta_point, pred_rotvec, pred_theta = output
        pred_delta_point = pred_delta_point.cpu().detach().numpy()
        pred_rotvec = pred_rotvec.cpu().detach().numpy()
        pred_theta = pred_theta.cpu().detach().numpy()
        pred_rotmat = Rotation.from_rotvec(pred_theta*pred_rotvec).as_matrix()
        pred_torso = rotate_torso_by_R(src_torso, pred_rotmat[0]) + pred_delta_point
        
        if label is not None:
            #print('label in')
            gt_delta_point, gt_rotvec, gt_theta = label[:, :3], label[:, 3:6], label[:, 6]
            gt_rotmat = Rotation.from_rotvec(gt_theta.cpu().detach()*gt_rotvec.cpu().detach()).as_matrix()
            gt_torso = rotate_torso_by_R(src_torso, gt_rotmat[0]) + gt_delta_point.cpu().detach().numpy()
        elif tar_torso is not None:
            src_direction = get_torso_direction(src_torso)
            tar_direction = get_torso_direction(tar_torso)
            gt_delta_point = tar_torso - src_torso
            gt_rotvec = np.cross(src_direction, tar_direction) 
            gt_theta  = np.arccos(np.dot(src_direction, tar_direction))
            #print(pred_rotvec.cpu().detach().numpy(), gt_rotvec)
            #print(pred_theta.cpu().detach().numpy(), gt_theta)
            gt_rotmat = Rotation.from_rotvec(gt_theta*gt_rotvec).as_matrix()
            #gt_torso = rotate_torso_by_R(src_torso, pred_rotmat) + gt_delta_point
        print(pred_theta, gt_theta)
    elif output_type == 'gt_dp_rotvec_th':
        if tar_torso is not None:
            src_direction = get_torso_direction(src_torso)
            tar_direction = get_torso_direction(tar_torso)
            gt_delta_point = tar_torso - src_torso
            gt_rotvec = np.cross(src_direction, tar_direction) 
            gt_theta  = np.arccos(np.dot(src_direction, tar_direction))
            pred_rotmat = Rotation.from_rotvec(gt_theta*gt_rotvec).as_matrix()
            pred_torso = rotate_torso_by_R(src_torso, pred_rotmat) + gt_delta_point
    elif output_type == 'dq_quat':
        pred_delta_point, pred_quat = output
        pred_rotmat = R.from_quat(pred_quat.cpu().detach()).as_matrix()
        pred_torso = rotate_torso_by_R(src_torso, pred_rotmat[0]) + pred_delta_point.cpu().detach().numpy()
        # if label is not None:
        #     gt_delta_point, gt_quat = label[:, :3], label[:, 3:]
        #     gt_rotmat = R.from_quat(gt_quat.cpu().detach()).as_matrix()
        #     gt_torso = rotate_torso_by_R(src_torso, gt_rotmat[0]) + gt_delta_point.cpu().detach().numpy()

    return pred_torso, gt_torso


def get_output_type(output_list):
    output_type = ''
    for output in output_list:
        if 'tar_torso'          == output: output_type += 'tt_'
        elif 'tar_delta_torso'  == output: output_type += 'dt_'
        elif 'tar_delta_point'  == output: output_type += 'dp_'
        elif 'tar_delta_quat'   == output: output_type += 'dq_'
        elif 'tar_delta_rotvec' == output: output_type += 'drv'
        
    output_type = output_type[:-1]
    return output_type


def construct_torso_from_output(output_type, output, src_torso):
    from hpe_library.my_utils import rotate_torso_by_R
    pred_delta_rot = np.eye(3)
    if output_type in ['dp']:
        pred_delta_point = output[0].detach().cpu().numpy()
    elif output_type in ['dt', 'tt']:
        pred_delta_point = output[0].detach().cpu().numpy().reshape(5,3)
    elif output_type == 'dp_dq': 
        pred_delta_point, pred_delta_quat = output
        pred_delta_point = pred_delta_point[0].detach().cpu().numpy()
        pred_delta_quat = pred_delta_quat[0].detach().cpu().numpy()
        try:
            pred_delta_rot = Rotation.from_quat(pred_delta_quat).as_matrix()
        except:
            pass
    pred_torso = rotate_torso_by_R(src_torso, pred_delta_rot) + pred_delta_point
    return pred_torso

# 
def load_best_model_for_inference(out_dir, test_trial):
    from hpe_library.my_utils import load_args, load_best_model, load_dataset
    # --------------------------------------------- Load args
    args_dict = load_args(out_dir, test_trial) # load args dict 
    args = args_dict_to_namespace(args_dict) # convert args dict to namespace
    # --------------------------------------------- Load best model
    # Load dataset
    args, targs = load_dataset(args, auto_load_data=False)
    # Set device
    assert torch.cuda.is_available(), "CUDA is not available."
    device = torch.device("cuda" if args.use_cuda else "cpu")
    # model
    model = load_best_model(args, targs, eval=True, device=device)

    return model, args, targs


def infer_one_segment(model, args, segment, cam_proj, stride, device='cuda', use_gt_torso=False, use_pred_2d=False):
    from hpe_library.my_utils import get_model_input, get_output_type, projection
    assert isinstance(segment, dict), 'segment must be a dict'
    assert 'torsos' in segment.keys(), 'segment must have key "torsos"'
    assert 'rots' in segment.keys(), 'segment must have key "rots"'
    assert 'torsos_projected' in segment.keys(), 'segment must have key "torsos_projected"'
    assert len(segment['torsos']) > 1, 'segment must have more than 1 frame'
    
    torsos = segment['torsos']
    rots = segment['rots']
    torsos_projected = segment['torsos_projected']
    #print('num frames', len(torsos))
    
    # --------------------------------------------- Get output type
    output_type = get_output_type(args.output_list)
    #print(output_type)

    # --------------------------------------------- Inference
    preds_3d = [torsos[0]]
    preds_2d = [torsos_projected[0]]
    gts_3d = [torsos[0]]
    gts_2d = [torsos_projected[0]]
    prev_torso = torsos[0]
    prev_torso_projected = torsos_projected[0]
    for i in range(0, len(torsos)-stride, stride):
        # Prepare inputs
        src_torso = torsos[i].copy() if use_gt_torso else prev_torso 
        src_2d_old = prev_torso_projected if use_pred_2d else torsos_projected[i].copy()
        tar_torso = torsos[i+1].copy() 
        src_2d_new = torsos_projected[i+1].copy()
        src_rot = rots[i].copy()

        # Make input for model
        input = get_model_input(args.input_list, src_torso=src_torso, src_rot=src_rot, src_2d_old=src_2d_old, src_2d_new=src_2d_new)
        # label = get_label(args.output_list,
        #                   src_torso=src_torso, src_rot=srt_rot, tar_torso=tar_torso, tar_rot=tar_rot)
        # Inference
        output = model(input)

        # Construct torso from output
        pred_torso = construct_torso_from_output(output_type, output, src_torso)
        # Project torso to 2D
        pred_torso_projected = projection(pred_torso, cam_proj)
        # Store 
        preds_3d.append(pred_torso)
        preds_2d.append(pred_torso_projected)
        gts_3d.append(tar_torso)
        gts_2d.append(src_2d_new*1000)
        # Update previous values
        prev_torso = pred_torso
        prev_torso_projected = pred_torso_projected
    # Convert to numpy
    gts_3d = np.array(gts_3d)
    preds_3d = np.array(preds_3d)
    gts_2d = np.array(gts_2d)
    preds_2d = np.array(preds_2d)
    
    return preds_3d, preds_2d, gts_3d, gts_2d

def test_model_by_segment_file(out_dir, test_trial, segment_file='', data_type='test', stride=5, use_gt_torso=False, use_pred_2d=False):
    # --------------------------------------------- Load model
    model, args, targs = load_best_model_for_inference(out_dir, test_trial)

    # --------------------------------------------- Load test segment
    if segment_file == '': segment_file = args.segment_file
    with open(file=segment_file, mode='rb') as f:
        traj_segment_dataset=pickle.load(f)
    test_segment = traj_segment_dataset[data_type]['cam1']['traj_segment']
    cam_proj = traj_segment_dataset[data_type]['cam1']['cam_param']['proj'] # camera projection matrix

    # --------------------------------------------- Calculate Averega MPJPE
    mpjpe_list = []
    for segment in tqdm(test_segment):
        if len(segment['torsos']) < 2: continue
        # --------------------------------------------- Inference
        preds_3d, preds_2d, gts_3d, gts_2d = infer_one_segment(model, args, segment, cam_proj, stride, use_gt_torso=use_gt_torso, use_pred_2d=use_pred_2d)
        # --------------------------------------------- Error (MPJPE)
        mpjpe = np.mean(np.linalg.norm(gts_3d-preds_3d, axis=-1))
        #print('3D error: ', mpjpe)
        mpjpe_list.append(mpjpe)
    avg_mpjpe = np.mean(mpjpe_list)
    
    return avg_mpjpe

def get_dataset_info_from_segment_folder(segment_folder):
    step_size, max_dist, max_deg, mac_bias, rt_type, xyz_range, _, stride, window_size = segment_folder.split('_')[5:]
    return float(step_size), max_dist, max_deg, mac_bias, rt_type, xyz_range, stride.split('stride')[1], window_size.split('window')[1]

def denormalize_motionbert_result(test_data, W, H):
    # data: (N, n_frames, 51) or data: (N, n_frames, 17, 3)        
    n_clips = test_data.shape[0]
    data = test_data.reshape([n_clips, -1, 17, 3])
    # denormalize (x,y,z) coordiantes for results
    for idx, item in enumerate(data):
        res_w, res_h = W, H
        data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
        data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
    return data # [n_clips, -1, 17, 3]

def get_inference_from_motionbert(model, input_data, args, W, H):
    model.eval()  
    output = []
    with torch.no_grad():
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        if args.flip:
            input_flip = flip_data(input_data)
            predicted_3d_pos_1 = model(input_data)
            predicted_3d_pos_flip = model(input_flip)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)                   # Flip back
            predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
        else:
            predicted_3d_pos = model(input_data)
        output.append(predicted_3d_pos.cpu().numpy())
    output = np.concatenate(output)
    #output = denormalize_motionbert_result(output, W, H)
    return output

def get_inference_from_dhdst(model, input_data, args, W, H, denormalize=False, input_type='video'):
    model.eval()  
    output = []
    with torch.no_grad():
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        if args.flip:
            input_flip = flip_data(input_data)
            predicted_3d_pos_1 = model(batch_input=input_data, length_type=args.test_length_type, ref_frame=args.length_frame)
            predicted_3d_pos_flip = model(batch_input=input_flip, length_type=args.test_length_type, ref_frame=args.length_frame)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)                   # Flip back
            predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
        else:
            predicted_3d_pos = model(batch_input=input_data, length_type=args.test_length_type, ref_frame=args.length_frame)
        output.append(predicted_3d_pos.cpu().numpy())
    output = np.concatenate(output)
    if denormalize:
        output = denormalize_motionbert_result(output, W, H)
    return output

def get_inference_from_dhdst_torso(model, input_data, args, W, H, denormalize=False, with_frame=False,  input_type='video'):
    model.eval()  
    pred_torso_output = []
    if with_frame:
        pred_lower_frame_R_output = []
        pred_upper_frame_R_output = []
    with torch.no_grad():
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        if args.flip:
            input_flip = flip_data(input_data)
            pred_torso_1 = model(batch_input=input_data)
            pred_torso_flip = model(batch_input=input_flip)
            pred_torso_2 = flip_data(pred_torso_flip)                   # Flip back
            pred_torso = (pred_torso_1+pred_torso_2) / 2
        else:
            if with_frame:
                pred_torso, pred_lower_frame_R, pred_upper_frame_R = model(batch_input=input_data)
            else:
                pred_torso = model(batch_input=input_data)
        pred_torso_output.append(pred_torso.cpu().numpy())
        if with_frame:
            pred_lower_frame_R_output.append(pred_lower_frame_R.cpu().numpy())
            pred_upper_frame_R_output.append(pred_upper_frame_R.cpu().numpy())
    pred_torso_output = np.concatenate(pred_torso_output)
    if with_frame:
        pred_lower_frame_R_output = np.concatenate(pred_lower_frame_R_output)
        pred_upper_frame_R_output = np.concatenate(pred_upper_frame_R_output)
    if denormalize:
        pred_torso_output = denormalize_motionbert_result(pred_torso_output, W, H)
    if with_frame:
        return pred_torso_output, pred_lower_frame_R_output, pred_upper_frame_R_output
    else:
        return pred_torso_output

def get_inference_from_DHDSTformer_limb(model, input_data, args, W, H, denormalize=False, input_type='video'):
    model.eval()  
    output = []
    with torch.no_grad():
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        if args.flip:
            input_flip = flip_data(input_data)
            predicted_3d_pos_1 = model(batch_input=input_data, length_type=args.test_length_type, ref_frame=args.length_frame)
            predicted_3d_pos_flip = model(batch_input=input_flip, length_type=args.test_length_type, ref_frame=args.length_frame)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)                   # Flip back
            predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
        else:
            predicted_3d_pos = model(batch_input=input_data, length_type=args.test_length_type, ref_frame=args.length_frame)
        output.append(predicted_3d_pos.cpu().numpy())
    output = np.concatenate(output)
    if denormalize:
        output = denormalize_motionbert_result(output, W, H)
    return output

def normalize_input(input_data, W, H):
    # input range: [0, W] -> [-1, 1]
    if input_data.shape[-1] == 3:
        return input_data / W * 2 - [1, H / W, 0]
    elif input_data.shape[-1] == 2:
        return input_data / W * 2 - [1, H / W]
    else:
        raise ValueError('Invalid input shape: {input_data.shape}')
    
def denormalize_input(input_data, W, H):
    # input range: [-1, 1] -> [0, W]
    if input_data.shape[-1] == 3:
        return (input_data + [1, H / W, 0]) * W / 2
    elif input_data.shape[-1] == 2:
        return (input_data + [1, H / W]) * W / 2
    else:
        raise ValueError('Invalid input shape: {input_data.shape}')

# def normalize_3d_pose(pose_3d, W, H):
#     # input range: [0, W] -> [-1, 1]
#     return (pose_3d + [1, H / W, 0]) * W / 2