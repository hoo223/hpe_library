from hpe_library.lib_import import *

def data_augmentation(pose3d, data_aug):
    assert len(pose3d.shape) == 3, f'pose3d shape should be (num_frames, num_joints, 3), but got {pose3d.shape}'
    assert pose3d.shape[1] == 17, f'pose3d shape should be (num_frames, 17, 3), but got {pose3d.shape}'
    
    pose3d_hat = pose3d.copy() - pose3d[:, 0:1]
    if data_aug['step_rot'] > 0:
        rot = Rotation.from_euler('y', np.deg2rad(data_aug['step_rot'])*np.arange(0, len(pose3d))).as_matrix()
    elif data_aug['sinu_yaw_mag'] > 0:
        sinu_yaw_mag = data_aug['sinu_yaw_mag']
        sinu_yaw_period = data_aug['sinu_yaw_period']
        rot = Rotation.from_euler('y', np.deg2rad(sinu_yaw_mag)*np.sin(np.arange(0, len(pose3d))/sinu_yaw_period*2*np.pi)).as_matrix()
    elif data_aug['rand_yaw_mag'] > 0:
        sinu_yaw_mag = random.randrange(-data_aug['rand_yaw_mag'], data_aug['rand_yaw_mag'])
        if data_aug['rand_yaw_period'] > 0: sinu_yaw_period = data_aug['sinu_yaw_period'] + random.randrange(-data_aug['rand_yaw_period'], data_aug['rand_yaw_period'])
        else : sinu_yaw_period = data_aug['sinu_yaw_period']
        rot = Rotation.from_euler('y', np.deg2rad(sinu_yaw_mag)*np.sin(np.arange(0, len(pose3d))/sinu_yaw_period*2*np.pi)).as_matrix()
    else:
        rot = np.eye(3).reshape(1, 3, 3).repeat(len(pose3d), axis=0)
    new_pose3d = np.einsum('fij,fkj->fki', rot, pose3d_hat)
    new_pose3d += pose3d[:, 0:1]
    pose3d = new_pose3d
        
    pose3d_hat = pose3d.copy() - pose3d[:, 0:1]
    if data_aug['sinu_pitch_mag'] > 0:
        sinu_pitch_mag = data_aug['sinu_pitch_mag']
        sinu_pitch_period = data_aug['sinu_pitch_period']
        rot = Rotation.from_euler('x', np.deg2rad(sinu_pitch_mag)*np.sin(np.arange(0, len(pose3d))/sinu_pitch_period*2*np.pi)).as_matrix()
    elif data_aug['rand_pitch_mag'] > 0:
        sinu_pitch_mag = random.randrange(-data_aug['rand_pitch_mag'], data_aug['rand_pitch_mag'])
        if data_aug['rand_pitch_period'] > 0: sinu_pitch_period = data_aug['sinu_pitch_period'] + random.randrange(-data_aug['rand_pitch_period'], data_aug['rand_pitch_period'])
        else: sinu_pitch_period = data_aug['sinu_pitch_period']
        rot = Rotation.from_euler('x', np.deg2rad(sinu_pitch_mag)*np.sin(np.arange(0, len(pose3d))/sinu_pitch_period*2*np.pi)).as_matrix()
    else:
        rot = np.eye(3).reshape(1, 3, 3).repeat(len(pose3d), axis=0)
    new_pose3d = np.einsum('fij,fkj->fki', rot, pose3d_hat)
    new_pose3d += pose3d[:, 0:1]
    pose3d = new_pose3d
    
    pose3d_hat = pose3d.copy() - pose3d[:, 0:1]
    if data_aug['sinu_roll_mag'] > 0:
        sinu_roll_mag = data_aug['sinu_roll_mag']
        sinu_roll_period = data_aug['sinu_roll_period']
        rot = Rotation.from_euler('z', np.deg2rad(sinu_roll_mag)*np.sin(np.arange(0, len(pose3d))/sinu_roll_period*2*np.pi)).as_matrix()
    elif data_aug['rand_roll_mag'] > 0:
        sinu_roll_mag = random.randrange(-data_aug['rand_roll_mag'], data_aug['rand_roll_mag'])
        if data_aug['rand_roll_period'] > 0: sinu_roll_period = data_aug['sinu_roll_period'] + random.randrange(-data_aug['rand_roll_period'], data_aug['rand_roll_period'])
        else: sinu_roll_period = data_aug['sinu_roll_period']
        rot = Rotation.from_euler('z', np.deg2rad(sinu_roll_mag)*np.sin(np.arange(0, len(pose3d))/sinu_roll_period*2*np.pi)).as_matrix()
    else:
        rot = np.eye(3).reshape(1, 3, 3).repeat(len(pose3d), axis=0)
    new_pose3d = np.einsum('fij,fkj->fki', rot, pose3d_hat)
    new_pose3d += pose3d[:, 0:1]
    pose3d = new_pose3d
    
    return pose3d