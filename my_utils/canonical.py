from hpe_library.lib_import import *

def canonicalization_cam_3d(cam_3d, canonical_type):
    if len(cam_3d.shape) == 2:
        if 'same_z' in canonical_type:       dist = cam_3d[0, 2] # z value of pelvis joint for each frame 
        elif canonical_type == 'same_dist':  dist = np.linalg.norm(cam_3d[0], axis=1) # dist from origin to pelvis joint for each frame 
        elif 'fixed_dist' in canonical_type: dist = np.array([float(canonical_type.split('_')[-1])]*len(cam_3d)) 
        else: raise ValueError(f'canonical type {canonical_type} not found')
        cam_3d_canonical = cam_3d.copy() - cam_3d[0, None] # move to cam origin
        cam_3d_canonical[..., 2] += dist[None]
    elif len(cam_3d.shape) == 3:
        if 'same_z' in canonical_type:       dist = cam_3d[:, 0, 2] # z value of pelvis joint for each frame 
        elif canonical_type == 'same_dist':  dist = np.linalg.norm(cam_3d[:, 0], axis=1) # dist from origin to pelvis joint for each frame 
        elif 'fixed_dist' in canonical_type: dist = np.array([float(canonical_type.split('_')[-1])]*len(cam_3d)) 
        else: raise ValueError(f'canonical type {canonical_type} not found')
        cam_3d_canonical = cam_3d.copy() - cam_3d[:, 0, None] # move to cam origin
        cam_3d_canonical[..., 2] += dist[:, None]
    else:
        raise ValueError(f'cam_3d shape {cam_3d.shape} not supported')
    
    
    return cam_3d_canonical