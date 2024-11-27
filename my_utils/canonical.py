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

# Function to compute rotation matrices from 2D pose batch input
def compute_rotation_matrix(input_2d_p):
    # Extract pelvis coordinates (0th joint, [x, y])
    pelvis_coords = input_2d_p[..., 0, :]  # Shape: [B, F, 2]
    px = pelvis_coords[..., 0]  # Shape: [B, F]
    py = pelvis_coords[..., 1]  # Shape: [B, F]

    # Intermediate values
    one_plus_px2 = 1 + px**2  # Shape: [B, F]
    one_plus_px2_py2 = 1 + px**2 + py**2  # Shape: [B, F]

    # Rotation matrix components
    r11 = 1 / torch.sqrt(one_plus_px2)
    r12 = -px * py / torch.sqrt(one_plus_px2 * one_plus_px2_py2)
    r13 = px / torch.sqrt(one_plus_px2_py2)

    r21 = torch.zeros_like(r11)  # Shape: [B, F]
    r22 = torch.sqrt(one_plus_px2) / torch.sqrt(one_plus_px2_py2)
    r23 = py / torch.sqrt(one_plus_px2_py2)

    r31 = -px / torch.sqrt(one_plus_px2)
    r32 = -py / torch.sqrt(one_plus_px2 * one_plus_px2_py2)
    r33 = 1 / torch.sqrt(one_plus_px2_py2)

    # Construct the rotation matrix
    R = torch.stack([
        torch.stack([r11, r12, r13], dim=-1),  # First row
        torch.stack([r21, r22, r23], dim=-1),  # Second row
        torch.stack([r31, r32, r33], dim=-1)   # Third row
    ], dim=-2)  # Shape: [B, F, 3, 3]

    return R

# Function to apply rotation matrices to 3D pose output
def rotate_3d_pose(input_3d_pose, rotation_matrices):
    # Reshape input_3d_pose to match rotation_matrices dimensions for batch matrix multiplication
    B, F, J, C = input_3d_pose.shape  # Shape: [B, F, 17, 3]
    input_3d_pose_expanded = input_3d_pose.view(B, F, J, C, 1)  # Shape: [B, F, 17, 3, 1]

    # Apply rotation
    rotated_pose = torch.matmul(rotation_matrices.unsqueeze(2), input_3d_pose_expanded)  # Shape: [B, F, 17, 3, 1]
    rotated_pose = rotated_pose.squeeze(-1)  # Shape: [B, F, 17, 3]

    return rotated_pose