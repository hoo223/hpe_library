from hpe_library.lib_import import *

def canonicalization_cam_3d(cam_3d, canonical_type):
    if len(cam_3d.shape) == 2:
        if 'same_z' in canonical_type:       dist = cam_3d[0, 2] # z value of pelvis joint for each frame
        elif 'same_dist' in canonical_type:  dist = np.linalg.norm(cam_3d[0], axis=1) # dist from origin to pelvis joint for each frame
        elif 'fixed_dist' in canonical_type: dist = np.array([float(canonical_type.split('_')[-1])]*len(cam_3d))
        #elif 'revolute' in canonical_type: dist = np.linalg.norm(cam_3d[0], axis=1) # dist from origin to pelvis joint for each frame
        else: raise ValueError(f'canonical type {canonical_type} not found')
        cam_3d_canonical = cam_3d.copy() - cam_3d[0, None] # move to cam origin
        cam_3d_canonical[..., 2] += dist[None]
    elif len(cam_3d.shape) == 3:
        cam_3d_canonical = cam_3d.copy() - cam_3d[:, 0, None] # move to cam origin
        if 'same_z' in canonical_type:       dist = cam_3d[:, 0, 2] # z value of pelvis joint for each frame
        elif canonical_type == 'same_dist':  dist = np.linalg.norm(cam_3d[:, 0], axis=1) # dist from origin to pelvis joint for each frame
        elif 'fixed_dist' in canonical_type: dist = np.array([float(canonical_type.split('_')[-1])]*len(cam_3d))
        elif 'revolute_no_Rz' in canonical_type:
            dist = np.linalg.norm(cam_3d[:, 0], axis=1)
            # v_origin_to_pelvis = cam_3d[:, 0] / dist[:, None]
            # v_origin_to_pelvis_proj_on_xz = v_origin_to_pelvis.copy()
            # v_origin_to_pelvis_proj_on_xz[:, 1] = 0
            # v_origin_to_principle = np.array([0, 0, 1]).reshape(1, 3).repeat(len(cam_3d), axis=0)
            # assert v_origin_to_principle.shape == v_origin_to_pelvis.shape, (v_origin_to_principle.shape, v_origin_to_pelvis.shape)
            # R1 = batch_rotation_matrix_from_vectors(v_origin_to_pelvis, v_origin_to_pelvis_proj_on_xz)
            # R2 = batch_rotation_matrix_from_vectors(v_origin_to_pelvis_proj_on_xz, v_origin_to_principle)
            # R_real2virt_from_3d = R2 @ R1
            # R_real2virt_from_3d_inv = np.linalg.inv(R_real2virt_from_3d)
            R_real2virt_from_3d, R_real2virt_from_3d_inv = get_batch_R_real2virt_from_3d(cam_3d, no_Rz=True)
            cam_3d_canonical = np.einsum('ijk,ikl->ijl', cam_3d_canonical, R_real2virt_from_3d_inv)
        elif 'revolute' == canonical_type:
            dist = np.linalg.norm(cam_3d[:, 0], axis=1)
            # v_origin_to_pelvis = cam_3d[:, 0] / dist[:, None]
            # v_origin_to_principle = np.array([0, 0, 1]).reshape(1, 3).repeat(len(cam_3d), axis=0)
            # R_pelvis_to_principle = batch_rotation_matrix_from_vectors(v_origin_to_pelvis, v_origin_to_principle)
            # R_pelvis_to_principle_inv = np.linalg.inv(R_pelvis_to_principle)
            # assert v_origin_to_principle.shape == v_origin_to_pelvis.shape, (v_origin_to_principle.shape, v_origin_to_pelvis.shape)
            R_real2virt_from_3d, R_real2virt_from_3d_inv = get_batch_R_real2virt_from_3d(cam_3d)
            cam_3d_canonical = np.einsum('ijk,ikl->ijl', cam_3d_canonical, R_real2virt_from_3d_inv)
        else: raise ValueError(f'canonical type {canonical_type} not found')
        cam_3d_canonical[..., 2] += dist[:, None]
    else:
        raise ValueError(f'cam_3d shape {cam_3d.shape} not supported')
    return cam_3d_canonical

def genertate_pcl_img_2d(img_2d, cam_param, no_Rz=True):
    from hpe_library.my_utils.dh import projection
    K = cam_param['intrinsic']
    K_inv = np.linalg.inv(K)
    norm_2d = img_2d.copy() # np.stack([img_2d, np.ones([img_2d.shape[0], img_2d.shape[1], 1])])
    norm_2d = np.concatenate([norm_2d, np.ones((norm_2d.shape[0], norm_2d.shape[1], 1))], axis=-1)
    norm_2d = norm_2d @ K_inv.T

    locations = img_2d[:, 0]
    locations = np.hstack([locations, np.ones((locations.shape[0], 1))]) # to homogeneous coordinates
    locations = locations @ K_inv.T
    if no_Rz:
        R_virt2real = batch_virtualCameraRotationFromPosition(locations)
        # R_real2virts = np.linalg.inv(R_virt2reals)
        R_real2virt_inv = R_virt2real
    else:
        R_real2virt, R_real2virt_inv = get_batch_R_real2virt_from_3d(norm_2d, no_Rz=False)

    norm_2d_virt = np.einsum('ijk,ikl->ijl', norm_2d, R_real2virt_inv)
    img_2d_pcl = projection(norm_2d_virt, K)
    return img_2d_pcl

# Function to compute rotation matrices from 2D pose batch input
def batch_virtualCameraRotationFromBatchInput(input_2d_p, inverse=False):
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
    if inverse:
        R = R.transpose(-2, -1)

    return R

def batch_rotation_matrix_from_vectors_torch(vec1, vec2):
    """
    Returns the rotation matrices that align vec1 to vec2 for a batch of vectors using torch tensors.
    :param vec1: A batch of 3d "source" vectors. Shape (batch, frame, 3)
    :param vec2: A batch of 3d "destination" vectors. Shape (batch, frame, 3)
    :return: A batch of transformation matrices (batch, frame, 3, 3) which when applied to vec1, aligns it with vec2.
    """
    # Normalize the input vectors
    a = vec1 / torch.norm(vec1, dim=2, keepdim=True)
    b = vec2 / torch.norm(vec2, dim=2, keepdim=True)
    v = torch.cross(a, b, dim=2)
    c = torch.einsum('bij,bij->bi', a, b)
    s = torch.norm(v, dim=2)

    # Compute the skew-symmetric cross-product matrices of v
    kmat = torch.zeros((vec1.shape[0], vec1.shape[1], 3, 3), device=vec1.device)
    kmat[:, :, 0, 1] = -v[:, :, 2]
    kmat[:, :, 0, 2] = v[:, :, 1]
    kmat[:, :, 1, 0] = v[:, :, 2]
    kmat[:, :, 1, 2] = -v[:, :, 0]
    kmat[:, :, 2, 0] = -v[:, :, 1]
    kmat[:, :, 2, 1] = v[:, :, 0]

    # Compute the rotation matrices
    eye = torch.eye(3, device=vec1.device).reshape(1, 1, 3, 3)
    rotation_matrices = eye + kmat + torch.einsum('bijk,bikl->bijl', kmat, kmat) * ((1 - c) / (s ** 2)).unsqueeze(-1).unsqueeze(-1)

    return rotation_matrices

def batch_inverse_rotation_matrices(rotation_matrices):
    """
    Returns the inverse of a batch of rotation matrices.
    :param rotation_matrices: A batch of rotation matrices. Shape (Batch, Frame, 3, 3)
    :return: A batch of inverse rotation matrices. Shape (Batch, Frame, 3, 3)
    """
    return torch.linalg.inv(rotation_matrices)

def batch_rotation_matrix_from_vectors(vec1, vec2):
    """
    Returns the rotation matrices that align vec1 to vec2 for a batch of vectors.
    :param vec1: A batch of 3d "source" vectors. Shape (N, 3)
    :param vec2: A batch of 3d "destination" vectors. Shape (N, 3)
    :return: A batch of transformation matrices (N, 3, 3) which when applied to vec1, aligns it with vec2.
    """
    # Normalize the input vectors
    a = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
    b = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
    v = np.cross(a, b)
    c = np.einsum('ij,ij->i', a, b)
    s = np.linalg.norm(v, axis=1)
    # check if v is zero (i.e. vec1 and vec2 are parallel)
    wherezero = np.where(s == 0)
    if len(wherezero[0]) > 0:
        #print(wherezero)
        #print(v[wherezero], s[wherezero])
        s[wherezero] = 1

    # Compute the skew-symmetric cross-product matrices of v
    kmat = np.zeros((vec1.shape[0], 3, 3))
    kmat[:, 0, 1] = -v[:, 2]
    kmat[:, 0, 2] = v[:, 1]
    kmat[:, 1, 0] = v[:, 2]
    kmat[:, 1, 2] = -v[:, 0]
    kmat[:, 2, 0] = -v[:, 1]
    kmat[:, 2, 1] = v[:, 0]

    # Compute the rotation matrices
    eye = np.eye(3).reshape(1, 3, 3)
    rotation_matrices = eye + kmat + np.einsum('ijk,ikl->ijl', kmat, kmat) * ((1 - c) / (s ** 2))[:, None, None]

    # check if v is zero (i.e. vec1 and vec2 are parallel)
    if len(wherezero[0]) > 0:
        rotation_matrices[wherezero] = np.eye(3)
        #print(rotation_matrices[wherezero])

    return rotation_matrices

def batch_virtualCameraRotationFromPosition(positions):
    """
    Returns the rotation matrices for a batch of positions.
    :param positions: A batch of 3d positions. Shape (N, 3)
    :return: A batch of rotation matrices (N, 3, 3)
    """
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    n1x = np.sqrt(1 + x ** 2)
    d1x = 1 / n1x
    d1xy = 1 / np.sqrt(1 + x ** 2 + y ** 2)
    d1xy1x = 1 / np.sqrt((1 + x ** 2 + y ** 2) * (1 + x ** 2))

    R_virt2orig = np.zeros((positions.shape[0], 3, 3))
    R_virt2orig[:, 0, 0] = d1x
    R_virt2orig[:, 0, 1] = -x * y * d1xy1x
    R_virt2orig[:, 0, 2] = x * d1xy
    R_virt2orig[:, 1, 1] = n1x * d1xy
    R_virt2orig[:, 1, 2] = y * d1xy
    R_virt2orig[:, 2, 0] = -x * d1x
    R_virt2orig[:, 2, 1] = -y * d1xy1x
    R_virt2orig[:, 2, 2] = d1xy

    return R_virt2orig

def get_batch_R_orig2virt_from_2d(img_2d, K):
    # K: intrinsic matrix
    assert len(img_2d.shape) == 3, img_2d.shape
    assert img_2d.shape[-1] == 2, img_2d.shape
    locations = img_2d[:, 0]
    locations = np.hstack([locations, np.ones((locations.shape[0], 1))]) # to homogeneous coordinates
    locations = locations @ np.linalg.inv(K).T
    R_virt2orig_from_2d = batch_virtualCameraRotationFromPosition(locations)
    R_orig2virt_from_2d = np.linalg.inv(R_virt2orig_from_2d)
    R_orig2virt_from_2d_inv = R_virt2orig_from_2d
    return R_orig2virt_from_2d, R_orig2virt_from_2d_inv

def get_batch_R_orig2virt_from_3d(cam_3d, no_Rz=False):
    from hpe_library.my_utils.test_utils import rotation_matrix_from_vectors
    if len(cam_3d.shape) == 2: # single frame
        assert len(cam_3d.shape) == 2, cam_3d.shape
        assert cam_3d.shape[-1] == 3, cam_3d.shape
        dist = np.linalg.norm(cam_3d[0])
        v_origin_to_pelvis = cam_3d[0] / dist
        v_origin_to_principle = np.array([0, 0, 1])
        assert v_origin_to_principle.shape == v_origin_to_pelvis.shape, (v_origin_to_principle.shape, v_origin_to_pelvis.shape)
        if no_Rz:
            v_origin_to_pelvis_proj_on_xz = v_origin_to_pelvis.copy()
            v_origin_to_pelvis_proj_on_xz[1] = 0
            R1 = rotation_matrix_from_vectors(v_origin_to_pelvis, v_origin_to_pelvis_proj_on_xz)
            R2 = rotation_matrix_from_vectors(v_origin_to_pelvis_proj_on_xz, v_origin_to_principle)
            R_orig2virt_from_3d = R2 @ R1
        else:
            R_orig2virt_from_3d = rotation_matrix_from_vectors(v_origin_to_pelvis, v_origin_to_principle)
    elif len(cam_3d.shape) == 3: # multiple frames
        assert len(cam_3d.shape) == 3, cam_3d.shape
        assert cam_3d.shape[-1] == 3, cam_3d.shape
        dist = np.linalg.norm(cam_3d[:, 0], axis=1)
        v_origin_to_pelvis = cam_3d[:, 0] / dist[:, None]
        v_origin_to_principle = np.array([0, 0, 1]).reshape(1, 3).repeat(len(cam_3d), axis=0)
        assert v_origin_to_principle.shape == v_origin_to_pelvis.shape, (v_origin_to_principle.shape, v_origin_to_pelvis.shape)
        if no_Rz:
            v_origin_to_pelvis_proj_on_xz = v_origin_to_pelvis.copy()
            v_origin_to_pelvis_proj_on_xz[:, 1] = 0
            R1 = batch_rotation_matrix_from_vectors(v_origin_to_pelvis, v_origin_to_pelvis_proj_on_xz)
            R2 = batch_rotation_matrix_from_vectors(v_origin_to_pelvis_proj_on_xz, v_origin_to_principle)
            R_orig2virt_from_3d = R2 @ R1
        else:
            R_orig2virt_from_3d = batch_rotation_matrix_from_vectors(v_origin_to_pelvis, v_origin_to_principle)
    elif len(cam_3d.shape) == 4: # batches of multiple frame
        assert len(cam_3d.shape) == 4, cam_3d.shape
        assert cam_3d.shape[-1] == 3, cam_3d.shape
        b, f, j, c = cam_3d.shape
        cam_3d = cam_3d.reshape(b*f, j, c)
        dist = np.linalg.norm(cam_3d[:, 0], axis=1)
        v_origin_to_pelvis = cam_3d[:, 0] / dist[:, None]
        v_origin_to_principle = np.array([0, 0, 1]).reshape(1, 3).repeat(len(cam_3d), axis=0)
        assert v_origin_to_principle.shape == v_origin_to_pelvis.shape, (v_origin_to_principle.shape, v_origin_to_pelvis.shape)
        if no_Rz:
            v_origin_to_pelvis_proj_on_xz = v_origin_to_pelvis.copy()
            v_origin_to_pelvis_proj_on_xz[:, 1] = 0
            R1 = batch_rotation_matrix_from_vectors(v_origin_to_pelvis, v_origin_to_pelvis_proj_on_xz)
            R2 = batch_rotation_matrix_from_vectors(v_origin_to_pelvis_proj_on_xz, v_origin_to_principle)
            R_orig2virt_from_3d = R2 @ R1
        else:
            R_orig2virt_from_3d = batch_rotation_matrix_from_vectors(v_origin_to_pelvis, v_origin_to_principle)
        R_orig2virt_from_3d = R_orig2virt_from_3d.reshape(b, f, 3, 3)
    else:
        raise ValueError(cam_3d.shape)
    R_orig2virt_from_3d_inv = np.linalg.inv(R_orig2virt_from_3d)
    return R_orig2virt_from_3d, R_orig2virt_from_3d_inv