from lib_import import *
from my_utils import *
from .test_utils import get_h36m_keypoint_index

def generate_world_frame():
    world_origin = np.zeros(3)
    #print(world_origin.shape)
    dx, dy, dz = np.eye(3)

    world_frame = ReferenceFrame(
        origin=world_origin, 
        dx=dx, 
        dy=dy,
        dz=dz,
        name="World",
    )
    return world_frame

def generate_camera_frame(cam_ext, mm_to_m=True, name='camera'):
    R = np.array(cam_ext['R']) #Rotation.from_quat(cam['orientation']).as_matrix()
    R_C = R.transpose()
    if mm_to_m:
        t = np.array(cam_ext['t'])[:, 0]*0.001
    else:
        t = np.array(cam_ext['t'])[:, 0]
    C = -R_C @ t
    dx_c, dy_c, dz_c = R
    cam_frame = ReferenceFrame(
        origin=C,
        dx=dx_c, 
        dy=dy_c,
        dz=dz_c,
        name="{}".format(name),
    )
    return cam_frame

def projection(pose_3d, proj_mat):
    if len(pose_3d.shape) == 2:
        homo = np.insert(pose_3d, 3, 1, axis=1) # to homogeneous coordinates
        projected = (proj_mat @ homo.T).T
        pose_2d = projected / projected[:, 2].repeat(3).reshape(-1, 3)
        return pose_2d
    elif len(pose_3d.shape) == 3:
        pose_2ds = []
        for pose in pose_3d:
            homo = np.insert(pose, 3, 1, axis=1) # to homogeneous coordinates
            projected = (proj_mat @ homo.T).T
            pose_2d = projected / projected[:, 2].repeat(3).reshape(-1, 3)
            pose_2ds.append(pose_2d)
        return np.array(pose_2ds)
    
def batch_projection(batch_pose_3d, batch_proj_mat):
    if len(batch_pose_3d.shape) == 4:
        homo = torch.cat((batch_pose_3d, torch.ones(batch_pose_3d.shape[0], batch_pose_3d.shape[1], batch_pose_3d.shape[2], 1).to(batch_pose_3d.device)), dim=-1)
    elif len(batch_pose_3d.shape) == 3:
        homo = torch.cat((batch_pose_3d, torch.ones(batch_pose_3d.shape[0], batch_pose_3d.shape[1], 1).to(batch_pose_3d.device)), dim=-1)
    batch_pose_projected = homo @ batch_proj_mat.transpose(-2, -1)
    batch_pose_projected = batch_pose_projected / batch_pose_projected[:, :, :, 2].unsqueeze(-1)
    return batch_pose_projected
    
def project_batch_tensor(pose_3d_batch, proj_mat_tensor, device='cuda'):
    homo = torch.cat((pose_3d_batch, torch.ones(pose_3d_batch.shape[0], pose_3d_batch.shape[1], 1).to(device)), dim=2)
    projected_batch = homo @ proj_mat_tensor.T
    projected_batch /= projected_batch[:, :, 2].reshape(-1, 5, 1).repeat(1, 1, 3)
    return projected_batch

def get_torso_direction(torso):
    if torso.shape[0] == 17:
        L_Hip = torso[4]
        R_Hip = torso[1]
        L_Shoulder = torso[11]
    elif torso.shape[0] == 5:
        L_Shoulder = torso[2]
        L_Hip = torso[1]
        R_Hip = torso[4]
    
    vec1 = L_Hip - R_Hip #  L_Hip - R_Hip
    vec1 /= np.linalg.norm(vec1)
    vec2 = L_Shoulder - L_Hip # L_Shoulder - L_Hip
    pelvis_normal = np.cross(vec1, vec2)
    pelvis_normal /= np.linalg.norm(pelvis_normal)

    return pelvis_normal

def get_torso_shape(torso):
    gt_length = []
    gt_angle = []
    for i in range(5):
        i_prev = i-1
        i_next = i+1
        if i_next > 4:
            i_next = 0
        #print(i_prev, i, i_next)

        p0 = torso[i_prev]
        p1 = torso[i]
        p2 = torso[i_next]

        l = np.linalg.norm(p1-p2)
        gt_length.append(l)

        v_prev = p0 - p1
        v_next = p2 - p1
        a = np.dot(v_prev, v_next) / (np.linalg.norm(v_prev) * np.linalg.norm(v_next))
        gt_angle.append(a)

    return gt_length, gt_angle

def draw_torso_direction(ax, torso):
    pelvis_normal = get_torso_direction(torso)
    Pelvis = torso[0]
    draw3d_arrow(Pelvis, pelvis_normal/2, head_length=0.1, color="tab:red", ax = ax)

    return pelvis_normal
        


def generate_dh_frame(pos, R, name='dh_frame'):
    dh_frame = ReferenceFrame(
        origin=pos, 
        dx=R[0], 
        dy=R[1],
        dz=R[2],
        name=name,
    )
    return dh_frame

def draw_subline(ax, src, dst):
    # d = dst[2] - src[2]
    # point1 = dst - np.array([0, 0, d])
    # point2 = src + np.array([0, 0, d])
    # ax.plot(*np.c_[src, point1], color="tab:gray", ls='--' )
    # ax.plot(*np.c_[point1, dst], color="tab:gray", ls='--' )
    # ax.plot(*np.c_[point2, dst], color="tab:gray", ls='--' )
    ax.plot(*np.c_[src, dst], color="tab:gray", ls='--' )
    return ax

def draw_arm(ax, root, link1, link2):
    # upper arm
    ax.plot(*np.c_[root, link1], color="tab:gray", ls='--')

    # elbow
    ax.plot(link1[0], link1[1], link1[2], 'ob')

    # lower arm
    ax.plot(*np.c_[link1, link2], color="tab:gray", ls='--') 

    # wrist frame
    ax.plot(link2[0], link2[1], link2[2], 'or')

## 두 벡터가 같을 때 nan 발생
# def rotation_matrix_from_vectors(vec1, vec2): # from vec1 to vec2
#     # Normalize the vectors
#     vec1 = vec1 / np.linalg.norm(vec1)
#     vec2 = vec2 / np.linalg.norm(vec2)

#     # Calculate the cross product and dot product
#     cross_product = np.cross(vec1, vec2)
#     dot_product = np.dot(vec1, vec2)

#     # Skew-symmetric matrix of cross product
#     # cross_matrix = np.array([[0, -cross_product[2], cross_product[1]],
#     #                          [cross_product[2], 0, -cross_product[0]],
#     #                          [-cross_product[1], cross_product[0], 0]])
#     cross_matrix = skew_symmetric_matrix(cross_product)

#     # Rotation matrix
#     rotation_matrix = np.eye(3) + cross_matrix + np.dot(cross_matrix, cross_matrix) * (1 - dot_product) / np.linalg.norm(cross_product)**2

#     return rotation_matrix

# rotation_matrix_from_vectors와 동일
def rotation_matrix_to_vector_align(from_v, to_v):
    # align from_v to to_v
    # https://gist.github.com/kevinmoran/b45980723e53edeb8a5a43c49f134724#:~:text=final%20rotateAlign()%20function%3A-,mat3%20rotateAlign(%20vec3%20v1%2C%20vec3%20v2)%0A%7B%0A%20%20%20%20vec3%20axis%20%3D%20cross(%20v1,%2C%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20(axis.z%20*%20axis.z%20*%20k)%20%2B%20cosA%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20)%3B%0A%0A%20%20%20%20return%20result%3B%0A%7D,-And%20there%20we
    axis = np.cross(from_v, to_v)
    cosA = np.dot(from_v, to_v)
    k = 1.0 / (1.0 + cosA)
    result = np.array([[axis[0] * axis[0] * k + cosA, axis[1] * axis[0] * k - axis[2], axis[2] * axis[0] * k + axis[1]],
                       [axis[0] * axis[1] * k + axis[2], axis[1] * axis[1] * k + cosA, axis[2] * axis[1] * k - axis[0]],
                       [axis[0] * axis[2] * k - axis[1], axis[1] * axis[2] * k + axis[0], axis[2] * axis[2] * k + cosA]])
    return result

def get_torso_rotation_matrix(torso):
    if torso.shape[0] == 17:
        pelvis = torso[0]
        l_hip = torso[4]
        r_hip = torso[1]
        l_shoulder = torso[11]
        r_shoulder = torso[14]
    elif torso.shape[0] == 5:
        pelvis = torso[0]
        l_hip = torso[1]
        l_shoulder = torso[2]
        r_shoulder = torso[3]
        r_hip = torso[4]
        
    left = l_hip - r_hip
    left /= np.linalg.norm(left)
    vec = l_shoulder - l_hip
    forward = np.cross(left, vec)
    forward /= np.linalg.norm(forward)
    up = np.cross(forward, left)

    rot_from_world = np.array([forward, left, up]).T

    return rot_from_world

def rotation_matrix_torso2torso(torso1, torso2):
    src_R = get_torso_rotation_matrix(torso1)
    tar_R = get_torso_rotation_matrix(torso2)
    align_R = tar_R @ src_R.T
    return align_R, src_R, tar_R

def normalize_vector(v):
    return v / np.linalg.norm(v)

def calculate_rotation_quaternion(vec1, vec2):
    v1 = normalize_vector(vec1)
    v2 = normalize_vector(vec2)
    
    n = np.cross(v1, v2)
    theta = np.arccos(np.dot(v1, v2))
    
    if np.isclose(theta, 0):
        # If the vectors are nearly parallel, return an identity quaternion
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    q = np.array([np.cos(theta/2), np.sin(theta/2) * n[0], np.sin(theta/2) * n[1], np.sin(theta/2) * n[2]])
    return q

# def quaternion_from_vectors(vec1, vec2):
#     # Ensure vectors are normalized
#     vec1 = normalize(vec1)
#     vec2 = normalize(vec2)

#     # Calculate the rotation axis
#     axis = np.cross(vec1, vec2)
#     angle = np.arccos(np.dot(vec1, vec2))

#     # Convert axis-angle representation to quaternion
#     half_angle = 0.5 * angle
#     sin_half_angle = np.sin(half_angle)
#     return np.array([np.cos(half_angle), sin_half_angle * axis[0], sin_half_angle * axis[1], sin_half_angle * axis[2]])

# def rotate_vector_with_quaternion(quaternion, vec):
#     # Ensure quaternion is normalized
#     quaternion = normalize(quaternion)

#     # Convert quaternion to rotation matrix
#     w, x, y, z = quaternion
#     rotation_matrix = np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
#                                 [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
#                                 [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])

#     # Apply rotation to vector
#     return np.dot(rotation_matrix, vec)

def rotate_torso_by_R(torso, R):
    if len(torso.shape) == 2: # single pose
        root = torso[0]
        root_rel = torso - root
        return (R @ root_rel.T).T + root
    elif len(torso.shape) == 3: # multiple poses
        rotated = torso.copy()
        for i in range(torso.shape[0]):
            root = torso[i, 0]
            root_rel = torso[i] - root
            rotated[i] = (R @ root_rel.T).T + root
        return rotated

def rotate_torso_by_R_for_batch_tensor(torso_batch, R, device='cuda'):
    # input
        # torso: [batch, 5, 3]
        # R: [batch, 3, 3]
    # output
        # rotated_torso: [batch, 5, 3]
    root_batch = torso_batch[:, 0, :].reshape(-1, 1, 3).repeat(1, 5, 1)
    root_rel_batch = torso_batch - root_batch

    # bmm -> batch matrix multiplication 
    # https://kh-kim.github.io/nlp_with_deep_learning_blog/docs/1-04-linear_layer/02-matmul_exercise/ 
    return torch.bmm(R, root_rel_batch.transpose(1, 2)).transpose(1, 2) + root_batch

# New DH matrix convention for appendage (param a removed)
# Rot_z_theta -> Rot_y_minus_alpha -> Trans_x_d
def DH_matrix(theta, alpha, d):
    dh_matrix = [
        [cos(theta)*cos(alpha), -sin(theta), -cos(theta)*sin(alpha), d*cos(alpha)*cos(theta)],
        [sin(theta)*cos(alpha), cos(theta), -sin(theta)*sin(alpha), d*sin(theta)*cos(alpha)],
        [sin(alpha), 0, cos(alpha), d*sin(alpha)],
        [0, 0, 0, 1]
    ]
    # Rot_z_theta = np.eye(4)
    # Rot_z_theta[:3, :3] = Rotation.from_rotvec(np.array([0, 0, theta])).as_matrix()
    # Rot_y_alpha = np.eye(4)
    # Rot_y_alpha[:3, :3] = Rotation.from_rotvec(np.array([0, -alpha, 0])).as_matrix()
    # Trans_x_d = np.eye(4)
    # Trans_x_d[0, 3] = d
    # #dh_matrix = Trans_x_d @ Rot_y_alpha @ Rot_z_theta
    # dh_matrix = Rot_z_theta @ Rot_y_alpha @ Trans_x_d
    return np.array(dh_matrix)

def generate_two_link(link1_yaw, link1_pitch, link2_yaw, link2_pitch, l1, l2, root_tf=np.eye(4), degrees=True):
    # input: yaw, pitch, length of link1, link2, root transformation matrix
    # output: origins of link1, link2, terminal, and frames of link1, link2, terminal
    
    if degrees:
        link1_yaw = radians(link1_yaw)
        link1_pitch = radians(link1_pitch)
        link2_yaw = radians(link2_yaw)
        link2_pitch = radians(link2_pitch)
    
    m01 = DH_matrix(theta=link1_yaw, alpha=link1_pitch, d=l1)
    m12 = DH_matrix(theta=link2_yaw, alpha=link2_pitch, d=l2)
    m02 = m01 @ m12
    link1_tf = root_tf @ m01
    link2_tf = root_tf @ m02
    
    # root
    root_origin = root_tf[:3, 3]
    root_R = root_tf[:3, :3]
    root_frame = generate_dh_frame(root_origin, root_R.transpose(), 'root')
    
    # link1
    link1_origin = root_tf[:3, 3]
    link1_R = link1_tf[:3, :3]
    link1_frame = generate_dh_frame(link1_origin, link1_R.transpose(), 'link1')

    # link2
    link2_origin = link1_tf[:3, 3]
    link2_R = link2_tf[:3, :3]
    link2_frame = generate_dh_frame(link2_origin, link2_R.transpose(), 'link2')
    
    # terminal
    terminal_origin = link2_tf[:3, 3]

    return link1_origin, link2_origin, terminal_origin, root_R, link1_R, link2_R, root_frame, link1_frame, link2_frame

def calculate_azimuth_elevation(vector, root_R, degrees=False):
    x, y, z = root_R.T @ vector
    azimuth = math.atan2(y, x)
    elevation = math.atan2(z, math.sqrt(x**2 + y**2))
    if degrees:
        return math.degrees(azimuth), math.degrees(elevation)
    else:
        return azimuth, elevation  # Converting to degrees for readability
    
def azim_elev_to_vec(azim, elev, magnitude=1, origin=[0, 0, 0], degrees=False):
    if degrees:
        azim = math.radians(azim)
        elev = math.radians(elev)
    x = magnitude * np.cos(azim) * np.cos(elev) + origin[0]
    y = magnitude * np.sin(azim) * np.cos(elev) + origin[1]
    z = magnitude * np.sin(elev) + origin[2]
    return np.array([x,y,z])

def batch_azim_elev_to_vec(batch_azim, batch_elev, batch_magnitude, batch_origin, degrees=False):
    if degrees:
        batch_azim = torch.deg2rad(batch_azim)
        batch_elev = torch.deg2rad(batch_elev)
    batch_x = batch_magnitude * torch.cos(batch_azim) * torch.cos(batch_elev)
    batch_y = batch_magnitude * torch.sin(batch_azim) * torch.cos(batch_elev)
    batch_z = batch_magnitude * torch.sin(batch_elev)
    batch_cam_origin = torch.cat([batch_x, batch_y, batch_z], dim=-1).to(batch_origin.device)
    batch_cam_origin = batch_cam_origin + batch_origin
    return batch_cam_origin
    
class Appendage:
    def __init__(self, link1_length, link2_length, link1_yaw_init, link1_pitch_init, link2_yaw_init, link2_pitch_init, degree=True, root_tf=np.eye(4)):
        self.link1_length = link1_length
        self.link2_length = link2_length
        self.link1_yaw = link1_yaw_init
        self.link1_pitch = link1_pitch_init
        self.link2_yaw = link2_yaw_init
        self.link2_pitch = link2_pitch_init
        self.root_tf = root_tf
        self.root_origin = self.root_tf[:3, 3]
        self.root_R = self.root_tf[:3, :3]
        self.root_frame = self.generate_vis_frame(self.root_tf[:3, 3], self.root_tf[:3, :3].transpose(), name='root')

        self.update_link(self.link1_yaw, self.link1_pitch, self.link2_yaw, self.link2_pitch, degree=degree)

    def DH_matrix(self, theta, alpha, d):
        dh_matrix = [
            [cos(theta)*cos(alpha), -sin(theta), -cos(theta)*sin(alpha), d*cos(alpha)*cos(theta)],
            [sin(theta)*cos(alpha), cos(theta), -sin(theta)*sin(alpha), d*sin(theta)*cos(alpha)],
            [sin(alpha), 0, cos(alpha), d*sin(alpha)],
            [0, 0, 0, 1]
        ]
        return np.array(dh_matrix)
    
    def build_dh_frame(self, yaw, pitch, d, parent_tf, degree=False, name=''): # yaw = theta, pitch = alpha
        if degree:
            yaw = radians(yaw)
            pitch = radians(pitch)
        dh_matrix = self.DH_matrix(theta=yaw, alpha=pitch, d=d)
        child_tf = parent_tf @ dh_matrix
        parent_origin = parent_tf[:3, 3]
        child_rot = child_tf[:3, :3]
        child_frame = self.generate_vis_frame(parent_origin, child_rot.transpose(), name)
        return child_tf, child_frame, dh_matrix
    
    def generate_vis_frame(self, pos, R, name='dh_frame'):
        dh_frame = ReferenceFrame(
            origin=pos, 
            dx=R[0], 
            dy=R[1],
            dz=R[2],
            name=name,
        )
        return dh_frame
    
    def update_link(self, link1_yaw=None, link1_pitch=None, link2_yaw=None, link2_pitch=None, degree=False):
        # update angles
        if link1_yaw != None:
            self.link1_yaw = link1_yaw
        if link1_pitch != None:
            self.link1_pitch = link1_pitch
        if link2_yaw != None:
            self.link2_yaw = link2_yaw
        if link2_pitch != None:
            self.link2_pitch = link2_pitch
            
        # link1
        self.link1_tf, self.link1_frame, self.link1_dh_mat = self.build_dh_frame(self.link1_yaw, self.link1_pitch, self.link1_length, self.root_tf, degree=degree, name='link1')
        self.link1_origin = self.root_tf[:3, 3]
        self.link1_R = self.link1_tf[:3, :3]
        
        # link2
        self.link2_tf, self.link2_frame, self.link2_dh_mat = self.build_dh_frame(self.link2_yaw, self.link2_pitch, self.link2_length, self.link1_tf, degree=degree, name='link2')
        self.link2_origin = self.link1_tf[:3, 3]
        self.link2_R = self.link2_tf[:3, :3]

        # terminal
        #self.terminal_tf, self.terminal_frame, self.terminal_dh_mat = self.build_dh_frame(0, 0, 0, self.link2_tf, degree=degree, name='terminal')
        self.terminal_origin = self.link2_tf[:3, 3]
        
        # vector
        self.link1_vec = self.link2_origin - self.link1_origin
        self.link2_vec = self.terminal_origin - self.link2_origin
    
    def draw(self, ax, draw_frame=False, head_length=0.01, scale=0.1, fontsize=10, show_name=False, show_axis=False):
        #plt.sca(ax)
        ax.plot(self.link1_origin[0], self.link1_origin[1], self.link1_origin[2],  '.k') # link1 origin        
        ax.plot(*np.c_[self.link1_origin, self.link2_origin], color="tab:gray", ls='--') # link1
        ax.plot(self.link2_origin[0], self.link2_origin[1], self.link2_origin[2], '.b') # link2 origin
        ax.plot(*np.c_[self.link2_origin, self.terminal_origin], color="tab:gray", ls='--') # link2 arm
        ax.plot(self.terminal_origin[0], self.terminal_origin[1], self.terminal_origin[2], '.r') # terminal origin
        if draw_frame:
            self.root_frame.draw3d(color='tab:blue', head_length=head_length, scale=scale, fontsize=fontsize, show_name=show_name, show_axis=show_axis) # root frame
            self.link1_frame.draw3d(color='tab:red', head_length=head_length, scale=scale, fontsize=fontsize, show_name=show_name, show_axis=show_axis) # link1 frame
            self.link2_frame.draw3d(color='tab:red', head_length=head_length, scale=scale, fontsize=fontsize, show_name=show_name, show_axis=show_axis) # link2 frame
            

class DHModel:
    def __init__(self, init_pose_3d) -> None:
        # rotation matrices
        self.left_init_R = Rotation.from_rotvec(-np.pi/2 * np.array([0, 0, 1])).as_matrix() # rotate -90 deg wrt z-axis
        self.right_init_R = Rotation.from_rotvec(np.pi/2 * np.array([0, 0, 1])).as_matrix() # rotate -90 deg wrt z-axis
        
        # appendage id
        self.head_id = 0
        self.right_arm_id = 1
        self.left_arm_id = 2
        self.right_leg_id = 3
        self.left_leg_id = 4
        
        # set keypoints
        self.set_keypoints_from_pose(init_pose_3d)
        # extract vectors wrt world frame
        self.set_limb_vectors()
        # set lengths
        self.set_limb_length() # only for init_pose_3d
        # set dh model
        self.set_dh_model_from_pose(init_pose_3d)
        
    def set_keypoints_from_pose(self, pose):
        # get head points
        self.head_origin = pose[10]
        self.nose_origin = pose[9]
        
        # get torso points
        self.pelvis_origin = pose[0]
        self.r_hip_origin = pose[1]
        self.l_hip_origin = pose[4]
        self.torso_origin = pose[7]
        self.neck_origin = pose[8]
        self.l_shoulder_origin = pose[11]
        self.r_shoulder_origin = pose[14]
        
        # get appendage points
        self.l_elbow_origin = pose[12]
        self.l_wrist_origin = pose[13]
        self.r_elbow_origin = pose[15]
        self.r_wrist_origin = pose[16]
        self.l_knee_origin  = pose[5]
        self.l_ankle_origin = pose[6]
        self.r_knee_origin  = pose[2]
        self.r_ankle_origin = pose[3]
        
    def set_limb_vectors(self):
        self.neck_to_nose_vector    = self.nose_origin    - self.neck_origin
        self.nose_to_head_vector    = self.head_origin    - self.nose_origin
        self.left_upper_arm_vector  = self.l_elbow_origin - self.l_shoulder_origin
        self.left_lower_arm_vector  = self.l_wrist_origin - self.l_elbow_origin
        self.right_upper_arm_vector = self.r_elbow_origin - self.r_shoulder_origin
        self.right_lower_arm_vector = self.r_wrist_origin - self.r_elbow_origin
        self.left_upper_leg_vector  = self.l_knee_origin  - self.l_hip_origin
        self.left_lower_leg_vector  = self.l_ankle_origin - self.l_knee_origin
        self.right_upper_leg_vector = self.r_knee_origin  - self.r_hip_origin
        self.right_lower_leg_vector = self.r_ankle_origin - self.r_knee_origin
        
    def set_limb_length(self):
        self.neck_to_nose_length    = np.linalg.norm(self.neck_to_nose_vector)
        self.nose_to_head_length    = np.linalg.norm(self.nose_to_head_vector)
        self.left_upper_arm_length  = np.linalg.norm(self.left_upper_arm_vector ) # limb_lens[11]
        self.left_lower_arm_length  = np.linalg.norm(self.left_lower_arm_vector ) # limb_lens[12]
        self.right_upper_arm_length = np.linalg.norm(self.right_upper_arm_vector) # limb_lens[14]
        self.right_lower_arm_length = np.linalg.norm(self.right_lower_arm_vector) # limb_lens[15]
        self.left_upper_leg_length  = np.linalg.norm(self.left_upper_leg_vector )# limb_lens[4]
        self.left_lower_leg_length  = np.linalg.norm(self.left_lower_leg_vector )# limb_lens[5]
        self.right_upper_leg_length = np.linalg.norm(self.right_upper_leg_vector)# limb_lens[1]
        self.right_lower_leg_length = np.linalg.norm(self.right_lower_leg_vector)# limb_lens[2]
        self.limb_length = np.array([self.neck_to_nose_length, 
                                     self.nose_to_head_length, 
                                     self.right_upper_arm_length,
                                     self.right_lower_arm_length,
                                     self.left_upper_arm_length,
                                     self.left_lower_arm_length,
                                     self.right_upper_leg_length,
                                     self.right_lower_leg_length,
                                     self.left_upper_leg_length,
                                     self.left_lower_leg_length])
        
    def get_limb_length(self, by_dict=False):
        if by_dict:
            limb_length = {}
            limb_length['ra_l1_length']    = self.right_upper_arm_length
            limb_length['ra_l2_length']    = self.right_lower_arm_length
            limb_length['la_l1_length']    = self.left_upper_arm_length
            limb_length['la_l2_length']    = self.left_lower_arm_length
            limb_length['rl_l1_length']    = self.right_upper_leg_length
            limb_length['rl_l2_length']    = self.right_lower_leg_length
            limb_length['ll_l1_length']    = self.left_upper_leg_length
            limb_length['ll_l2_length']    = self.left_lower_leg_length
        else:
            limb_length = self.limb_length[2:]
        return limb_length
        
    def set_torso(self, batch_torso):
        self.batch_torso_origin      = batch_torso[0]
        self.batch_r_hip_origin      = batch_torso[1]
        self.batch_l_hip_origin      = batch_torso[2]
        self.batch_torso_origin      = batch_torso[3]
        self.batch_neck_origin       = batch_torso[4]
        self.batch_l_shoulder_origin = batch_torso[5]
        self.batch_r_shoulder_origin = batch_torso[6]
        
    def set_appendage_from_angles(self, angles, head=False):
        if head:
            self.head.update_link(angles['h_l1_yaw'], angles['h_l1_pitch'])
        self.right_arm.update_link(angles['ra_l1_yaw'], angles['ra_l1_pitch'], angles['ra_l2_yaw'], angles['ra_l2_pitch'])
        self.left_arm.update_link(angles['la_l1_yaw'], angles['la_l1_pitch'], angles['la_l2_yaw'], angles['la_l2_pitch'])
        self.right_leg.update_link(angles['rl_l1_yaw'], angles['rl_l1_pitch'], angles['rl_l2_yaw'], angles['rl_l2_pitch'])
        self.left_leg.update_link(angles['ll_l1_yaw'], angles['ll_l1_pitch'], angles['ll_l2_yaw'], angles['ll_l2_pitch'])
        
    def set_dh_model_from_pose(self, pose):
        # update keypoints
        self.set_keypoints_from_pose(pose)        
        # get body reference frame
        self.set_body_reference_frame()
        # extract vectors wrt world frame
        self.set_limb_vectors()
        ## generate appendages ---------------------------------------------------------------------
        self.head      = self.generate_appendage(self.head_id,      self.neck_origin,       self.neck_to_nose_length,    self.nose_to_head_length)
        self.right_arm = self.generate_appendage(self.right_arm_id, self.r_shoulder_origin, self.right_upper_arm_length, self.right_lower_arm_length)
        self.left_arm  = self.generate_appendage(self.left_arm_id,  self.l_shoulder_origin, self.left_upper_arm_length,  self.left_lower_arm_length)
        self.left_leg  = self.generate_appendage(self.left_leg_id,  self.l_hip_origin,      self.left_upper_leg_length,  self.left_lower_leg_length)
        self.right_leg = self.generate_appendage(self.right_leg_id, self.r_hip_origin,      self.right_upper_leg_length, self.right_lower_leg_length)
        # extract angles and update appendages
        self.extract_and_update_appendage_angles(self.head,      self.neck_to_nose_vector,    self.nose_to_head_vector)
        self.extract_and_update_appendage_angles(self.right_arm, self.right_upper_arm_vector, self.right_lower_arm_vector)
        self.extract_and_update_appendage_angles(self.left_arm,  self.left_upper_arm_vector,  self.left_lower_arm_vector)
        self.extract_and_update_appendage_angles(self.right_leg, self.right_upper_leg_vector, self.right_lower_leg_vector)
        self.extract_and_update_appendage_angles(self.left_leg,  self.left_upper_leg_vector,  self.left_lower_leg_vector)
        
    def set_body_reference_frame(self):
        self.z_axis = np.array([0, 0, 1])
        pelvis_to_l_hip = self.l_hip_origin - self.pelvis_origin
        pelvis_to_l_hip[2] = 0 # project to x-y plane
        self.y_axis = pelvis_to_l_hip / np.linalg.norm(pelvis_to_l_hip)
        self.x_axis = np.cross(self.y_axis, self.z_axis)
        self.body_R = np.c_[self.x_axis, self.y_axis, self.z_axis].T
        self.body_frame = ReferenceFrame(
            origin=self.pelvis_origin, 
            dx=self.body_R[0], 
            dy=self.body_R[1],
            dz=self.body_R[2],
            name='body',
        )
        
    def generate_appendage(self, appendage_id, root_origin, link1_length, link2_length):
        if appendage_id == self.head_id:
            R = self.body_R
        elif appendage_id in [self.right_arm_id, self.right_leg_id]:
            R = self.right_init_R @ self.body_R
        elif appendage_id in [self.left_arm_id, self.left_leg_id]:
            R = self.left_init_R @ self.body_R
            
        root_tf = np.eye(4)
        root_tf[:3, :3] = R
        root_tf[:3, 3] = root_origin
        return Appendage(link1_length, link2_length, 0, 0, 0, 0, degree=True, root_tf=root_tf)
    
    def extract_and_update_appendage_angles(self, appendage, link1_vector, link2_vector):
        # extract angles from predefined vectors (inverse kinematics), and then update appendage angles (forward kinematics)
        # link1
        link1_yaw, link1_pitch = self.get_dh_angle_from_pose_vector(link1_vector, appendage.root_tf)
        appendage.update_link(link1_yaw, link1_pitch, 0, 0)
        # link2
        link2_yaw, link2_pitch = self.get_dh_angle_from_pose_vector(link2_vector, appendage.link1_tf)
        appendage.update_link(link1_yaw, link1_pitch, link2_yaw, link2_pitch)
    
    def get_dh_angle_from_pose_vector(self, vec, root_tf):
        return self.calculate_azimuth_elevation(vec, root_tf[:3, :3]) # yaw, pitch
    
    def calculate_azimuth_elevation(self, vector, root_R, degrees=False):
        x, y, z = root_R.T @ vector
        azimuth = math.atan2(y, x)
        elevation = math.atan2(z, math.sqrt(x**2 + y**2))
        if degrees:
            return math.degrees(azimuth), math.degrees(elevation)
        else:
            return azimuth, elevation  # Converting to degrees for readability
        
    def get_pose_3d(self):
        pose_3d     = np.zeros((17, 3))
        pose_3d[0]  = self.pelvis_origin
        pose_3d[1]  = self.r_hip_origin
        pose_3d[2]  = self.right_leg.link2_origin # self.r_knee_origin
        pose_3d[3]  = self.right_leg.terminal_origin # self.r_ankle_origin
        pose_3d[4]  = self.l_hip_origin
        pose_3d[5]  = self.left_leg.link2_origin # self.l_knee_origin
        pose_3d[6]  = self.left_leg.terminal_origin # self.l_ankle_origin
        pose_3d[7]  = self.torso_origin
        pose_3d[8]  = self.neck_origin
        pose_3d[9]  = self.head.link2_origin # self.nose_origin
        pose_3d[10] = self.head.terminal_origin # self.head_origin
        pose_3d[11] = self.l_shoulder_origin
        pose_3d[12] = self.left_arm.link2_origin # self.l_elbow_origin
        pose_3d[13] = self.left_arm.terminal_origin # self.l_wrist_origin
        pose_3d[14] = self.r_shoulder_origin
        pose_3d[15] = self.right_arm.link2_origin # self.r_elbow_origin
        pose_3d[16] = self.right_arm.terminal_origin # self.r_wrist_origin
        return pose_3d
    
    def get_dh_angles(self, by_dict=False, degree=False):
        if by_dict:
            dh_angles = {}
            # head
            dh_angles['h_l1_yaw'],  dh_angles['h_l1_pitch']  = self.head.link1_yaw,  self.head.link1_pitch
            # right arm
            dh_angles['ra_l1_yaw'], dh_angles['ra_l1_pitch'] = self.right_arm.link1_yaw, self.right_arm.link1_pitch
            dh_angles['ra_l2_yaw'], dh_angles['ra_l2_pitch'] = self.right_arm.link2_yaw, self.right_arm.link2_pitch
            # left arm
            dh_angles['la_l1_yaw'], dh_angles['la_l1_pitch'] = self.left_arm.link1_yaw, self.left_arm.link1_pitch
            dh_angles['la_l2_yaw'], dh_angles['la_l2_pitch'] = self.left_arm.link2_yaw, self.left_arm.link2_pitch
            # right leg
            dh_angles['rl_l1_yaw'], dh_angles['rl_l1_pitch'] = self.right_leg.link1_yaw, self.right_leg.link1_pitch
            dh_angles['rl_l2_yaw'], dh_angles['rl_l2_pitch'] = self.right_leg.link2_yaw, self.right_leg.link2_pitch
            # left leg
            dh_angles['ll_l1_yaw'], dh_angles['ll_l1_pitch'] = self.left_leg.link1_yaw, self.left_leg.link1_pitch
            dh_angles['ll_l2_yaw'], dh_angles['ll_l2_pitch'] = self.left_leg.link2_yaw, self.left_leg.link2_pitch
            if degree:
                for key in dh_angles.keys():
                    dh_angles[key] = math.degrees(dh_angles[key])
        else:
            dh_angles = np.zeros(18)
            # head
            dh_angles[0:2] = np.array([self.head.link1_yaw, self.head.link1_pitch])
            # right arm
            dh_angles[2:6] = np.array([self.right_arm.link1_yaw, self.right_arm.link1_pitch, self.right_arm.link2_yaw, self.right_arm.link2_pitch])
            # left arm
            dh_angles[6:10] = np.array([self.left_arm.link1_yaw, self.left_arm.link1_pitch, self.left_arm.link2_yaw, self.left_arm.link2_pitch])
            # right leg
            dh_angles[10:14] = np.array([self.right_leg.link1_yaw, self.right_leg.link1_pitch, self.right_leg.link2_yaw, self.right_leg.link2_pitch])
            # left leg
            dh_angles[14:18] = np.array([self.left_leg.link1_yaw, self.left_leg.link1_pitch, self.left_leg.link2_yaw, self.left_leg.link2_pitch])
            
            if degree:
                dh_angles = np.degrees(dh_angles)
        return dh_angles
    
    def get_appendage_length(self, by_dict=False):
        if by_dict:
            appendage_length = {}  
            appendage_length['h_l1_length']  = self.head.link1_length
            appendage_length['h_l2_length']  = self.head.link2_length
            appendage_length['ra_l1_length'] = self.right_arm.link1_length
            appendage_length['ra_l2_length'] = self.right_arm.link2_length
            appendage_length['la_l1_length'] = self.left_arm.link1_length
            appendage_length['la_l2_length'] = self.left_arm.link2_length
            appendage_length['rl_l1_length'] = self.right_leg.link1_length
            appendage_length['rl_l2_length'] = self.right_leg.link2_length
            appendage_length['ll_l1_length'] = self.left_leg.link1_length
            appendage_length['ll_l2_length'] = self.left_leg.link2_length
        else:
            appendage_length = np.zeros(10)
            appendage_length[0:2] = np.array([self.head.link1_length, self.head.link2_length])
            appendage_length[2:4] = np.array([self.right_arm.link1_length, self.right_arm.link2_length])
            appendage_length[4:6] = np.array([self.left_arm.link1_length, self.left_arm.link2_length])
            appendage_length[6:8] = np.array([self.right_leg.link1_length, self.right_leg.link2_length])
            appendage_length[8:10] = np.array([self.left_leg.link1_length, self.left_leg.link2_length])
        return appendage_length
    
    def get_keypoints(self, by_dict=False):
        if by_dict:
            keypoints = {}
            keypoints['pelvis'] = self.pelvis_origin  # 0
            keypoints['r_hip'] = self.r_hip_origin # 1
            keypoints['l_hip'] = self.l_hip_origin # 2
            keypoints['torso'] = self.torso_origin # 3
            keypoints['neck'] = self.neck_origin # 4
            keypoints['nose'] = self.nose_origin # 5
            keypoints['head'] = self.head_origin # 6
            keypoints['l_shoulder'] = self.l_shoulder_origin  # 7
            keypoints['r_shoulder'] = self.r_shoulder_origin  # 8
        else:
            keypoints = []
            keypoints.append(self.pelvis_origin)
            keypoints.append(self.r_hip_origin)
            keypoints.append(self.l_hip_origin)
            keypoints.append(self.torso_origin)
            keypoints.append(self.neck_origin)
            keypoints.append(self.nose_origin)
            keypoints.append(self.head_origin)
            keypoints.append(self.l_shoulder_origin)
            keypoints.append(self.r_shoulder_origin)
            keypoints = np.array(keypoints).reshape(-1)
        return keypoints
    
    def mpjpe(self, gt):
        return np.mean(np.linalg.norm(self.get_pose_3d() - gt, axis=1))
    
    def draw(self, ax, draw_frame=False, head_length=0.01, scale=0.1, fontsize=10, show_name=False, show_axis=False):
        self.body_frame.draw3d(color='tab:orange', head_length=head_length, scale=scale, show_name=show_name, show_axis=show_axis)
        self.head.draw(ax, draw_frame, head_length, scale, fontsize, show_name, show_axis=show_axis)
        self.right_arm.draw(ax, draw_frame, head_length, scale, fontsize, show_name, show_axis=show_axis)
        self.left_arm.draw(ax, draw_frame, head_length, scale, fontsize, show_name, show_axis=show_axis)
        self.right_leg.draw(ax, draw_frame, head_length, scale, fontsize, show_name, show_axis=show_axis)
        self.left_leg.draw(ax, draw_frame, head_length, scale, fontsize, show_name, show_axis=show_axis)
        
# --------------------------------------------------------------------------------

def get_reference_frame(cam_origin, pelvis_point):
    # define reference frame when cam_origin and pelvis are determined_
    ref_origin = pelvis_point * np.array([1, 1, 0])
    ref_forward = (cam_origin - pelvis_point) * np.array([1, 1, 0])
    ref_forward /= np.linalg.norm(ref_forward)
    ref_up = np.array([0, 0, 1])
    ref_left = np.cross(ref_up, ref_forward)
    return ref_origin, frame_vec_to_matrix(ref_forward, ref_left, ref_up)

def get_batch_reference_frame(batch_cam_origin, batch_pelvis_point):
    # define reference frame when cam_origin and pelvis are determined_
    data_type = batch_cam_origin.dtype
    device = batch_cam_origin.device
    batch_ref_origin = batch_pelvis_point * torch.tensor([1, 1, 0]).to(device)
    batch_ref_forward = (batch_cam_origin - batch_pelvis_point) * torch.tensor([1, 1, 0]).type(data_type).to(device)
    batch_ref_forward = batch_ref_forward / torch.norm(batch_ref_forward, dim=-1, keepdim=True)
    batch_ref_up = torch.tensor([0, 0, 1]).unsqueeze(0).unsqueeze(0).repeat(batch_cam_origin.shape[0], batch_cam_origin.shape[1], 1).to(device).type(data_type)
    batch_ref_left = torch.cross(batch_ref_up, batch_ref_forward)
    return batch_ref_origin, torch.stack([batch_ref_forward, batch_ref_left, batch_ref_up], dim=-1)

def generate_tf_from_origin_R(origin, R):
    # generate transform matrix from origin and rotation matrix
    tf = np.eye(4)
    tf[:3, :3] = R
    tf[:3, 3] = origin
    return tf

def generate_batch_tf_from_batch_origin_R(batch_origin, batch_R):
    # generate batch transform matrix from batch origin and rotation matrix
    batch_tf = torch.eye(4)
    if len(batch_R.shape) == 3:
        batch_tf = batch_tf.unsqueeze(0).repeat(batch_origin.shape[0], 1, 1)
        batch_tf[:, :3, :3] = batch_R
        batch_tf[:, :3, 3] = batch_origin
    elif len(batch_R.shape) == 4:
        batch_tf = batch_tf.unsqueeze(0).unsqueeze(0).repeat(batch_origin.shape[0], batch_origin.shape[1], 1, 1)
        batch_tf[:, :, :3, :3] = batch_R
        batch_tf[:, :, :3, 3] = batch_origin
    return batch_tf

def inverse_tf(tf):
    # inverse transform matrix
    inv_tf = np.eye(4)
    inv_tf[:3, :3] = tf[:3, :3].T
    inv_tf[:3, 3] = -tf[:3, :3].T @ tf[:3, 3]
    return inv_tf

def batch_inverse_tf(batch_tf):
    # inverse batch transform matrix
    batch_inv_tf = torch.eye(4)
    if len(batch_tf.shape) == 3:
        batch_inv_tf = batch_inv_tf.unsqueeze(0).repeat(batch_tf.shape[0], 1, 1)
        batch_inv_tf[:, :3, :3] = batch_tf[:, :3, :3].transpose(1, 2)
        batch_inv_tf[:, :3, 3] = -batch_tf[:, :3, :3].transpose(1, 2) @ batch_tf[:, :3, 3]
    elif len(batch_tf.shape) == 4:
        batch_inv_tf = batch_inv_tf.unsqueeze(0).unsqueeze(0).repeat(batch_tf.shape[0], batch_tf.shape[1], 1, 1)
        batch_inv_tf[:, :, :3, :3] = batch_tf[:, :, :3, :3].transpose(2, 3)
        print(batch_tf[:, :, :3, :3].transpose(2, 3).shape, batch_tf[:, :, :3, 3].unsqueeze(-1).shape)
        batch_inv_tf[:, :, :3, 3] = (-batch_tf[:, :, :3, :3].transpose(2, 3) @ batch_tf[:, :, :3, 3].unsqueeze(-1)).squeeze(-1)
    return batch_inv_tf

# torso frame

def get_frame_from_keypoints(kp1, kp2, kp3, kp4):
    # kp1: left tail, kp2: left head
    # lower frame
    left = kp2 - kp1 # [B, F, 3]
    left = left / np.linalg.norm(left)
    kp3_to_kp4 = kp4 - kp3 # [B, F, 3]
    forward = np.cross(left, kp3_to_kp4) # [B, F, 3]
    forward = forward / np.linalg.norm(forward)
    up = np.cross(forward, left) # [B, F, 3]
    return forward, left, up

def get_lower_torso_frame_from_pose(pose):
    r_hip, l_hip, pelvis, torso = get_h36m_keypoints(pose, ['r_hip', 'l_hip', 'pelvis', 'torso'])
    forward, left, up = get_frame_from_keypoints(r_hip, l_hip, pelvis, torso)
    lower_origin = pelvis
    lower_frame_R = frame_vec_to_matrix(forward, left, up) # x-axis: [:, 0], y-axis: [:, 1], z-axis: [:, 2]
    return lower_origin, lower_frame_R

def get_upper_torso_frame_from_pose(pose):
    r_shoulder, l_shoulder, torso, neck = get_h36m_keypoints(pose, ['r_shoulder', 'l_shoulder', 'torso', 'neck'])
    forward, left, up = get_frame_from_keypoints(r_shoulder, l_shoulder, torso, neck)
    upper_origin = (r_shoulder + l_shoulder)/2
    upper_frame_R = frame_vec_to_matrix(forward, left, up) # x-axis: [:, 0], y-axis: [:, 1], z-axis: [:, 2]
    return upper_origin, upper_frame_R

def frame_vec_to_matrix(forward, left, up):
    # forward: x-axis
    # left: y-axis
    # up: z-axis
    return np.array([forward, left, up]).T

def generate_vis_frame_from_R(origin, R, name=''):
    forward, left, up = R[:, 0], R[:, 1], R[:, 2]
    frame = ReferenceFrame(
        origin=origin,
        dx=forward,
        dy=left,
        dz=up,
        name=name,
    )
    return frame
        
## Batch tosro frame

def batch_rot_z_matrix(batch_roll):
    m11 = torch.cos(batch_roll)
    m12 = -torch.sin(batch_roll)
    m13 = torch.zeros_like(batch_roll)
    m21 = torch.sin(batch_roll)
    m22 = torch.cos(batch_roll)
    m23 = torch.zeros_like(batch_roll)
    m31 = torch.zeros_like(batch_roll)
    m32 = torch.zeros_like(batch_roll)
    m33 = torch.ones_like(batch_roll)
    row1 = torch.concat([m11, m12, m13], dim=-1)
    row2 = torch.concat([m21, m22, m23], dim=-1)
    row3 = torch.concat([m31, m32, m33], dim=-1)
    return torch.stack([row1, row2, row3], dim=-1).transpose(2, 3) # [B, F, 3, 3]

def batch_rot_y_matrix(batch_yaw):
    m11 = torch.cos(batch_yaw)
    m12 = torch.zeros_like(batch_yaw)
    m13 = torch.sin(batch_yaw)
    m21 = torch.zeros_like(batch_yaw)
    m22 = torch.ones_like(batch_yaw)
    m23 = torch.zeros_like(batch_yaw)
    m31 = -torch.sin(batch_yaw)
    m32 = torch.zeros_like(batch_yaw)
    m33 = torch.cos(batch_yaw)
    row1 = torch.concat([m11, m12, m13], dim=-1)
    row2 = torch.concat([m21, m22, m23], dim=-1)
    row3 = torch.concat([m31, m32, m33], dim=-1)
    return torch.stack([row1, row2, row3], dim=-1).transpose(2, 3) # [B, F, 3, 3]

def batch_rot_x_matrix(batch_pitch):
    m11 = torch.ones_like(batch_pitch)
    m12 = torch.zeros_like(batch_pitch)
    m13 = torch.zeros_like(batch_pitch)
    m21 = torch.zeros_like(batch_pitch)
    m22 = torch.cos(batch_pitch)
    m23 = -torch.sin(batch_pitch)
    m31 = torch.zeros_like(batch_pitch)
    m32 = torch.sin(batch_pitch)
    m33 = torch.cos(batch_pitch)
    row1 = torch.concat([m11, m12, m13], dim=-1)
    row2 = torch.concat([m21, m22, m23], dim=-1)
    row3 = torch.concat([m31, m32, m33], dim=-1)
    return torch.stack([row1, row2, row3], dim=-1).transpose(2, 3) # [B, F, 3, 3]

def get_batch_frame_vec_from_keypoints(batch_kp1, batch_kp2, batch_kp3, batch_kp4):
    # kp1: left tail, kp2: left head
    # lower frame
    batch_left = batch_kp2 - batch_kp1 # [B, F, 3]
    batch_left = batch_left / torch.norm(batch_left, dim=-1, keepdim=True)
    batch_kp3_to_kp4 = batch_kp4 - batch_kp3 # [B, F, 3]
    batch_forward = torch.cross(batch_left, batch_kp3_to_kp4) # [B, F, 3]
    batch_forward = batch_forward / torch.norm(batch_forward, dim=-1, keepdim=True)
    batch_up = torch.cross(batch_forward, batch_left) # [B, F, 3]
    batch_R = torch.stack([batch_forward, batch_left, batch_up], dim=-1) # [B, F, 3, 3]
    return batch_R
    
def get_batch_lower_torso_frame_from_keypoints(batch_r_hip, batch_l_hip, batch_pelvis, batch_torso):
    batch_lower_frame_origin = batch_pelvis
    batch_lower_frame_R = get_batch_frame_vec_from_keypoints(batch_r_hip, batch_l_hip, batch_pelvis, batch_torso)
    return batch_lower_frame_origin, batch_lower_frame_R

def get_batch_upper_torso_frame_from_keypoints(batch_r_shoulder, batch_l_shoulder, batch_torso, batch_neck):
    batch_upper_frame_origin = (batch_r_shoulder + batch_l_shoulder) / 2 # [B, F, 3]
    batch_upper_frame_R = get_batch_frame_vec_from_keypoints(batch_r_shoulder, batch_l_shoulder, batch_torso, batch_neck)
    return batch_upper_frame_origin, batch_upper_frame_R

def get_batch_lower_torso_frame_from_pose(batch_pose):
    output = get_batch_h36m_keypoints(batch_pose, ['r_hip', 'l_hip', 'pelvis', 'torso'])
    return get_batch_lower_torso_frame_from_keypoints(output[:, :, 0], output[:, :, 1], output[:, :, 2], output[:, :, 3])

def get_batch_upper_torso_frame_from_pose(batch_pose):
    output = get_batch_h36m_keypoints(batch_pose, ['r_shoulder', 'l_shoulder', 'torso', 'neck'])
    return get_batch_upper_torso_frame_from_keypoints(output[:, :, 0], output[:, :, 1], output[:, :, 2], output[:, :, 3])



# Batch version of Appendage class
class BatchAppendage:
    def __init__(self, batch_link1_length, batch_link2_length, batch_link1_yaw_init=None, batch_link1_pitch_init=None, batch_link2_yaw_init=None, batch_link2_pitch_init=None, 
                 degree=True, batch_root_tf=np.eye(4), device='cuda', data_type=torch.float32):
        self.batch_size = batch_link1_length.shape[0]
        self.num_frames = batch_link1_length.shape[1]
        self.data_type = data_type
        self.device = device
        
        if batch_link1_yaw_init   == None: batch_link1_yaw_init   = torch.zeros(self.batch_size, self.num_frames)
        if batch_link1_pitch_init == None: batch_link1_pitch_init = torch.zeros(self.batch_size, self.num_frames)
        if batch_link2_yaw_init   == None: batch_link2_yaw_init   = torch.zeros(self.batch_size, self.num_frames)
        if batch_link2_pitch_init == None: batch_link2_pitch_init = torch.zeros(self.batch_size, self.num_frames)
        
        if type(batch_link1_length)     == np.ndarray: self.batch_link1_length = torch.tensor(batch_link1_length)
        if type(batch_link2_length)     == np.ndarray: self.batch_link2_length = torch.tensor(batch_link2_length)
        if type(batch_link1_yaw_init)   == np.ndarray: self.batch_link1_yaw    = torch.tensor(batch_link1_yaw_init)
        if type(batch_link1_pitch_init) == np.ndarray: self.batch_link1_pitch  = torch.tensor(batch_link1_pitch_init)
        if type(batch_link2_yaw_init)   == np.ndarray: self.batch_link2_yaw    = torch.tensor(batch_link2_yaw_init)       
        if type(batch_link2_pitch_init) == np.ndarray: self.batch_link2_pitch  = torch.tensor(batch_link2_pitch_init)
        if type(batch_root_tf)          == np.ndarray: self.batch_root_tf      = torch.tensor(batch_root_tf)

        self.batch_link1_length = batch_link1_length.type(data_type).to(device)
        self.batch_link2_length = batch_link2_length.type(data_type).to(device)
        self.batch_link1_yaw    = batch_link1_yaw_init.type(data_type).to(device)
        self.batch_link1_pitch  = batch_link1_pitch_init.type(data_type).to(device)
        self.batch_link2_yaw    = batch_link2_yaw_init.type(data_type).to(device)
        self.batch_link2_pitch  = batch_link2_pitch_init.type(data_type).to(device)
        self.batch_root_tf      = batch_root_tf.type(data_type).to(device)
        
        # forwad kinematics
        self.update_batch_link(self.batch_link1_yaw, self.batch_link1_pitch, self.batch_link2_yaw, self.batch_link2_pitch, degree=degree)

    def batch_DH_matrix(self, batch_theta, batch_alpha, batch_d):
        m11  = (torch.cos(batch_theta)*torch.cos(batch_alpha)).unsqueeze(-1)
        m12  = -torch.sin(batch_theta).unsqueeze(-1)
        m13  = (-torch.cos(batch_theta)*torch.sin(batch_alpha)).unsqueeze(-1)
        m14  = (batch_d*torch.cos(batch_alpha)*torch.cos(batch_theta)).unsqueeze(-1)
        m21  = (torch.sin(batch_theta)*torch.cos(batch_alpha)).unsqueeze(-1)
        m22  = torch.cos(batch_theta).unsqueeze(-1)
        m23  = (-torch.sin(batch_theta)*torch.sin(batch_alpha)).unsqueeze(-1)
        m24  = (batch_d*torch.sin(batch_theta)*torch.cos(batch_alpha)).unsqueeze(-1)
        m31  = torch.sin(batch_alpha).unsqueeze(-1)
        m32  = torch.zeros_like(batch_theta).unsqueeze(-1)
        m33  = torch.cos(batch_alpha).unsqueeze(-1)
        m34  = (batch_d*torch.sin(batch_alpha)).unsqueeze(-1)
        m41  = torch.zeros_like(batch_theta).unsqueeze(-1)
        m42  = torch.zeros_like(batch_theta).unsqueeze(-1)
        m43  = torch.zeros_like(batch_theta).unsqueeze(-1)
        m44  = torch.ones_like(batch_theta).unsqueeze(-1)
        row1 = torch.concat([m11, m12, m13, m14], dim=-1)
        row2 = torch.concat([m21, m22, m23, m24], dim=-1)
        row3 = torch.concat([m31, m32, m33, m34], dim=-1)
        row4 = torch.concat([m41, m42, m43, m44], dim=-1)
        return torch.stack([row1, row2, row3, row4], dim=-1).transpose(2, 3) # [B, F, 4, 4]
    
    def batch_build_dh_frame(self, batch_yaw, batch_pitch, batch_d, batch_parent_tf, degree=False, name=''): # yaw = theta, pitch = alpha
        ## input size
        # batch_yaw, batch_pitch, batch_d: [B, F]
        # batch_parent_tf: [B, F, 4, 4]
        if degree:
            batch_yaw = torch.deg2rad(batch_yaw) 
            batch_pitch = torch.deg2rad(batch_pitch)
        batch_dh_matrix = self.batch_DH_matrix(batch_theta=batch_yaw, batch_alpha=batch_pitch, batch_d=batch_d)
        batch_child_tf = batch_parent_tf @ batch_dh_matrix
        ## output size
        # batch_child_tf, batch_dh_matrix: [B, F, 4, 4]
        return batch_child_tf, batch_dh_matrix
    
    def generate_vis_frame(self, pos, R, name='dh_frame'):
        dh_frame = ReferenceFrame(
            origin=pos, 
            dx=R[0], 
            dy=R[1],
            dz=R[2],
            name=name,
        )
        return dh_frame
    
    def get_vis_frame(self, batch_num, frame_num, frame_type='root'):
        if frame_type == 'root':
            pos = self.batch_root_tf[batch_num, frame_num, :3, 3]
            R = self.batch_root_tf[batch_num, frame_num, :3, :3]
        elif frame_type == 'link1':
            pos = self.batch_root_tf[batch_num, frame_num, :3, 3]
            R = self.batch_link1_tf[batch_num, frame_num, :3, :3]
        elif frame_type == 'link2':
            pos = self.batch_link1_tf[batch_num, frame_num, :3, 3]
            R = self.batch_link2_tf[batch_num, frame_num, :3, :3]
        # elif frame_type == 'terminal':
        #     pos = self.batch_terminal_tf[batch_num, frame_num, :3, 3]
        #     R = self.batch_terminal_tf[batch_num, frame_num, :3, :3]
        else:
            raise ValueError('frame_type should be root, link1, or link2')
       
        dh_frame = self.generate_vis_frame(pos.cpu().detach().numpy(), R.cpu().detach().numpy(), name=frame_type)
        return dh_frame
    
    def update_batch_link(self, batch_link1_yaw=None, batch_link1_pitch=None, batch_link2_yaw=None, batch_link2_pitch=None, degree=False):
        # update angles
        if batch_link1_yaw != None:
            self.batch_link1_yaw = batch_link1_yaw
        if batch_link1_pitch != None:
            self.batch_link1_pitch = batch_link1_pitch
        if batch_link2_yaw != None:
            self.batch_link2_yaw = batch_link2_yaw
        if batch_link2_pitch != None:
            self.batch_link2_pitch = batch_link2_pitch
            
        # link1
        self.batch_link1_tf, self.batch_link1_dh_mat = self.batch_build_dh_frame(self.batch_link1_yaw, self.batch_link1_pitch, self.batch_link1_length, self.batch_root_tf, degree=degree, name='link1')
        self.batch_link1_origin = self.batch_root_tf[:, :, :3, 3]
        self.batch_link1_R = self.batch_link1_tf[:, :, :3, :3]
        
        # link2
        self.batch_link2_tf, self.batch_link2_dh_mat = self.batch_build_dh_frame(self.batch_link2_yaw, self.batch_link2_pitch, self.batch_link2_length, self.batch_link1_tf, degree=degree, name='link2')
        self.batch_link2_origin = self.batch_link1_tf[:, :, :3, 3]
        self.batch_link2_R = self.batch_link2_tf[:, :, :3, :3]

        # terminal
        #self.terminal_tf, self.terminal_frame, self.terminal_dh_mat = self.build_dh_frame(0, 0, 0, self.link2_tf, degree=degree, name='terminal')
        self.batch_terminal_origin = self.batch_link2_tf[:, :, :3, 3]
        
        # vector
        self.batch_link1_vec = self.batch_link2_origin - self.batch_link1_origin
        self.batch_link2_vec = self.batch_terminal_origin - self.batch_link2_origin
    
    def draw(self, ax, batch_num, frame_num, draw_frame=False, head_length=0.01, scale=0.1, fontsize=10, show_name=False):
        link1_origin = self.batch_link1_origin[batch_num, frame_num].cpu().detach().numpy()
        link2_origin = self.batch_link2_origin[batch_num, frame_num].cpu().detach().numpy()
        terminal_origin = self.batch_terminal_origin[batch_num, frame_num].cpu().detach().numpy()
        
        #plt.sca(ax)
        ax.plot(link1_origin[0], link1_origin[1], link1_origin[2],  '.k') # link1 origin        
        ax.plot(*np.c_[link1_origin, link2_origin], color="tab:gray", ls='--') # link1
        ax.plot(link2_origin[0], link2_origin[1], link2_origin[2], '.b') # link2 origin
        ax.plot(*np.c_[link2_origin, terminal_origin], color="tab:gray", ls='--') # link2 arm
        ax.plot(terminal_origin[0], terminal_origin[1], terminal_origin[2], '.r') # terminal origin
        if draw_frame:
            self.get_vis_frame(batch_num, frame_num, frame_type='root').draw3d(color='tab:orange', head_length=head_length, scale=scale, fontsize=fontsize, show_name=show_name) # root frame
            self.get_vis_frame(batch_num, frame_num, frame_type='link1').draw3d(color='tab:red', head_length=head_length, scale=scale, fontsize=fontsize, show_name=show_name) # link1 frame
            self.get_vis_frame(batch_num, frame_num, frame_type='link2').draw3d(color='tab:blue', head_length=head_length, scale=scale, fontsize=fontsize, show_name=show_name) # link2 frame
            
            
# Batch version of DH Model class
class BatchDHModel:
    def __init__(self, batch_pose_3d=None, head=False, device='cuda', batch_size=8, num_frames=243, data_type=torch.float32, world_z_direction=[0, 0, 1]) -> None:
        self.head = head
        self.data_type = data_type
        self.device = device
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.world_z_direction = world_z_direction
        self.batch_zero = torch.zeros(self.batch_size, self.num_frames, dtype=self.data_type).to(device)
        
        # rotation matrices
        self.left_init_R  = Rotation.from_rotvec( np.pi/2 * np.array([0, 0, 1])).as_matrix() # rotate -90 deg wrt z-axis
        self.right_init_R = Rotation.from_rotvec(-np.pi/2 * np.array([0, 0, 1])).as_matrix() # rotate  90 deg wrt z-axis
        self.left_init_R  = torch.tensor(self.left_init_R,  dtype=self.data_type).to(device) # to tensor
        self.right_init_R = torch.tensor(self.right_init_R, dtype=self.data_type).to(device) # to tensor
        
        # appendage id
        self.head_id = 0
        self.right_arm_id = 1
        self.left_arm_id = 2
        self.right_leg_id = 3
        self.left_leg_id = 4
        self.right_upper_arm_id = 2
        self.right_lower_arm_id = 3
        self.left_upper_arm_id = 4
        self.left_lower_arm_id = 5
        self.right_upper_leg_id = 6
        self.right_lower_leg_id = 7
        self.left_upper_leg_id = 8
        self.left_lower_leg_id = 9
        
        if batch_pose_3d is not None:
            # set dh model
            self.set_batch_dh_model_from_batch_pose(batch_pose_3d)
    # -------------------------------------------------------------------------------------------
    # set functions
    def set_batch_dh_model_from_batch_pose(self, batch_pose_3d, batch_size=None, num_frames=None, length_type='each'):
        ## length type : 'each', 'mean', 'first'
        # if batch_size != None: 
        #     self.batch_size = batch_size
        #     self.batch_zero = torch.zeros(self.batch_size, self.num_frames, dtype=self.data_type).to(self.device)
        # if num_frames != None:
        #     self.num_frames = num_frames
        self.batch_size = batch_pose_3d.shape[0]
        self.batch_zero = torch.zeros(self.batch_size, self.num_frames, dtype=self.data_type).to(self.device)
                
        # check input
        assert batch_pose_3d.shape == (self.batch_size, self.num_frames, 17, 3), 'batch_pose_3d should be (batch_size, num_frames, 17, 3)'
        if type(batch_pose_3d) == np.ndarray: batch_pose_3d = torch.tensor(batch_pose_3d)
        batch_pose_3d = batch_pose_3d.type(self.data_type).to(self.device)
        # update keypoints
        self.set_batch_keypoints_from_batch_pose(batch_pose_3d)        
        # extract vectors wrt world frame
        self.set_batch_limb_vectors()
        # set lengths
        self.set_batch_limb_length(length_type) # only for init_pose_3d
        # get body reference frame
        self.set_batch_body_reference_frame()
        ## generate appendages ---------------------------------------------------------------------
        self.generate_all_batch_appendages()
        # extract angles and update appendages
        if self.head:
            out = self.extract_and_update_batch_appendage_angles(self.batch_head,      self.batch_neck_to_nose_vector,    self.batch_nose_to_head_vector)
        out = self.extract_and_update_batch_appendage_angles(self.batch_right_arm, self.batch_right_upper_arm_vector, self.batch_right_lower_arm_vector)
        self.batch_right_upper_arm_yaw, self.batch_right_lower_arm_pitch, self.batch_right_upper_arm_yaw, self.batch_right_lower_arm_pitch = out
        out = self.extract_and_update_batch_appendage_angles(self.batch_left_arm,  self.batch_left_upper_arm_vector,  self.batch_left_lower_arm_vector)
        self.batch_left_upper_arm_yaw,  self.batch_left_lower_arm_pitch,  self.batch_left_upper_arm_yaw,  self.batch_left_lower_arm_pitch  = out
        out = self.extract_and_update_batch_appendage_angles(self.batch_right_leg, self.batch_right_upper_leg_vector, self.batch_right_lower_leg_vector)
        self.batch_right_upper_leg_yaw, self.batch_right_lower_leg_pitch, self.batch_right_upper_leg_yaw, self.batch_right_lower_leg_pitch = out
        out = self.extract_and_update_batch_appendage_angles(self.batch_left_leg,  self.batch_left_upper_leg_vector,  self.batch_left_lower_leg_vector)
        self.batch_left_upper_leg_yaw,  self.batch_left_lower_leg_pitch,  self.batch_left_upper_leg_yaw,  self.batch_left_lower_leg_pitch  = out
        
    def set_batch_keypoints_from_batch_pose(self, batch_pose_3d):
        # get head points
        self.batch_head_origin       = batch_pose_3d[:, :, 10]
        self.batch_nose_origin       = batch_pose_3d[:, :, 9]
        # get torso points
        self.batch_pelvis_origin     = batch_pose_3d[:, :, 0]
        self.batch_r_hip_origin      = batch_pose_3d[:, :, 1]
        self.batch_l_hip_origin      = batch_pose_3d[:, :, 4]
        self.batch_torso_origin      = batch_pose_3d[:, :, 7]
        self.batch_neck_origin       = batch_pose_3d[:, :, 8]
        self.batch_l_shoulder_origin = batch_pose_3d[:, :, 11]
        self.batch_r_shoulder_origin = batch_pose_3d[:, :, 14]
        # get appendage points
        self.batch_l_elbow_origin    = batch_pose_3d[:, :, 12]
        self.batch_l_wrist_origin    = batch_pose_3d[:, :, 13]
        self.batch_r_elbow_origin    = batch_pose_3d[:, :, 15]
        self.batch_r_wrist_origin    = batch_pose_3d[:, :, 16]
        self.batch_l_knee_origin     = batch_pose_3d[:, :, 5]
        self.batch_l_ankle_origin    = batch_pose_3d[:, :, 6]
        self.batch_r_knee_origin     = batch_pose_3d[:, :, 2]
        self.batch_r_ankle_origin    = batch_pose_3d[:, :, 3]
        
    def set_batch_limb_vectors(self):
        # (B, F, 3)
        self.batch_neck_to_nose_vector    = self.batch_nose_origin    - self.batch_neck_origin
        self.batch_nose_to_head_vector    = self.batch_head_origin    - self.batch_nose_origin
        self.batch_left_upper_arm_vector  = self.batch_l_elbow_origin - self.batch_l_shoulder_origin
        self.batch_left_lower_arm_vector  = self.batch_l_wrist_origin - self.batch_l_elbow_origin
        self.batch_right_upper_arm_vector = self.batch_r_elbow_origin - self.batch_r_shoulder_origin
        self.batch_right_lower_arm_vector = self.batch_r_wrist_origin - self.batch_r_elbow_origin
        self.batch_left_upper_leg_vector  = self.batch_l_knee_origin  - self.batch_l_hip_origin
        self.batch_left_lower_leg_vector  = self.batch_l_ankle_origin - self.batch_l_knee_origin
        self.batch_right_upper_leg_vector = self.batch_r_knee_origin  - self.batch_r_hip_origin
        self.batch_right_lower_leg_vector = self.batch_r_ankle_origin - self.batch_r_knee_origin
        
    def set_batch_limb_length(self, length_type='each'):
        ## length type : 'each', 'mean', 'first'
        # (B, F)
        if self.head:
            self.batch_neck_to_nose_length = torch.norm(self.batch_neck_to_nose_vector   , dim=-1) 
            self.batch_nose_to_head_length = torch.norm(self.batch_nose_to_head_vector   , dim=-1)
        self.batch_right_upper_arm_length = torch.norm(self.batch_right_upper_arm_vector, dim=-1) # limb_lens[14]
        self.batch_right_lower_arm_length = torch.norm(self.batch_right_lower_arm_vector, dim=-1) # limb_lens[15]
        self.batch_left_upper_arm_length  = torch.norm(self.batch_left_upper_arm_vector , dim=-1) # limb_lens[11]
        self.batch_left_lower_arm_length  = torch.norm(self.batch_left_lower_arm_vector , dim=-1) # limb_lens[12]
        self.batch_right_upper_leg_length = torch.norm(self.batch_right_upper_leg_vector, dim=-1) # limb_lens[1]
        self.batch_right_lower_leg_length = torch.norm(self.batch_right_lower_leg_vector, dim=-1) # limb_lens[2]
        self.batch_left_upper_leg_length  = torch.norm(self.batch_left_upper_leg_vector , dim=-1) # limb_lens[4]
        self.batch_left_lower_leg_length  = torch.norm(self.batch_left_lower_leg_vector , dim=-1) # limb_lens[5]
        if length_type == 'each':
            pass
        elif length_type == 'mean':
            if self.head:
                self.batch_neck_to_nose_length = torch.mean(self.batch_neck_to_nose_length, dim=1).unsqueeze(-1).repeat(1, self.num_frames)
                self.batch_nose_to_head_length = torch.mean(self.batch_nose_to_head_length, dim=1).unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_right_upper_arm_length = torch.mean(self.batch_right_upper_arm_length, dim=1).unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_right_lower_arm_length = torch.mean(self.batch_right_lower_arm_length, dim=1).unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_left_upper_arm_length  = torch.mean(self.batch_left_upper_arm_length,  dim=1).unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_left_lower_arm_length  = torch.mean(self.batch_left_lower_arm_length,  dim=1).unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_right_upper_leg_length = torch.mean(self.batch_right_upper_leg_length, dim=1).unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_right_lower_leg_length = torch.mean(self.batch_right_lower_leg_length, dim=1).unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_left_upper_leg_length  = torch.mean(self.batch_left_upper_leg_length,  dim=1).unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_left_lower_leg_length  = torch.mean(self.batch_left_lower_leg_length,  dim=1).unsqueeze(-1).repeat(1, self.num_frames)
        elif length_type == 'first':
            if self.head:
                self.batch_neck_to_nose_length = self.batch_neck_to_nose_length[:, 0].unsqueeze(-1).repeat(1, self.num_frames)
                self.batch_nose_to_head_length = self.batch_nose_to_head_length[:, 0].unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_right_upper_arm_length = self.batch_right_upper_arm_length[:, 0].unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_right_lower_arm_length = self.batch_right_lower_arm_length[:, 0].unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_left_upper_arm_length  = self.batch_left_upper_arm_length[:,  0].unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_left_lower_arm_length  = self.batch_left_lower_arm_length[:,  0].unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_right_upper_leg_length = self.batch_right_upper_leg_length[:, 0].unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_right_lower_leg_length = self.batch_right_lower_leg_length[:, 0].unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_left_upper_leg_length  = self.batch_left_upper_leg_length[:,  0].unsqueeze(-1).repeat(1, self.num_frames)
            self.batch_left_lower_leg_length  = self.batch_left_lower_leg_length[:,  0].unsqueeze(-1).repeat(1, self.num_frames)
        
    def set_batch_body_reference_frame(self):
        ## z axis
        batch_world_z_axis = torch.tensor(self.world_z_direction, dtype=self.data_type).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.num_frames, 1).to(self.device) # [B, F, 3]
        # y axis
        batch_y_axis = self.batch_l_hip_origin - self.batch_pelvis_origin # [B, F, 3]
        batch_y_axis = batch_y_axis.type(self.data_type)
        batch_y_axis[:, :, 2] = 0
        batch_y_axis_mag = torch.norm(batch_y_axis, dim=2).unsqueeze(-1)
        batch_y_axis = batch_y_axis/batch_y_axis_mag # '/=' is inplace operation
        # x axis
        # batch_pelvis_to_spine = self.batch_torso_origin - self.batch_pelvis_origin # [B, F, 3]
        # batch_pelvis_to_spine_mag = torch.norm(batch_pelvis_to_spine, dim=2).unsqueeze(-1)
        # batch_pelvis_to_spine = batch_pelvis_to_spine/batch_pelvis_to_spine_mag # '/=' is inplace operation
        # batch_x_axis = torch.cross(batch_y_axis, batch_pelvis_to_spine, dim=2) # [B, F, 3]
        batch_x_axis = torch.cross(batch_y_axis, batch_world_z_axis, dim=2) # [B, F, 3]

        # body_R
        self.batch_body_R = torch.cat([batch_x_axis.unsqueeze(-1), batch_y_axis.unsqueeze(-1), batch_world_z_axis.unsqueeze(-1)], dim=-1).transpose(2, 3) # [B, F, 3, 3]
        
    def generate_batch_appendage(self, appendage_id, root_origin, batch_link1_length, batch_link2_length):
        # if appendage_id == self.head_id:
        #     batch_R = self.batch_body_R
        # elif appendage_id in [self.right_arm_id, self.right_leg_id]:
        #     batch_R = self.right_init_R.T @ self.batch_body_R
        # elif appendage_id in [self.left_arm_id, self.left_leg_id]:
        #     batch_R = self.left_init_R.T @ self.batch_body_R
        batch_R = self.batch_body_R
 
        batch_root_tf = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.num_frames, 1, 1).type(self.data_type).to(self.device)
        batch_root_tf[:, :, :3, :3] = batch_R
        batch_root_tf[:, :, :3, 3] = root_origin
        
        return BatchAppendage(batch_link1_length, batch_link2_length, degree=True, batch_root_tf=batch_root_tf)
    
    def generate_all_batch_appendages(self):
        if self.head:
            self.batch_head  = self.generate_batch_appendage(self.head_id,      self.batch_neck_origin,       self.batch_neck_to_nose_length,    self.batch_nose_to_head_length)
        self.batch_right_arm = self.generate_batch_appendage(self.right_arm_id, self.batch_r_shoulder_origin, self.batch_right_upper_arm_length, self.batch_right_lower_arm_length)
        self.batch_left_arm  = self.generate_batch_appendage(self.left_arm_id,  self.batch_l_shoulder_origin, self.batch_left_upper_arm_length,  self.batch_left_lower_arm_length)
        self.batch_left_leg  = self.generate_batch_appendage(self.left_leg_id,  self.batch_l_hip_origin,      self.batch_left_upper_leg_length,  self.batch_left_lower_leg_length)
        self.batch_right_leg = self.generate_batch_appendage(self.right_leg_id, self.batch_r_hip_origin,      self.batch_right_upper_leg_length, self.batch_right_lower_leg_length)
    
    def extract_and_update_batch_appendage_angles(self, batch_appendage, batch_link1_vector, batch_link2_vector):
        # extract angles from predefined vectors (inverse kinematics), and then update appendage angles (forward kinematics)
        # link1
        batch_link1_yaw, batch_link1_pitch = self.get_batch_dh_angle_from_batch_pose_vector(batch_link1_vector, batch_appendage.batch_root_tf)
        batch_appendage.update_batch_link(batch_link1_yaw, batch_link1_pitch, self.batch_zero, self.batch_zero)
        # link2
        batch_link2_yaw, batch_link2_pitch = self.get_batch_dh_angle_from_batch_pose_vector(batch_link2_vector, batch_appendage.batch_link1_tf)
        batch_appendage.update_batch_link(batch_link1_yaw, batch_link1_pitch, batch_link2_yaw, batch_link2_pitch)    
        return batch_link1_yaw, batch_link1_pitch, batch_link2_yaw, batch_link2_pitch
        
    def get_batch_dh_angle_from_batch_pose_vector(self, batch_vec, batch_root_tf):
        return self.calculate_batch_azimuth_elevation(batch_vec, batch_root_tf[:, :, :3, :3]) # yaw, pitch
    
    def calculate_batch_azimuth_elevation(self, batch_vector, batch_root_R, degrees=False):
        batch_vector = (batch_root_R.transpose(2, 3) @ batch_vector.unsqueeze(-1)).squeeze(-1)
        batch_x, batch_y, batch_z = batch_vector[:, :, 0], batch_vector[:, :, 1], batch_vector[:, :, 2]
        batch_azimuth = torch.atan2(batch_y, batch_x)
        batch_elevation = torch.atan2(batch_z, torch.sqrt(batch_x**2 + batch_y**2))
        if degrees:
            return torch.rad2deg(batch_azimuth), torch.rad2deg(batch_elevation)
        else:
            return batch_azimuth, batch_elevation  # Converting to degrees for readability
    
    # -------------------------------------------------------------------------------------------
    def set_dh_model_from_dhdst_output(self, torso_output, right_arm_output, left_arm_output, right_leg_output, left_leg_output, head_output=None):
        # set keypoints from dhdst output
        self.set_batch_torso(torso_output)
        # set dh angles from dhdst output
        self.batch_right_upper_arm_yaw   = right_arm_output[:, :, 0]
        self.batch_right_upper_arm_pitch = right_arm_output[:, :, 1]
        self.batch_right_lower_arm_yaw   = right_arm_output[:, :, 2]
        self.batch_right_lower_arm_pitch = right_arm_output[:, :, 3]
        self.batch_left_upper_arm_yaw    = left_arm_output[:, :, 0]
        self.batch_left_upper_arm_pitch  = left_arm_output[:, :, 1]
        self.batch_left_lower_arm_yaw    = left_arm_output[:, :, 2]
        self.batch_left_lower_arm_pitch  = left_arm_output[:, :, 3]
        self.batch_right_upper_leg_yaw   = right_leg_output[:, :, 0]
        self.batch_right_upper_leg_pitch = right_leg_output[:, :, 1]
        self.batch_right_lower_leg_yaw   = right_leg_output[:, :, 2]
        self.batch_right_lower_leg_pitch = right_leg_output[:, :, 3]
        self.batch_left_upper_leg_yaw    = left_leg_output[:, :, 0]
        self.batch_left_upper_leg_pitch  = left_leg_output[:, :, 1]
        self.batch_left_lower_leg_yaw    = left_leg_output[:, :, 2]
        self.batch_left_lower_leg_pitch  = left_leg_output[:, :, 3]
        # set dh lengths from dhdst output
        self.batch_right_upper_arm_length = right_arm_output[:, :, 4] 
        self.batch_right_lower_arm_length = right_arm_output[:, :, 5]
        self.batch_left_upper_arm_length  = left_arm_output[:, :, 4]
        self.batch_left_lower_arm_length  = left_arm_output[:, :, 5] 
        self.batch_right_upper_leg_length = right_leg_output[:, :, 4]
        self.batch_right_lower_leg_length = right_leg_output[:, :, 5]
        self.batch_left_upper_leg_length  = left_leg_output[:, :, 4]
        self.batch_left_lower_leg_length  = left_leg_output[:, :, 5] 
        # get body reference frame
        self.set_batch_body_reference_frame()
        # generate appendages
        self.generate_all_batch_appendages()
        # update appendages from dh angles and lengths
        self.set_batch_appendage_from_batch_angles()
    
    def set_batch_torso(self, batch_torso):
        assert batch_torso.shape == (self.batch_size, self.num_frames, 9, 3), 'batch_torso should be (batch_size, num_frames, 9, 3)'
        self.batch_pelvis_origin     = batch_torso[:, :, 0]
        self.batch_r_hip_origin      = batch_torso[:, :, 1]
        self.batch_l_hip_origin      = batch_torso[:, :, 2]
        self.batch_torso_origin      = batch_torso[:, :, 3]
        self.batch_neck_origin       = batch_torso[:, :, 4]
        self.batch_nose_origin       = batch_torso[:, :, 5]
        self.batch_head_origin       = batch_torso[:, :, 6]
        self.batch_l_shoulder_origin = batch_torso[:, :, 7]
        self.batch_r_shoulder_origin = batch_torso[:, :, 8]
        
    def set_batch_angles(self, batch_angles, by_dict=False, degree=False):
        # batch_angles: [B, F, 16] or [B, F, 20] (with head)
        if by_dict:
            if self.head:
                self.batch_lower_head_yaw, self.batch_lower_head_pitch = batch_angles['h_l1_yaw'], batch_angles['h_l1_pitch']
                self.batch_upper_head_yaw, self.batch_upper_head_pitch = batch_angles['h_l2_yaw'], batch_angles['h_l2_pitch']
            self.batch_right_upper_arm_yaw, self.batch_right_upper_arm_pitch = batch_angles['ra_l1_yaw'], batch_angles['ra_l1_pitch']
            self.batch_right_lower_arm_yaw, self.batch_right_lower_arm_pitch = batch_angles['ra_l2_yaw'], batch_angles['ra_l2_pitch']
            self.batch_left_upper_arm_yaw,  self.batch_left_upper_arm_pitch  = batch_angles['la_l1_yaw'], batch_angles['la_l1_pitch']
            self.batch_left_lower_arm_yaw,  self.batch_left_lower_arm_pitch  = batch_angles['la_l2_yaw'], batch_angles['la_l2_pitch']
            self.batch_right_upper_leg_yaw, self.batch_right_upper_leg_pitch = batch_angles['rl_l1_yaw'], batch_angles['rl_l1_pitch']
            self.batch_right_lower_leg_yaw, self.batch_right_lower_leg_pitc  = batch_angles['rl_l2_yaw'], batch_angles['rl_l2_pitch']
            self.batch_left_upper_leg_yaw,  self.batch_left_upper_leg_pitch  = batch_angles['ll_l1_yaw'], batch_angles['ll_l1_pitch']
            self.batch_left_lower_leg_yaw,  self.batch_left_lower_leg_pitch  = batch_angles['ll_l2_yaw'], batch_angles['ll_l2_pitch']
        else:
            offset = 0
            if self.head:
                self.batch_upper_head_yaw, self.batch_upper_head_pitch = batch_angles[:, :, 0:2]
                self.batch_lower_head_yaw, self.batch_lower_head_pitch = batch_angles[:, :, 2:4]
                offset = 4
            self.batch_right_upper_arm_yaw, self.batch_right_upper_arm_pitch = batch_angles[:, :, 0+offset:2+offset]
            self.batch_right_lower_arm_yaw, self.batch_right_lower_arm_pitch = batch_angles[:, :, 2+offset:4+offset]
            self.batch_left_upper_arm_yaw,  self.batch_left_upper_arm_pitch  = batch_angles[:, :, 4+offset:6+offset]
            self.batch_left_lower_arm_yaw,  self.batch_left_lower_arm_pitch  = batch_angles[:, :, 6+offset:8+offset]
            self.batch_right_upper_leg_yaw, self.batch_right_upper_leg_pitch = batch_angles[:, :, 8+offset:10+offset]
            self.batch_right_lower_leg_yaw, self.batch_right_lower_leg_pitc  = batch_angles[:, :, 10+offset:12+offset]
            self.batch_left_upper_leg_yaw,  self.batch_left_upper_leg_pitch  = batch_angles[:, :, 12+offset:14+offset]
            self.batch_left_lower_leg_yaw,  self.batch_left_lower_leg_pitch  = batch_angles[:, :, 14+offset:16+offset]
        # convert to radian
        if degree: 
            if self.head:
                self.batch_upper_head_yaw, self.batch_upper_head_pitch = torch.deg2rad(self.batch_upper_head_yaw), torch.deg2rad(self.batch_upper_head_pitch)
                self.batch_lower_head_yaw, self.batch_lower_head_pitch = torch.deg2rad(self.batch_lower_head_yaw), torch.deg2rad(self.batch_lower_head_pitch)
            self.batch_right_upper_arm_yaw, self.batch_right_upper_arm_pitch = torch.deg2rad(self.batch_right_upper_arm_yaw), torch.deg2rad(self.batch_right_upper_arm_pitch)
            self.batch_right_lower_arm_yaw, self.batch_right_lower_arm_pitch = torch.deg2rad(self.batch_right_lower_arm_yaw), torch.deg2rad(self.batch_right_lower_arm_pitch)
            self.batch_left_upper_arm_yaw,  self.batch_left_upper_arm_pitch  = torch.deg2rad(self.batch_left_upper_arm_yaw),  torch.deg2rad(self.batch_left_upper_arm_pitch)
            self.batch_left_lower_arm_yaw,  self.batch_left_lower_arm_pitch  = torch.deg2rad(self.batch_left_lower_arm_yaw),  torch.deg2rad(self.batch_left_lower_arm_pitch)
            self.batch_right_upper_leg_yaw, self.batch_right_upper_leg_pitch = torch.deg2rad(self.batch_right_upper_leg_yaw), torch.deg2rad(self.batch_right_upper_leg_pitch)
            self.batch_right_lower_leg_yaw, self.batch_right_lower_leg_pitc  = torch.deg2rad(self.batch_right_lower_leg_yaw), torch.deg2rad(self.batch_right_lower_leg_pitc)
            self.batch_left_upper_leg_yaw,  self.batch_left_upper_leg_pitch  = torch.deg2rad(self.batch_left_upper_leg_yaw),  torch.deg2rad(self.batch_left_upper_leg_pitch)
            self.batch_left_lower_leg_yaw,  self.batch_left_lower_leg_pitch  = torch.deg2rad(self.batch_left_lower_leg_yaw),  torch.deg2rad(self.batch_left_lower_leg_pitch)
            
    def set_batch_appendage_from_batch_angles(self):
        if self.head:
            self.batch_head.update_batch_link(self.batch_lower_head_yaw, self.batch_lower_head_pitch, self.batch_upper_head_yaw, self.batch_upper_head_pitch)
        self.batch_right_arm.update_batch_link(self.batch_right_upper_arm_yaw, self.batch_right_upper_arm_pitch, self.batch_right_lower_arm_yaw, self.batch_right_lower_arm_pitch)
        self.batch_left_arm.update_batch_link(self.batch_left_upper_arm_yaw, self.batch_left_upper_arm_pitch, self.batch_left_lower_arm_yaw, self.batch_left_lower_arm_pitch)
        self.batch_right_leg.update_batch_link(self.batch_right_upper_leg_yaw, self.batch_right_upper_leg_pitch, self.batch_right_lower_leg_yaw, self.batch_right_lower_leg_pitch)
        self.batch_left_leg.update_batch_link(self.batch_left_upper_leg_yaw, self.batch_left_upper_leg_pitch, self.batch_left_lower_leg_yaw, self.batch_left_lower_leg_pitch)    
        
    # -------------------------------------------------------------------------------------------
    # get functions
    def get_batch_limb_length(self, by_dict=False):
        if by_dict:
            batch_limb_length = {}
            batch_limb_length['ra_l1']    = self.batch_right_upper_arm_length
            batch_limb_length['ra_l2']    = self.batch_right_lower_arm_length
            batch_limb_length['la_l1']    = self.batch_left_upper_arm_length
            batch_limb_length['la_l2']    = self.batch_left_lower_arm_length
            batch_limb_length['rl_l1']    = self.batch_right_upper_leg_length
            batch_limb_length['rl_l2']    = self.batch_right_lower_leg_length
            batch_limb_length['ll_l1']    = self.batch_left_upper_leg_length
            batch_limb_length['ll_l2']    = self.batch_left_lower_leg_length
        else:
            self.batch_limb_length = torch.cat([self.batch_right_upper_arm_length.unsqueeze(-1),
                                                self.batch_right_lower_arm_length.unsqueeze(-1),
                                                self.batch_left_upper_arm_length.unsqueeze(-1),
                                                self.batch_left_lower_arm_length.unsqueeze(-1),
                                                self.batch_right_upper_leg_length.unsqueeze(-1),
                                                self.batch_right_lower_leg_length.unsqueeze(-1),
                                                self.batch_left_upper_leg_length.unsqueeze(-1),
                                                self.batch_left_lower_leg_length.unsqueeze(-1)], dim=-1)
    
        return batch_limb_length
        
    def get_body_frame(self, batch_num, frame_num):
        pelvis_origin = self.batch_pelvis_origin[batch_num, frame_num].cpu().detach().numpy()
        body_R = self.batch_body_R[batch_num, frame_num].cpu().detach().numpy()
        body_frame = ReferenceFrame(
            origin=pelvis_origin, 
            dx=body_R[0], 
            dy=body_R[1],
            dz=body_R[2],
            name='body',
        )
        return body_frame
    
    def get_batch_dh_angles(self, by_dict=False, degree=False):
        if by_dict:
            batch_dh_angles = {}
            # head
            batch_dh_angles['h_l1_yaw'],  batch_dh_angles['h_l1_pitch']  = self.head.link1_yaw,  self.head.link1_pitch
            # right arm
            batch_dh_angles['ra_l1_yaw'], batch_dh_angles['ra_l1_pitch'] = self.right_arm.link1_yaw, self.right_arm.link1_pitch
            batch_dh_angles['ra_l2_yaw'], batch_dh_angles['ra_l2_pitch'] = self.right_arm.link2_yaw, self.right_arm.link2_pitch
            # left arm
            batch_dh_angles['la_l1_yaw'], batch_dh_angles['la_l1_pitch'] = self.left_arm.link1_yaw, self.left_arm.link1_pitch
            batch_dh_angles['la_l2_yaw'], batch_dh_angles['la_l2_pitch'] = self.left_arm.link2_yaw, self.left_arm.link2_pitch
            # right leg
            batch_dh_angles['rl_l1_yaw'], batch_dh_angles['rl_l1_pitch'] = self.right_leg.link1_yaw, self.right_leg.link1_pitch
            batch_dh_angles['rl_l2_yaw'], batch_dh_angles['rl_l2_pitch'] = self.right_leg.link2_yaw, self.right_leg.link2_pitch
            # left leg
            batch_dh_angles['ll_l1_yaw'], batch_dh_angles['ll_l1_pitch'] = self.left_leg.link1_yaw, self.left_leg.link1_pitch
            batch_dh_angles['ll_l2_yaw'], batch_dh_angles['ll_l2_pitch'] = self.left_leg.link2_yaw, self.left_leg.link2_pitch
            if degree:
                for key in batch_dh_angles.keys():
                    batch_dh_angles[key] = torch.rad2deg(batch_dh_angles[key])
        else:
            batch_dh_angles = np.zeros(18)
            # head
            batch_dh_angles[0:2]   = np.array([self.batch_head.link1_yaw,      self.batch_head.link1_pitch])
            # right arm
            batch_dh_angles[2:6]   = np.array([self.batch_right_arm.link1_yaw, self.batch_right_arm.link1_pitch, self.batch_right_arm.link2_yaw, self.batch_right_arm.link2_pitch])
            # left arm
            batch_dh_angles[6:10]  = np.array([self.batch_left_arm.link1_yaw,  self.batch_left_arm.link1_pitch,  self.batch_left_arm.link2_yaw,  self.batch_left_arm.link2_pitch])
            # right leg
            batch_dh_angles[10:14] = np.array([self.batch_right_leg.link1_yaw, self.batch_right_leg.link1_pitch, self.batch_right_leg.link2_yaw, self.batch_right_leg.link2_pitch])
            # left leg
            batch_dh_angles[14:18] = np.array([self.batch_left_leg.link1_yaw,  self.batch_left_leg.link1_pitch,  self.batch_left_leg.link2_yaw,  self.batch_left_leg.link2_pitch])
            
            if degree:
                batch_dh_angles = torch.rad2deg(batch_dh_angles)
        return batch_dh_angles
    
    def get_batch_appendage_length(self, by_dict=False, head=False):
        if by_dict:
            batch_appendage_length = {}  
            # batch_appendage_length['h_l1']  = self.batch_head.link1_length
            # batch_appendage_length['h_l2']  = self.batch_head.link2_length
            batch_appendage_length['ra_l1'] = self.batch_right_arm.link1_length
            batch_appendage_length['ra_l2'] = self.batch_right_arm.link2_length
            batch_appendage_length['la_l1'] = self.batch_left_arm.link1_length
            batch_appendage_length['la_l2'] = self.batch_left_arm.link2_length
            batch_appendage_length['rl_l1'] = self.batch_right_leg.link1_length
            batch_appendage_length['rl_l2'] = self.batch_right_leg.link2_length
            batch_appendage_length['ll_l1'] = self.batch_left_leg.link1_length
            batch_appendage_length['ll_l2'] = self.batch_left_leg.link2_length
        else:
            batch_appendage_length = torch.cat([self.batch_right_arm.batch_link1_length.unsqueeze(-1), self.batch_right_arm.batch_link2_length.unsqueeze(-1),
                                                self.batch_left_arm.batch_link1_length.unsqueeze(-1),  self.batch_left_arm.batch_link2_length.unsqueeze(-1),
                                                self.batch_right_leg.batch_link1_length.unsqueeze(-1), self.batch_right_leg.batch_link2_length.unsqueeze(-1),
                                                self.batch_left_leg.batch_link1_length.unsqueeze(-1),  self.batch_left_leg.batch_link2_length.unsqueeze(-1)], dim=-1)
        return batch_appendage_length
    
    def get_batch_keypoints(self, by_dict=False):
        if by_dict:
            batch_keypoints = {}
            batch_keypoints['pelvis']     = self.batch_pelvis_origin  # 0
            batch_keypoints['r_hip']      = self.batch_r_hip_origin # 1
            batch_keypoints['l_hip']      = self.batch_l_hip_origin # 2
            batch_keypoints['torso']      = self.batch_torso_origin # 3
            batch_keypoints['neck']       = self.batch_neck_origin # 4
            batch_keypoints['nose']       = self.batch_nose_origin # 5
            batch_keypoints['head']       = self.batch_head_origin # 6
            batch_keypoints['l_shoulder'] = self.batch_l_shoulder_origin  # 7
            batch_keypoints['r_shoulder'] = self.batch_r_shoulder_origin  # 8
        else:
            batch_keypoints = torch.cat([self.batch_pelvis_origin.unsqueeze(-1),
                                         self.batch_r_hip_origin.unsqueeze(-1),
                                         self.batch_l_hip_origin.unsqueeze(-1),
                                         self.batch_torso_origin.unsqueeze(-1),
                                         self.batch_neck_origin.unsqueeze(-1),
                                         self.batch_nose_origin.unsqueeze(-1),
                                         self.batch_head_origin.unsqueeze(-1),
                                         self.batch_l_shoulder_origin.unsqueeze(-1),
                                         self.batch_r_shoulder_origin.unsqueeze(-1)], dim=-1)
        return batch_keypoints
    
    def get_batch_pose_3d(self, device=None):
        batch_pose_3d     = torch.zeros((self.batch_size, self.num_frames, 17, 3)).to(self.device)
        batch_pose_3d[:, :, 0]  = self.batch_pelvis_origin
        batch_pose_3d[:, :, 1]  = self.batch_r_hip_origin
        batch_pose_3d[:, :, 2]  = self.batch_right_leg.batch_link2_origin 
        batch_pose_3d[:, :, 3]  = self.batch_right_leg.batch_terminal_origin 
        batch_pose_3d[:, :, 4]  = self.batch_l_hip_origin
        batch_pose_3d[:, :, 5]  = self.batch_left_leg.batch_link2_origin 
        batch_pose_3d[:, :, 6]  = self.batch_left_leg.batch_terminal_origin 
        batch_pose_3d[:, :, 7]  = self.batch_torso_origin
        batch_pose_3d[:, :, 8]  = self.batch_neck_origin
        batch_pose_3d[:, :, 9]  = self.batch_nose_origin # self.batch_head.batch_link2_origin
        batch_pose_3d[:, :, 10] = self.batch_head_origin # self.batch_head.batch_terminal_origin
        batch_pose_3d[:, :, 11] = self.batch_l_shoulder_origin
        batch_pose_3d[:, :, 12] = self.batch_left_arm.batch_link2_origin
        batch_pose_3d[:, :, 13] = self.batch_left_arm.batch_terminal_origin
        batch_pose_3d[:, :, 14] = self.batch_r_shoulder_origin
        batch_pose_3d[:, :, 15] = self.batch_right_arm.batch_link2_origin
        batch_pose_3d[:, :, 16] = self.batch_right_arm.batch_terminal_origin
        return batch_pose_3d
    
    # -------------------------------------------------------------------------------------------
    # result functions
    def batch_mpjpe(self, batch_gt):
        assert batch_gt.shape == (self.batch_size, self.num_frames, 17, 3), 'batch_gt shape should be ({}, {}, 17, 3)'.format(self.batch_size, self.num_frames)
        if type(batch_gt) != torch.Tensor: batch_gt = torch.tensor(batch_gt, dtype=self.data_type)
        batch_gt = batch_gt.to(self.device)
        return torch.mean(torch.norm(self.get_batch_pose_3d() - batch_gt, dim=-1))
    
    def draw(self, ax, batch_num, frame_num, draw_frame=False, draw_gt=False, head_length=0.01, scale=0.1, fontsize=10, show_name=False):
        if draw_frame:
            body_frame = self.get_body_frame(batch_num, frame_num)
            body_frame.draw3d(color='tab:orange', head_length=head_length, scale=scale, show_name=show_name)
        if self.head:
            self.batch_head.draw(ax, batch_num, frame_num, draw_frame, head_length, scale, fontsize, show_name)
        self.batch_right_arm.draw(ax, batch_num, frame_num, draw_frame, head_length, scale, fontsize, show_name)
        self.batch_left_arm.draw(ax, batch_num, frame_num, draw_frame, head_length, scale, fontsize, show_name)
        self.batch_right_leg.draw(ax, batch_num, frame_num, draw_frame, head_length, scale, fontsize, show_name)
        self.batch_left_leg.draw(ax, batch_num, frame_num, draw_frame, head_length, scale, fontsize, show_name)
        if draw_gt:
            batch_gt = self.get_batch_pose_3d().cpu().detach().numpy()
            pose = batch_gt[batch_num, frame_num]
            draw_3d_pose(ax, pose)
