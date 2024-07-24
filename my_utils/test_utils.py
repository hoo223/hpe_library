from lib_import import *

#plt.switch_backend('TkAgg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def readJSON(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def readpkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def savepkl(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

# return keypoint index from keypoint name (h36m)
def get_h36m_keypoint_index(keypoint_name):
    for key, value in h36m_keypoints.items():
        if value.lower() == keypoint_name.lower():
            return key
    assert False, 'Invalid keypoint name: {}'.format(keypoint_name)
    
# return keypoints from keypoint list (h36m)
def get_h36m_keypoints(pose3d, key_list=[]):
    assert pose3d.shape == (17, 3), 'pose3d shape is wrong'
    output = []
    for key in key_list:
        idx = get_h36m_keypoint_index(key)
        output.append(pose3d[idx])
    return output

def get_batch_h36m_keypoints(batch_pose3d, key_list=[]):
    assert type(batch_pose3d) == torch.Tensor, 'batch_pose3d should be torch.Tensor'
    if len(batch_pose3d.shape) == 3:
        output = torch.zeros([batch_pose3d.shape[0], len(key_list), 3], dtype=torch.float32).to(batch_pose3d.device)
    elif len(batch_pose3d.shape) == 4:
        output = torch.zeros([batch_pose3d.shape[0], batch_pose3d.shape[1], len(key_list), 3], dtype=torch.float32).to(batch_pose3d.device)
    for i, key in enumerate(key_list):
        key_idx = get_h36m_keypoint_index(key)
        if len(batch_pose3d.shape) == 3:
            output[:, i] = batch_pose3d[:, key_idx]
        elif len(batch_pose3d.shape) == 4:
            output[:, :, i] = batch_pose3d[:, :, key_idx]
    return output

h36m_keypoints ={
    0 : 'Pelvis',
    1 : 'R_Hip',
    2 : 'R_Knee',
    3 : 'R_Ankle',
    4 : 'L_Hip',
    5 : 'L_Knee',
    6 : 'L_Ankle',
    7 : 'Torso',
    8 : 'Neck',
    9 : 'Nose',
    10 : 'Head',
    11 : 'L_Shoulder',
    12 : 'L_Elbow',
    13 : 'L_Wrist',
    14 : 'R_Shoulder',
    15 : 'R_Elbow',
    16 : 'R_Wrist',
}

h36m_connections = [
    ('Pelvis','R_Hip'),
    ('Pelvis','L_Hip'),
    ('Pelvis','Torso'),
    ('R_Hip','R_Knee'),
    ('R_Knee','R_Ankle'),
    ('L_Hip','L_Knee'),
    ('L_Knee','L_Ankle'),
    ('Torso','Neck'),
    ('Neck','Nose'),
    ('Nose','Head'),
    ('Neck','R_Shoulder'),
    ('R_Shoulder','R_Elbow'),
    ('R_Elbow','R_Wrist'),
    ('Neck','L_Shoulder'),
    ('L_Shoulder','L_Elbow'),
    ('L_Elbow','L_Wrist')
]

len_ids = {
    'R_HIP' : 0,
    'R_UPPER_LEG' : 1,
    'R_LOWER_LEG' : 2,
    'L_HIP' : 3,
    'L_UPPER_LEG' : 4,
    'L_LOWER_LEG' : 5,
    'LOWER_TORSO' : 6,
    'UPPER_TORSO' : 7,
    'LOWER_FACE' : 8,
    'UPPER_FACE' : 9,
    'L_SHOULDER' : 10,
    'L_UPPER_ARM' : 11,
    'L_LOWER_ARM' : 12,
    'R_SHOULDER' : 13,
    'R_UPPER_ARM' : 14,
    'R_LOWER_ARM' : 15
}


# len_id_names = {
#     0: 'R_HIP',
#     1: 'R_UPPER_LEG',
#     2: 'R_LOWER_LEG',
#     3: 'L_HIP',
#     4: 'L_UPPER_LEG',
#     5: 'L_LOWER_LEG',
#     6: 'LOWER_TORSO',
#     7: 'UPPER_TORSO',
#     8: 'LOWER_FACE',
#     9: 'UPPER_FACE',
#     10: 'L_SHOULDER',
#     11: 'L_UPPER_ARM',
#     12: 'L_LOWER_ARM',
#     13: 'R_SHOULDER',
#     14: 'R_UPPER_ARM',
#     15: 'R_LOWER_ARM'
# }

part_ids = {
    'R_ARM': 0,
    'L_ARM': 1,
    'R_LEG': 2,
    'L_LEG': 3,
    'TORSO_SMALL': 4,
    'TORSO_FULL': 5
}

# part_id_names = {
#     0: 'R_ARM',
#     1: 'L_ARM',
#     2: 'R_LEG',
#     3: 'L_LEG',
#     4: 'TORSO_SMALL',
#     5: 'TORSO_FULL'
# }

h36m_part_keypoints = [
    [14, 15, 16], 
    [11, 12, 13], 
    [1, 2, 3], 
    [4, 5, 6], 
    [0, 1, 4, 8, 11, 14], 
    [0, 1, 4, 7, 8, 9, 10, 11, 14]
]

def get_h36m_limb_lens(x):
    '''
        Input: (F, 17, 3)
        Output: (F, 16)
    '''
    #assert type(x) == np.ndarray, 'x should be np.ndarray'
    if type(x) == torch.Tensor: x = x.cpu().detach().numpy()
    if len(x.shape) == 2:
        V, C = x.shape
        T = 1
        x = x.reshape(T, V, C)
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    limbs = x[:,limbs_id,:] # (F, 16, 2, 3)
    limbs = limbs[:,:,0,:]-limbs[:,:,1,:] # (F, 16, 3)
    limb_lens = np.linalg.norm(limbs, axis=-1) # (F, 16)
    return limb_lens

# h36m_connections = [
#     (0,1),
#     (0,4),
#     (0,7),
#     (1,2),
#     (2,3),
#     (4,5),
#     (5,6),
#     (7,8),
#     (8,9),
#     (9,10),
#     (8,14),
#     (14,15),
#     (15,16),
#     (8,11),
#     (11,12),
#     (12,13)
# ]

kookmin_keypoints = {
    1 : 'Head-top',
    2 : 'Neck',
    3 : 'Spine',
    4 : 'Shoulder-L',
    5 : 'Shoulder-R',
    6 : 'Elbow-L',
    7 : 'Elbow-R',
    8 : 'Wrist-L',
    9 : 'Wrist-R',
    10 : 'Middle Finger Tip-L',
    11 : 'Middle Finger Tip-R',
    12 : 'Upper Hip-L',
    13 : 'Upper Hip-R',
    14 : 'Under Hip-L',
    15 : 'Under Hip-R',
    16 : 'Knee-L',
    17 : 'Knee-R',
    18 : 'Lateral malleolus-L',
    19 : 'Lateral malleolus-R',
    20 : 'Medial malleolus-L',
    21 : 'Medial malleolus-R',
    22 : 'Little Toe-L',
    23 : 'Little Toe-R',
    24 : 'Hallux Toe-L',
    25 : 'Hallux Toe-R'
}

kookmin_connections = [
    [0, 1], 
    [1, 2], 
    [1, 3], 
    [1, 4], 
    [3, 5], 
    [5, 7], 
    [7, 9], 
    [4, 6], 
    [6, 8], 
    [8, 10], 
    [2, 11], 
    [2, 12], 
    [11, 13], 
    [12, 14], 
    [13, 14], 
    [13, 15], 
    [15, 17], 
    [15, 19], 
    [17, 19], 
    [17, 21], 
    [19, 23], 
    [14, 16], 
    [16, 18], 
    [16, 20], 
    [18, 20],
    [18, 22],
    [20, 24]
]

kookmin2_keypoints = {
    1  : 'Head-top',             # 0
    2  : 'Nose',                 # 1 
    3  : 'Neck',                 # 2 
    4  : 'Spine',                # 3  
    5  : 'Shoulder-L',           # 4
    6  : 'Shoulder-R',           # 5 
    7  : 'Elbow-L',              # 6 
    8  : 'Elbow-R',              # 7 
    9  : 'Wrist-L',              # 8 
    10 : 'Wrist-R',              # 9 
    11 : 'Middle Finger Tip-L',  # 10
    12 : 'Middle Finger Tip-R',  # 11
    13 : 'Upper Hip-L',          # 12
    14 : 'Upper Hip-R',          # 13
    15 : 'Under Hip-L',          # 14
    16 : 'Under Hip-R',          # 15
    17 : 'Knee-L',               # 16
    18 : 'Knee-R',               # 17
    19 : 'Lateral malleolu18s-L',   # 18
    20 : 'Lateral malleolus-R',     # 19
    21 : 'Medial malleolus-L',      # 20
    22 : 'Medial malleolus-R',      # 21
    23 : 'Little Toe-L',         # 22
    24 : 'Little Toe-R',         # 23
    25 : 'Hallux Toe-L',         # 24
    26 : 'Hallux Toe-R',         # 25
}

kookmin2_connections = [
    [0, 1],   # Head-top -> Nose
    [1, 2],   # Nose -> Neck
    [2, 3],   # Neck -> Spine
    [2, 5],   # Neck -> Shoulder-R
    [5, 7],   # Shoulder-R -> Elbow-R
    [7, 9],   # Elbow-R -> Wrist-R
    [9, 11],  # Wrist-R -> Middle Finger Tip-R
    [2, 4],   # Neck -> Shoulder-L
    [4, 6],   # Shoulder-L -> Elbow-L
    [6, 8],   # Elbow-L -> Wrist-L
    [8, 10],  # Wrist-L -> Middle Finger Tip-L
    [3, 12],  # Spine -> Upper Hip-L
    [3, 13],  # Spine -> Upper Hip-R
    [12, 14], # Upper Hip-L -> Under Hip-L
    [13, 15], # Upper Hip-R -> Under Hip-R
    [14, 15], # Under Hip-L -> Under Hip-R
    [14, 16], # Under Hip-L -> Knee-L
    [16, 18], # Knee-L -> Lateral malleolus-L
    [16, 20], # Knee-L -> Medial malleolus-L
    [18, 20], # Lateral malleolus-L -> Medial malleolus-L
    [18, 22], # Lateral malleolus-L -> Little Toe-L
    [20, 24], # Medial malleolus-L -> Hallux Toe-L
    [15, 17], # Under Hip-R -> Knee-R
    [17, 19], # Knee-R -> Lateral malleolus-R
    [17, 21], # Knee-R -> Medial malleolus-R
    [19, 21], # Lateral malleolus-R -> Medial malleolus-R
    [19, 23], # Lateral malleolus-R -> Little Toe-R
    [21, 25]  # Medial malleolus-R -> Hallux Toe-R
]

def mpi_inf_3dhp_original2posynda(x):
    # x: 3D pose (T x V x C) or (V x C)
    '''
    posynda kp 0  <- original kp 7
    posynda kp 1  <- original kp 5
    posynda kp 2  <- original kp 14
    posynda kp 3  <- original kp 15
    posynda kp 4  <- original kp 16
    posynda kp 5  <- original kp 9
    posynda kp 6  <- original kp 10
    posynda kp 7  <- original kp 11
    posynda kp 8  <- original kp 23
    posynda kp 9  <- original kp 24
    posynda kp 10 <- original kp 25
    posynda kp 11 <- original kp 18
    posynda kp 12 <- original kp 19
    posynda kp 13 <- original kp 20
    posynda kp 14 <- original kp 4
    posynda kp 15 <- original kp 3
    posynda kp 16 <- original kp 6
    '''
    if len(x.shape) == 2:
        V, C = x.shape
        T = 1
        x = x.reshape(T, V, C)
    else:
        T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,7,:]  # Pelvis
    y[:,1,:] = x[:,5,:]  # R_Hip
    y[:,2,:] = x[:,14,:]  # R_Knee
    y[:,3,:] = x[:,15,:]  # R_Ankle
    y[:,4,:] = x[:,16,:]  # L_Hip
    y[:,5,:] = x[:,9,:]  # L_Knee
    y[:,6,:] = x[:,10,:]  # L_Ankle
    y[:,7,:] = x[:,11,:]  # Torso
    y[:,8,:] = x[:,23,:]  # Neck
    y[:,9,:] = x[:,24,:]  # Nose
    y[:,10,:] = x[:,25,:]  # Head
    y[:,11,:] = x[:,18,:]  # L_Shoulder
    y[:,12,:] = x[:,19,:]  # L_Elbow
    y[:,13,:] = x[:,20,:]  # L_Wrist
    y[:,14,:] = x[:,4,:]  # R_Shoulder
    y[:,15,:] = x[:,3,:]  # R_Elbow
    y[:,16,:] = x[:,6,:]  # R_Wrist
    return y
    

# https://github.com/chaneyddtt/Generating-Multiple-Hypotheses-for-3D-Human-Pose-Estimation-with-Mixture-Density-Network/issues/12
def mpi_inf_3dhp2h36m(x):
    '''
    Input: x (T x V x C)  
    //mpi_inf_3dhp 17 body keypoints
    0'Right Ankle', 3
    1'Right Knee', 2
    2'Right Hip', 1
    3'Left Hip', 4
    4'Left Knee', 5
    5'Left Ankle', 6
    6'Right Wrist', 15
    7'Right Elbow', 14
    8'Right Shoulder', 13
    9'Left Shoulder', 10
    10'Left Elbow', 11
    11'Left Wrist', 12
    12'Neck', 8 
    13'Top of Head', 9
    14'Pelvis)', 0
    15'Thorax', 7 
    16'Spine', mpi3d
    17'Jaw', mpi3d
    18'Head', mpi3d
    '''
    if len(x.shape) == 2:
        V, C = x.shape
        T = 1
        x = x.reshape(T, V, C)
    else:
        T, V, C = x.shape
    
    y = np.zeros([T,17,C])
    if V == 17: # posynda mpi_inf_3dhp
        pass
    elif V == 28: # original mpi_inf_3dhp
        x = mpi_inf_3dhp_original2posynda(x)
        assert x.shape == (T, 17, C), 'x shape is wrong'
    y[:,0,:] = x[:,14,:]  # Pelvis
    y[:,1,:] = x[:,8,:]   # R_Hip
    y[:,2,:] = x[:,9,:]   # R_Knee
    y[:,3,:] = x[:,10,:]   # R_Ankle
    y[:,4,:] = x[:,11,:]   # L_Hip
    y[:,5,:] = x[:,12,:]   # L_Knee
    y[:,6,:] = x[:,13,:]   # L_Ankle
    y[:,7,:] = x[:,15,:]   # Torso
    y[:,8,:] = x[:,1,:]   # Neck
    y[:,9,:] = x[:,16,:]  # Nose
    y[:,10,:] = x[:,0,:] # Head
    y[:,11,:] = x[:,5,:] # L_Shoulder
    y[:,12,:] = x[:,6,:] # L_Elbow
    y[:,13,:] = x[:,7,:] # L_Wrist
    y[:,14,:] = x[:,2,:] # R_Shoulder
    y[:,15,:] = x[:,3,:] # R_Elbow
    y[:,16,:] = x[:,4,:] # R_Wrist
    
    return y

def kookmin2h36m(x):
    '''
    Input: x (T x V x C)  
    //kookmin 25 body keypoints
    1 : 'Head-top',
    2 : 'Neck',
    3 : 'Spine',
    4 : 'Shoulder-L',
    5 : 'Shoulder-R',
    6 : 'Elbow-L',
    7 : 'Elbow-R',
    8 : 'Wrist-L',
    9 : 'Wrist-R',
    10 : 'Middle Finger Tip-L',
    11 : 'Middle Finger Tip-R',
    12 : 'Upper Hip-L',
    13 : 'Upper Hip-R',
    14 : 'Under Hip-L',
    15 : 'Under Hip-R',
    16 : 'Knee-L',
    17 : 'Knee-R',
    18 : 'Lateral malleolus-L',
    19 : 'Lateral malleolus-R',
    20 : 'Medial malleolus-L',
    21 : 'Medial malleolus-R',
    22 : 'Little Toe-L',
    23 : 'Little Toe-R',
    24 : 'Hallux Toe-L',
    25 : 'Hallux Toe-R'
    '''
    if len(x.shape) == 2:
        V, C = x.shape
        T = 1
        x = x.reshape(T, V, C)
    else:
        T, V, C = x.shape
    center = lambda pose, x, y: (pose[:,x,:] + pose[:,y,:]) * 0.5
    y = np.zeros([T,17,C])
    y[:,0,:] = center(x, 13, 14) # Pelvis
    y[:,1,:] = x[:,14,:] # R_Hip
    y[:,2,:] = x[:,16,:] # R_Knee
    y[:,3,:] = center(x, 18, 20) # R_Ankle
    y[:,4,:] = x[:,13,:] # L_Hip
    y[:,5,:] = x[:,15,:] # L_Knee
    y[:,6,:] = center(x, 17, 19) # L_Ankle
    y[:,7,:] = x[:,2,:] # Torso
    y[:,8,:] = x[:,1,:] # Neck
    y[:,9,:] = center(x, 0, 1) # Nose
    y[:,10,:] = x[:,0,:] # Head
    y[:,11,:] = x[:,3,:] # L_Shoulder
    y[:,12,:] = x[:,5,:] # L_Elbow
    y[:,13,:] = x[:,7,:] # L_Wrist
    y[:,14,:] = x[:,4,:] # R_Shoulder
    y[:,15,:] = x[:,6,:] # R_Elbow
    y[:,16,:] = x[:,8,:] # R_Wrist
    return y

def kookmin2h36m_with_nose(x):
    '''
    Input: x (T x V x C)  
    //kookmin 25 body keypoints
    0  : 'Head-top',
    1  : 'Nose',
    2  : 'Neck',
    3  : 'Spine',
    4  : 'Shoulder-L',
    5  : 'Shoulder-R',
    6  : 'Elbow-L',
    7  : 'Elbow-R',
    8  : 'Wrist-L',
    9  : 'Wrist-R',
    10 : 'Middle Finger Tip-L',
    11 : 'Middle Finger Tip-R',
    12 : 'Upper Hip-L',
    13 : 'Upper Hip-R',
    14 : 'Under Hip-L',
    15 : 'Under Hip-R',
    16 : 'Knee-L',
    17 : 'Knee-R',
    18 : 'Lateral malleolus-L',
    19 : 'Lateral malleolus-R',
    20 : 'Medial malleolus-L',
    21 : 'Medial malleolus-R',
    22 : 'Little Toe-L',
    23 : 'Little Toe-R',
    24 : 'Hallux Toe-L',
    25 : 'Hallux Toe-R'
    '''
    if len(x.shape) == 2:
        V, C = x.shape
        T = 1
        x = x.reshape(T, V, C)
    else:
        T, V, C = x.shape
    center = lambda pose, x, y: (pose[:,x,:] + pose[:,y,:]) * 0.5
    y = np.zeros([T,17,C])
    y[:,0,:] = center(x, 14, 15) # Pelvis
    y[:,1,:] = x[:,15,:] # R_Hip
    y[:,2,:] = x[:,17,:] # R_Knee
    y[:,3,:] = center(x, 19, 21) # x[:,21,:] # center(x, 19, 21) # R_Ankle
    y[:,4,:] = x[:,14,:] # L_Hip
    y[:,5,:] = x[:,16,:] # L_Knee
    y[:,6,:] = center(x, 18, 20) # x[:,20,:] # center(x, 18, 20) # L_Ankle
    y[:,7,:] = x[:,3,:] # Torso
    y[:,8,:] = x[:,2,:] # Neck
    y[:,9,:] = x[:,1,:] # Nose
    y[:,10,:] = x[:,0,:] # Head
    y[:,11,:] = x[:,4,:] # L_Shoulder
    y[:,12,:] = x[:,6,:] # L_Elbow
    y[:,13,:] = x[:,8,:] # L_Wrist
    y[:,14,:] = x[:,5,:] # R_Shoulder
    y[:,15,:] = x[:,7,:] # R_Elbow
    y[:,16,:] = x[:,9,:] # R_Wrist
    return y

fit3d_connections = [[10, 9], [9, 8], [8, 11], [8, 14], [11, 12], [14, 15], [12, 13], [15, 16],
                    [8, 7], [7, 0], [0, 1], [0, 4], [1, 2], [4, 5], [2, 3], [5, 6],
                    [13, 21], [13, 22], [16, 23], [16, 24], [3, 17], [3, 18], [6, 19], [6, 20]]

aihub_keypoints = {
    0:  "Pelvis",
    1:  "L_Hip",
    2:  "R_Hip",
    3:  "Spine1",
    4:  "L_Knee",
    5:  "R_Knee",
    6:  "Spine2",
    7:  "L_Ankle",
    8:  "R_Ankle",
    9:  "Spine3",
    10: "L_Foot",
    11: "R_Foot",
    12: "Neck",
    13: "L_Collar",
    14: "R_Collar",
    15: "Head",
    16: "L_Shoulder",
    17: "R_Shoulder",
    18: "L_Elbow",
    19: "R_Elbow",
    20: "L_Wrist",
    21: "R_Wrist",
    22: "L_Hand",
    23: "R_Hand"
}

aihub_connections = [
    ("Head", "Neck"),
    ("Neck", "R_Shoulder"),
    ("Neck", "L_Shoulder"),
    ("R_Shoulder", "R_Elbow"),
    ("R_Elbow", "R_Wrist"),
    ("L_Shoulder", "L_Elbow"),
    ("L_Elbow", "L_Wrist"),
    ("Neck", "Pelvis"),
    ("Pelvis", "L_Hip"),
    ("Pelvis", "R_Hip"),
    ("R_Hip", "L_Hip"),
    ("R_Hip", "R_Knee"),
    ("R_Knee", "R_Ankle"),
    ("L_Hip", "L_Knee"),
    ("L_Knee", "L_Ankle"),
]




# aihub_connections = [
#     [15,12],
#     [12,17],
#     [12,16],
#     [17,19],
#     [19,21],
#     [16,18],
#     [18,20],
#     [12,0],
#     [0,1],
#     [0,2],
#     [2,1],
#     [2,5],
#     [5,8],
#     [1,4],
#     [4,7],
# ]

# lib/data/dataset_wild.py
def halpe2h36m(x):
    '''
        Input: x (T x V x C)  
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    '''
    if len(x.shape) == 2:
        V, C = x.shape
        T = 1
        x = x.reshape(T, V, C)
    else:
        T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,19,:]
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    y[:,8,:] = x[:,18,:]
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,17,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y

def aihub2h36m(x, mode='3d'):
    '''
        Input: x (T x V x C)  
       //aihub 24 body keypoints
    {0, "Pelvis"},
    {1, "L_Hip"},
    {2, "R_Hip"},
    {3, "Spine1"},
    {4, "L_Knee"},
    {5, "R_Knee"},
    {6, "Spine2"},
    {7, "L_Ankle"},
    {8, "R_Ankle"},
    {9, "Spine3"},
    {10, "L_Foot"},
    {11, "R_Foot"},
    {12, "Neck"},
    {13, "L_Collar"},
    {14, "R_Collar"},
    {15, "Head"},
    {16, "L_Shoulder"},
    {17, "R_Shoulder"},
    {18, "L_Elbow"},
    {19, "R_Elbow"},
    {20, "L_Wrist"},
    {21, "R_Wrist"},
    {22, "L_Hand"},
    {23, "R_Hand"}
    '''
    if len(x.shape) == 2:
        V, C = x.shape
        T = 1
        x = x.reshape(T, V, C)
    else:
        T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = (x[:,1,:] + x[:,2,:]) * 0.5 # Pelvis
    y[:,1,:] = x[:,2,:] # R_Hip
    y[:,2,:] = x[:,5,:] # R_Knee
    y[:,3,:] = x[:,8,:] # R_Ankle
    y[:,4,:] = x[:,1,:] # L_Hip
    y[:,5,:] = x[:,4,:] # L_Knee
    y[:,6,:] = x[:,7,:] # L_Ankle
    y[:,7,:] = x[:,6,:] # Torso
    y[:,8,:] = x[:,12,:] # Neck
    if mode == '3d':
        y[:,9,:] = x[:,12,:] + np.array([0, 100, 50]) # Nose
        y[:,10,:] = x[:,12,:] + np.array([0, 200, 0]) # Head -> Neck에서 z축으로 약 20cm 위
    else:
        y[:,9,:] = x[:,12,:] #+ np.array([0, -50]) # Nose
        y[:,10,:] = x[:,12,:] #+ np.array([0, -100]) # Head -> Neck에서 z축으로 약 20cm 위
    y[:,11,:] = x[:,16,:] # L_Shoulder
    y[:,12,:] = x[:,18,:] # L_Elbow
    y[:,13,:] = x[:,20,:] # L_Wrist
    y[:,14,:] = x[:,17,:] # R_Shoulder
    y[:,15,:] = x[:,19,:] # R_Elbow
    y[:,16,:] = x[:,21,:] # R_Wrist
    
    return y

def coco2h36m(x, mode='3d'):
    '''
        Input: x (T x V x C)  
       //coco 17 body keypoints
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
    '''
    if len(x.shape) == 2:
        V, C = x.shape
        T = 1
        x = x.reshape(T, V, C)
    else:
        T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:]  = (x[:,11,:] + x[:,12,:]) * 0.5 # Pelvis = (L_Hip + R_Hip) / 2
    y[:,1,:]  = x[:,12,:] # R_Hip
    y[:,2,:]  = x[:,14,:] # R_Knee
    y[:,3,:]  = x[:,16,:] # R_Ankle
    y[:,4,:]  = x[:,11,:] # L_Hip
    y[:,5,:]  = x[:,13,:] # L_Knee
    y[:,6,:]  = x[:,15,:] # L_Ankle
    y[:,8,:]  = (x[:,5,:] + x[:,6,:]) * 0.5 # Neck = (L_shoulder + R_shoulder) / 2
    y[:,7,:]  = (y[:,0,:] + y[:,8,:]) * 0.5 # Torso = (Neck + Pelvis) / 2
    y[:,9,:]  = x[:,0,:]  # Nose
    y[:,10,:] = (x[:,1,:] + x[:,2,:]) * 0.5   # Head = (left_eye + right_eye) / 2
    y[:,11,:] = x[:,5,:]  # L_Shoulder
    y[:,12,:] = x[:,7,:]  # L_Elbow
    y[:,13,:] = x[:,9,:]  # L_Wrist
    y[:,14,:] = x[:,6,:]  # R_Shoulder
    y[:,15,:] = x[:,8,:]  # R_Elbow
    y[:,16,:] = x[:,10,:] # R_Wrist
    
    return y

def fit3d2h36m(x):

    pass

# MPJPE
def euclidean_distance(a,b):
    re = np.sqrt(((a - b)**2).sum(axis=-1)).mean(axis=-1)
    return re


# 카메라 파라미터 읽어오기
def loadAIHubCameraParameter(json_path, trans_scale=1.0, W=1920):
    cam_json = readJSON(json_path)
    return getAIHubCameraParameter(cam_json, trans_scale, W)
    

def getAIHubCameraParameter(cam_json, trans_scale=1.0, W=1920):
    extrinsic_properties = np.array(cam_json['extrinsics'])
    R = copy.deepcopy(np.array(cam_json['extrinsics'])[:,:3])
    T = copy.deepcopy(np.array(cam_json['extrinsics'])[:,3]*trans_scale)
    R_c = Rotation.T
    C = - np.matmul(R_c, T)
    intrinsic_properties = np.array(cam_json['intrinsics']) # normalized intrinsic matrix
    intrinsic_properties[:2, :] *= W # denormalize
    fx = intrinsic_properties[0,0]
    fy = intrinsic_properties[1,1]
    cx = intrinsic_properties[0,2]
    cy = intrinsic_properties[1,2]
    
    return cam_json, extrinsic_properties, R, T, R_c, C, intrinsic_properties, fx, fy, cx, cy

# Coordinate Transformation
def World2CameraCoordinate(pos, extrinsic_properties):
    # input: pos (N, 4) -> (x, y, z, 1), extrinsic_properties (3, 4)
    # output: pos (N, 3) -> (x, y, z)
    if len(pos.shape) == 1: 
        C = pos.shape[0]
        pos = pos.reshape(1, C)
    elif len(pos.shape) == 2:
        if pos.shape[1] == 3:
            pos = np.concatenate([pos, np.ones([pos.shape[0], 1])], axis=1)
    
    return np.matmul(extrinsic_properties, pos.T).T # World coordinate -> Camera coordinate

def Camera2ImageCoordinate(pos, intrinsic_properties):
    # input: pos (N, 3) -> (x, y, z), intrinsic_properties (3, 3)
    # output: pos (N, 3) -> (u, v, 1)
    for i in range(pos.shape[0]): # World coordinate -> Image coordinate
        pos[i] = np.matmul(intrinsic_properties, pos[i])
        pos[i] /= pos[i, 2]
    return pos

def World2ImageCoordinate(P_W, extrinsic_properties, intrinsic_properties):
    # input: pos (N, 4) -> (x, y, z, 1), extrinsic_properties (3, 4), intrinsic_properties (3, 3)
    # output: pos (N, 3) -> (u, v, 1)
    P_C = World2CameraCoordinate(P_W, extrinsic_properties)
    P_p = Camera2ImageCoordinate(P_C, intrinsic_properties)
    return P_p

# 카메라 별 회전 보정
cam_rot = {
    1: [145, 0, 20], # z, y, x
    2: [90, -15, 0],
    3: [45, -15, -5],
    4: [0, 3, -7],
    5: [-45, 10, 0],
    6: [-90, 13, 10],
    7: [-145, 5, 15],
    8: [170, 5, 15]
}

# Functions
def getNumFromImgName(img): # 이미지 파일명에서 프레임 번호 추출
    return int(img.split('.')[0].split('_')[-1])

# normalize
def normalize(data, max, min):
    return (data - min) / (max - min)

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2

def array2dict(array, keypoints, verbose=False):
    # input: array (N, 2), keypoints (N,)
    # output: points_dict (dict) = {'keypoint1': array([x1, y1]), 'keypoint2': array([x2, y2]), ...}
    points_dict = {}
    for i in range(array.shape[0]):
        points_dict[keypoints[i]] = array[i]
        if verbose:
            print(i, keypoints[i], array[i])
    return points_dict

# check max, min
# def check_max_min(data): # (N, 3) array
#     #print'data: ', data)
#     max_ = max_x = max_y = max_z =  -100000
#     min_ = min_x = min_y = min_z = 100000
    
#     for pos in data:
#         x = pos[0]
#         y = pos[1]
#         z = pos[2]
#         if x > max_:
#             max_ = x
#         if x < min_:
#             min_ = x
#         # if x > max_x:
#         #     max_x = x
#         # if x < min_x:
#         #     min_x = x
            
#         if y > max_:
#             max_ = y
#         if y < min_:
#             min_ = y
#         # if y > max_y:
#         #     max_y = y
#         # if y < min_y:
#         #     min_y = y
            
#         if z > max_:
#             max_ = z
#         if z < min_:
#             min_ = z
#         # if z > max_z:
#         #     max_z = z
#         # if z < min_z:
#         #     min_z = z
        
#     return max_, min_
def check_max_min(array): # (N, M) array
    # input: (N, M) array
    # output: max, min value of array
    return array.reshape(-1).max(), array.reshape(-1).min()
    
# Draw skeleton function
# - points -> joint 좌표  
# - connections -> joint 연결 관계      
def draw_skeleton(points, connections, elevation=0, azimuth=0, xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]), zaxis=dict(range=[-1, 1]), camera=None):
    fig = go.Figure()
    
    # Add origin
    fig.add_trace(go.Scatter3d(x=[0.0], y=[0.0], z=[0.0],
                                   mode='markers', marker=dict(size=5, color='red')))

    # Add points
    for point in points.values():
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                                   mode='markers', marker=dict(size=2, color='blue')))

    # Add connections
    for connection in connections:
        p1, p2 = points[connection[0]], points[connection[1]]
        fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                                   mode='lines', line=dict(color='black')))

    # Set layout
    fig.update_layout(scene=dict(xaxis=xaxis, yaxis=yaxis, zaxis=zaxis,
                                 aspectratio=dict(x=1, y=1, z=1),
                                 camera=dict(eye=dict(x=np.cos(np.pi * azimuth / 180) * np.cos(np.pi * elevation / 180)*0.5,
                                                      y=np.sin(np.pi * azimuth / 180) * np.cos(np.pi * elevation / 180)*0.5,
                                                      z=np.sin(np.pi * elevation / 180*0.5)))),
                      width=700, height=700, autosize=False,
                      scene_camera_eye=dict(x=1, y=1, z=1))
    
    if camera is not None:
        fig.update_layout(scene_camera=camera)

    return fig
    
# Draw skeleton function
def draw_skeleton_both(points1, connections1, points2, connections2, elevation=0, azimuth=0):
    fig = go.Figure()

    # Add points_
    for point in points1.values():
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                                   mode='markers', marker=dict(size=2, color='yellow')))

    # Add connections_
    for connection in connections1:
        p1, p2 = points1[connection[0]], points1[connection[1]]
        fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                                   mode='lines', line=dict(color='red')))

    # Add points
    for point in points2.values():
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                                   mode='markers', marker=dict(size=2, color='blue')))

    # Add connections
    for connection in connections2:
        p1, p2 = points2[connection[0]], points2[connection[1]]
        fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                                   mode='lines', line=dict(color='black')))

    # Set layout
    fig.update_layout(scene=dict(xaxis=dict(range=[-1,1]), yaxis=dict(range=[-1,1]), zaxis=dict(range=[-1,1]),
                                 aspectratio=dict(x=1, y=1, z=1),
                                 camera=dict(eye=dict(x=np.cos(np.pi * azimuth / 180) * np.cos(np.pi * elevation / 180),
                                                      y=np.sin(np.pi * azimuth / 180) * np.cos(np.pi * elevation / 180),
                                                      z=np.sin(np.pi * elevation / 180)))),
                      width=700, height=700, autosize=False,
                      scene_camera_eye=dict(x=1, y=1, z=1))

    fig.show()

def draw_skeleton_2d(points, connections, elevation=0, azimuth=0, img_source=None):
    fig = go.Figure()
    
    # Add points
    for point in points.values():
        fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]],
                                   mode='markers', marker=dict(size=2, color='blue')))

    # Add connections
    for connection in connections:
        p1, p2 = points[connection[0]], points[connection[1]]
        fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                                   mode='lines', line=dict(color='black')))

    img_width = 1920
    img_height = 1080
    scale_factor = 0.5

    # https://wikidocs.net/185955 참고
    fig.update_xaxes(range=[0, img_width ])
    fig.update_yaxes(range=[img_height, 0])
    #fig.update_xaxes(autorange="reversed") # 축 범위 반전, 위의 range 설정이 초기화되버림

    fig.update_layout(width=1000,height=620)
    
    if img_source is not None:
        fig.add_layout_image(
            source=img_source,
            xref="x",
            yref="y",
            x = 0,
            y = 0,
            sizex=img_width,
            sizey=img_height,
            sizing="stretch",
            opacity=0.5,
            layer="below"
        )
    
    fig.update_layout(template="plotly_white")

    fig.show()


def get3DResult(img_path, actor_id, output_folder): # /home/lhs/Datasets/HAAI/30_Squat, M160D
    cam_list = {}

    for folder in os.listdir(img_path): # 30_F160A_1, 30_F160A_7, ..., 30_F170D_4, ..., 30_M160A_1, ...
        if actor_id in folder: # actor_id에 해당하는 폴더만 - 30_M160D_1, 30_M160D_3, 30_M160D_5, 30_M160D_6, 30_M160D_7 (= 같은 영상에 대한 다른 카메라 시점)
            cam_num = int(folder.split('_')[-1]) # 카메라 번호 - 1, 3, 5, 6, 7
            
            # folder_content = os.listdir(os.path.join(img_path, folder)) # 폴더 내 이미지 목록 - 30_M160D_1_0.jpg, 30_M160D_1_1.jpg, ...
            
            # # 이미지 프레임 번호 추출
            # frame_list = []
            # for file in folder_content: # 30_M160D_1_0.jpg, 30_M160D_1_1.jpg, ... (반드시 0부터 시작하진 않음, 중간에 비는 번호도 존재)
            #     frame_num = int(file.split('.')[0].split('_')[-1]) # 이미지 프레임 번호 - 0, 1, ...
            #     frame_list.append(frame_num)
            
            # 이미지 프레임 번호 추출
            frame_list = []
            with open("/home/lhs/codes/AlphaPose/examples/res_{}/alphapose-results.json".format(folder)) as f:
                data = json.load(f) 
                
            for i in range(len(data)):
                image_id = data[i]['image_id'] # '23_M170A_2_0.jpg', ...
                frame_num = int(image_id.split('.')[0].split('_')[-1]) # 이미지 프레임 번호 - 0, 1, ...
                frame_list.append(frame_num)
            
            # 번호 순으로 정렬 (sorting)
            for i in range(len(frame_list)):
                ref = frame_list[i]
                for j in range(i+1, len(frame_list)):
                    c = frame_list[j]
                    if c < ref:
                        frame_list[i], frame_list[j] = frame_list[j], frame_list[i]
                        ref = c

            # dictionary에 저장
            cam_list[folder] = {"frame_list": frame_list, "cam_num": cam_num, "joint_3d_raw": {}, "joint_3d_converted": {}}
            
            # 카메라 파라미터 읽어오기
            try:
                root = "/home/lhs/Datasets/HAAI/Camera_json/train"
                json_path = os.path.join(root, folder+".json")
                cam_js = readJSON(json_path)
                extrinsics = np.array(cam_js['extrinsics'])
                intrinsics = np.array(cam_js['intrinsics'])
                cam_pos = extrinsics[:, 3]
                cam_rot = intrinsics[:, :3]
            except Exception as e:
                print("No Camera Parameter Info: ", folder)
                cam_pos = None
                cam_rot = None
            
            # 추론 결과 저장
            root_motionbert = "/home/lhs/codes/MotionBERT"
            cam = folder
            name = "MotionBERT_{}".format(cam) # MotionBERT_30_M160D_1, ...
            path = os.path.join(root_motionbert, output_folder, name) # /home/lhs/codes/MotionBERT/output/MotionBERT_30_M160D_1, ...
            npy_path = os.path.join(path, name) + '.npy' # /home/lhs/codes/MotionBERT/output/MotionBERT_30_M160D_1/MotionBERT_30_M160D_1.npy, ...
            
            if os.path.exists(npy_path): # .npy 파일이 존재하면
                result_3d = np.load(npy_path) # .npy 파일 로드
                print(result_3d.shape, type(result_3d)) # (388, 17, 3) = (프레임 수, keypoint 수, 3차원 좌표)
                print(len(frame_list))
                if result_3d.shape[0] != len(frame_list):
                    print("frame number mismatch")
                    continue
                for i, frame_num in enumerate(frame_list):
                    #print(cam_num, i, frame_num)
                    try:
                        cam_list[cam]["joint_3d_raw"][frame_num] = {"points": result_3d[i]} # (17, 3)
                        # cam_list[cam]["joint_3d_converted"][frame_num] = convert3DResult(result_3d[i], cam_num) # {"points", "max", "min", "center_point"}
                        cam_list[cam]["joint_3d_converted"][frame_num] = convert3DResult(result_3d[i], cam_num, cam_rot) # {"points", "max", "min", "center_point"}
                    except Exception as e:
                        print(e)
                        cam_list[cam]["joint_3d_raw"][frame_num] = None
                        cam_list[cam]["joint_3d_converted"][frame_num] = None
                print("cam ", cam_num, "joint 3d loaded")
            else:
                print("No .npy file")
    
    return cam_list

def convert3DResult(joint_3d_raw, cam_num, cam_rot=None):
    joint_3d_normalized = {}
    joint_3d_converted = {}
    
    # 데이터 정규화를 위한 max, min 값 계산
    max_, min_ = check_max_min(joint_3d_raw)
    
    # 정답 데이터 정규화 
    center_point = 0
    #print(joint_3d_raw.shape)
    for joint in range(17):
        pos = joint_3d_raw[joint]
        x = normalize(pos[0], max_, min_)
        y = normalize(pos[1], max_, min_)
        z = normalize(pos[2], max_, min_)
        joint_3d_normalized[joint] = [x, z, -y]
        if joint == 0:
            center_point = [x, z, -y]
            
    # Move center_point to origin
    for key in joint_3d_normalized.keys():
        joint_3d_normalized[key][0] = joint_3d_normalized[key][0] - center_point[0]
        joint_3d_normalized[key][1] = joint_3d_normalized[key][1] - center_point[1]
        joint_3d_normalized[key][2] = joint_3d_normalized[key][2] - center_point[2]       
    
    # 회전 보정
    if cam_rot is None:
        r_z = Rotation.from_euler('z', cam_rot[cam_num][0], degrees=True)
        r_y = Rotation.from_euler('y', cam_rot[cam_num][1], degrees=True)
        r_x = Rotation.from_euler('x', cam_rot[cam_num][2], degrees=True)

        for key in joint_3d_normalized.keys():
            joint_3d_converted[key] = r_x.apply(r_y.apply(r_z.apply(joint_3d_normalized[key])))
    else:
        for key in joint_3d_normalized.keys():
            joint_3d_converted[key] = np.matmul(cam_rot, joint_3d_normalized[key])
        
    return {"points": joint_3d_converted, "max": max_, "min": min_, "center_point": center_point}

# 정답 저장 
def getGT(gt_path_3d): # /home/lhs/Datasets/HAAI/30_Squat_label3D/30_M160D
    gt = {"frame_list": [], "joint_3d_raw": {}, "joint_3d_converted": {}}
    # 3D 정답 데이터가 저장된 폴더가 존재하지 않으면 종료
    if not os.path.exists(gt_path_3d):
        print("No such directory: ", gt_path_3d)
        return None
    
    folder_content = os.listdir(gt_path_3d)
    
    # json 프레임 번호 추출
    frame_list = []
    for json_file in folder_content: # 3D_30_M160D_0.json, 3D_30_M160D_1.json, ... (반드시 0부터 시작하진 않음, 중간에 비는 번호도 존재)
        frame_num = int(json_file.split('.')[0].split('_')[-1]) # 이미지 프레임 번호 - 0, 1, ...
        frame_list.append(frame_num)
        
        gt["joint_3d_raw"][frame_num] = {}
        gt["joint_3d_converted"][frame_num] = {}
        
        # json 파일 로드
        with open(os.path.join(gt_path_3d, json_file)) as f:
            data = json.load(f) # dict_keys(['info', 'annotations'])
            
        joint_3d_raw = np.array(data['annotations']['3d_pos'])[:,:3, 0] # (24, 4, 1) list -> (24, 3) np.array
        gt["joint_3d_raw"][frame_num]["points"] = joint_3d_raw
            
        # 데이터 정규화를 위한 max, min 값 계산
        max_, min_ = check_max_min(joint_3d_raw)
        gt["joint_3d_converted"][frame_num]["max"] = max_
        gt["joint_3d_converted"][frame_num]["min"] = min_
        
        # 정답 데이터 정규화
        points = {}
        for i, part in enumerate(data['info']['3d_pos']):
            pos = data['annotations']['3d_pos'][i]
            x = normalize(pos[0][0], max_, min_)
            y = normalize(pos[2][0], max_, min_)
            z = normalize(pos[1][0], max_, min_)
            points[part] = [x, y, z]
            
        # 중심점 계산
        center_point = [(points['R_Hip'][0] + points['L_Hip'][0])/2.0, 
                        (points['R_Hip'][1] + points['L_Hip'][1])/2.0,
                        (points['R_Hip'][2] + points['L_Hip'][2])/2.0]
        gt["joint_3d_converted"][frame_num]["center_point"] = center_point
        
        # 중심점을 기준으로 좌표 이동
        for key in points.keys():
            points[key][0] -= center_point[0]
            points[key][1] -= center_point[1]
            points[key][2] -= center_point[2]
            
        gt["joint_3d_converted"][frame_num]["points"] = points
        
    # 번호 순으로 정렬 (sorting)
    for i in range(len(frame_list)):
        ref = frame_list[i]
        for j in range(i+1, len(frame_list)):
            c = frame_list[j]
            if c < ref:
                frame_list[i], frame_list[j] = frame_list[j], frame_list[i]
                ref = c
    gt["frame_list"] = frame_list

    return gt


def MPJPE(points1, points2, scale=1517.7871, offset=63.2871): # scale -> max-min, offset -> min
    sum = 0
    for key in h36m_keypoints.keys():
        #print(key)
        if h36m_keypoints[key] in ['Pelvis', 'Torso', 'Nose', 'Head']: # blacklist
            continue
        h36m_point = np.array(points1[key])*scale+offset
        aihub_point = np.array(points2[h36m_keypoints[key]])*scale+offset
        #print(key, h36m_keypoints[key], h36m_point, aihub_point)
        sum += distance.euclidean(h36m_point, aihub_point)
    sum /= 13
    return sum

# query average error for one camera
def avgErrorForOneCamera(dataset, action, actor_id, cam):
    frame_list = dataset[action]['result3d'][actor_id][cam]['frame_list']
    total_sum = 0
    try:
        for frame_num in frame_list:
            gt_points = dataset[action]['gt'][actor_id]['joint_3d_converted'][frame_num]['points']
            infer_points = dataset[action]['result3d'][actor_id][cam]['joint_3d_converted'][frame_num]['points']
            total_sum += MPJPE(infer_points, gt_points)
        total_sum /= len(frame_list)
        #print("total_sum {} {}: ".format(actor_id, cam), total_sum)
        return total_sum
    except Exception as e:
        print(e, action, actor_id, cam)
        return 0

def avgErrorForOneActor(dataset, action, actor_id):
    cam_list = [cam for cam in dataset[action]['result3d'][actor_id].keys()]
    total_sum = 0
    num_none = 0
    for cam in cam_list:
        error = avgErrorForOneCamera(dataset, action, actor_id, cam)
        total_sum += error
        if error == 0: # 에러에 의해 결과가 없는 경우
            num_none += 1
    if len(cam_list)-num_none != 0:
        total_sum /= (len(cam_list)-num_none)
    #print("total_sum {}: ".format(actor_id), total_sum)
    return total_sum

def avgErrorForOneAction(dataset, action):
    actor_list = dataset[action]["actor_list"]
    #print("actor_list: ", actor_list)
    # actor가 없으면 None 반환
    if len(actor_list) == 0:
        return None
    
    total_sum = 0
    for actor_id in actor_list:
        #print(actor_id)
        total_sum += avgErrorForOneActor(dataset, action, actor_id)
    total_sum /= len(actor_list)
    return total_sum


### for calculate scaling factor

def MPJPE_for_single_pose(pred, gt, root_rel=True):
    """
    pred: (17, 3)
    gt: (17, 3)
    """
    assert pred.shape == gt.shape
    assert len(pred.shape) == 2
    if root_rel:
        return np.sqrt(np.sum((pred - gt) ** 2, axis=1)).mean()
    else:
        return np.sqrt(np.sum((get_rootrel_pose(pred) - get_rootrel_pose(gt)) ** 2, axis=1)).mean()

def MPJPE_for_multiple_pose(pred, gt, root_rel=True):
    """
    pred: (N, 17, 3)
    gt: (N, 17, 3)
    """
    assert pred.shape == gt.shape
    assert len(pred.shape) == 3
    mpjpe = 0
    for i in range(pred.shape[0]):
        mpjpe += MPJPE_for_multiple_pose(pred[i], gt[i], root_rel)
    return mpjpe / pred.shape[0]

def get_rootrel_pose(pose):
    # input: pose (N, D) or 
    # output: rootrel_pose (1N, D)
    if len(pose.shape) == 2:
        rootrel_pose = pose - pose[0]
    elif len(pose.shape) == 3:
        rootrel_pose = pose - pose[:, 0:1]
    elif len(pose.shape) == 4:
        rootrel_pose = pose - pose[:, :, 0:1]
    return rootrel_pose

def get_xy_centered_pose(pose):
    # input: pose (N, D)
    # output: xy_centered_pose (N, D)
    xy_centered_pose = pose.copy()
    xy_centered_pose[:, 0] -= pose[:, 0].mean()
    xy_centered_pose[:, 1] -= pose[:, 1].mean()
    return xy_centered_pose

# https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/gendb.py
def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[..., :2] / pose3d[..., 2:3]
    pose2d[..., 0] *= fx
    pose2d[..., 1] *= fy
    pose2d[..., 0] += cx
    pose2d[..., 1] += cy
    return pose2d

# https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/gendb.py
def camera_to_image_frame(pose3d, box, camera, rootIdx):
    # x, y
    img_3d = np.zeros_like(pose3d)
    img_3d[:, :2] = _weak_project(
        pose3d.copy(), camera['fx'], camera['fy'], camera['cx'], camera['cy'])
    # z
    rectangle_3d_size = 2000.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size
    pose3d_depth = ratio * (pose3d[:, 2] - pose3d[rootIdx, 2])
    img_3d[:, 2] = pose3d_depth
    img_2d = img_3d[:, :2]
    return img_2d, img_3d

# https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/gendb.py
def infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[:2] -= 1000.0
    br_joint = root_joint.copy()
    br_joint[:2] += 1000.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))
    #print(root_joint, tl_joint, br_joint)

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])

def optimize_scaling_factor(cam_cs_hat, img_cs_hat, epochs=200, learningRate=0.000005, stop_tolerance=0.0001, gpus='0, 1'):
    # cam_cs_hat, img_cs_hat: (17, 3)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    # https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
    import torch
    from torch.autograd import Variable
    class linearRegression(torch.nn.Module):
        def __init__(self, inputSize, outputSize):
            super(linearRegression, self).__init__()
            self.linear = torch.nn.Linear(inputSize, outputSize, bias=False)

        def forward(self, x):
            out = self.linear(x)
            return out

    x_train = copy.deepcopy(cam_cs_hat.reshape(-1, 1).astype(np.float32)) # 모든 점을 batch로 취급
    y_train = copy.deepcopy(img_cs_hat.reshape(-1, 1).astype(np.float32)) # 모든 점을 batch로 취급

    inputDim = 1        # takes variable 'x' 
    outputDim = 1       # takes variable 'y'

    model = linearRegression(inputDim, outputDim)
    #model.linear.weight.data[0] = 0.25

    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    losses = []
    weights = []
    tol_cnt = 0
    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x_train))
            labels = Variable(torch.from_numpy(y_train))
        else:
            inputs = Variable(torch.from_numpy(x_train))
            labels = Variable(torch.from_numpy(y_train))

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        #loss.requires_grad = True
        losses.append(loss.item())
        
        # if loss is not decreasing, stop training
        if epoch > 1:
            if abs(losses[-1]-loss.item()) < stop_tolerance:
                tol_cnt += 1
                if tol_cnt > 5: break
            else:
                tol_cnt = 0    
            
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        
        #weights.append(model.linear.weight.data.item())

        #print('epoch {}, loss {}'.format(epoch, loss.item()))
    return model.linear.weight.data.item(), losses



def plot_cv2_image(img):
    #plt.axis('off')
    #fig = plt.figure()
    plt.imshow(img)
    plt.show()


def skew_symmetric_matrix(v):
    """
    Calculate the skew-symmetric matrix of a 3D vector.
    Args:
        v (torch.Tensor): 3D vector. Shape (3,).
    Returns:
        torch.Tensor: Skew-symmetric matrix. Shape (3, 3).
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float32)

def skew_symmetric_matrix_tensor(v):
    """
    Calculate the skew-symmetric matrix of a 3D vector.
    Args:
        v (torch.Tensor): 3D vector. Shape (3,).
    Returns:
        torch.Tensor: Skew-symmetric matrix. Shape (3, 3).
    """
    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=torch.float32)


def normalize_array(arr, max_value=None, min_value=None):
    arr = np.array(arr)
    if max_value is None:
        max_value = arr.max()
    if min_value is None:
        min_value = arr.min()
    return (arr - min_value) / (max_value - min_value)

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, num_frame, fps

def get_video_frame(video_path, frame_id=None):
    if not os.path.exists(video_path):
        print('Video does not exist')
        return None
    cap = cv2.VideoCapture(video_path)
    if frame_id is None:
        frames = []
        while True:
            ret, img_frame = cap.read()
            if ret == False: break
            frames.append(img_frame)
        return frames
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, img_frame = cap.read()
        cap.release()
        return cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
    
def get_bbox_area(bbox, input_type='xyxy'):
    if input_type == 'xxyy':
        x1, x2, y1, y2 = bbox
    elif input_type == 'xyxy':
        x1, y1, x2, y2 = bbox
    elif input_type == 'xywh':
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
    return (x2 - x1) * (y2 - y1)

def get_bbox_from_pose2d(pose_2d, output_type='xyxy'):
    assert len(pose_2d.shape) == 2, 'pose_2d should be (num_joints, 2)'
    x1 = float(pose_2d[:, 0].min())
    x2 = float(pose_2d[:, 0].max())
    y1 = float(pose_2d[:, 1].min())
    y2 = float(pose_2d[:, 1].max())
    
    if output_type == 'xyxy':
        return x1, y1, x2, y2
    elif output_type == 'xywh':
        cx = x1 + (x2 - x1) / 2
        cy = y1 + (y2 - y1) / 2
        return cx, cy, x2 - x1, y2 - y1
    elif output_type == 'xxyy':
        return x1, x2, y1, y2
    
def get_batch_bbox_from_pose2d(batch_pose_2d):
    assert type(batch_pose_2d) == torch.Tensor, 'batch_pose_2d should be torch.Tensor'
    batch_x1 = batch_pose_2d[:, :, 0].min(dim=1, keepdim=True).values
    batch_x2 = batch_pose_2d[:, :, 0].max(dim=1, keepdim=True).values
    batch_y1 = batch_pose_2d[:, :, 1].min(dim=1, keepdim=True).values
    batch_y2 = batch_pose_2d[:, :, 1].max(dim=1, keepdim=True).values
    batch_bbox = torch.cat([batch_x1, batch_y1, batch_x2, batch_y2], dim=1)
    return batch_bbox

def get_bbox_area_from_pose2d(pose_2d):
    bbox = get_bbox_from_pose2d(pose_2d, output_type='xyxy')
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def change_bbox_convention(bbox, input_type='xyxy', output_type='xywh'):
    if input_type == 'xxyy':
        x1, x2, y2, y2 = bbox
    elif input_type == 'xyxy':
        x1, y1, x2, y2 = bbox
    elif input_type == 'xywh':
        cx, cy, w, h = bbox
        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    if output_type == 'xxyy':
        return int(x1), int(x2), int(y1), int(y2)
    elif output_type == 'xyxy':
        return int(x1), int(y1), int(x2), int(y2)
    elif output_type == 'xywh':
        return int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2), int(x2 - x1), int(y2 - y1)
    
    
## ------------------------------------------  from pytorch3d
# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def get_pose_height(pose_2d):
    if len(pose_2d.shape) == 2:
        return pose_2d[:, 1].max(axis=-1) - pose_2d[:, 1].min(axis=-1)
    elif len(pose_2d.shape) == 3:
        return pose_2d[:, :, 1].max(axis=-1) - pose_2d[:, :, 1].min(axis=-1)
    elif len(pose_2d.shape) == 4:
        return pose_2d[:, :, :, 1].max(axis=-1) - pose_2d[:, :, :, 1].min(axis=-1)
    
def procrustes_align(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    predicted: (T, 17, 3)
    target: (T, 17, 3)
    """
    assert predicted.shape == target.shape
    
    # Translation
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    # Uniform scaling
    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    # Rotation
    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return predicted_aligned

def T_to_C(R, T):
    R = np.array(R)
    T = np.array(T).reshape(-1)
    assert R.shape == (3, 3)
    assert T.shape == (3,)
    return - R.T @ T
    
def C_to_T(R, C):
    R = np.array(R)
    C = np.array(C).reshape(-1)
    assert R.shape == (3, 3)
    assert C.shape == (3,)
    return - R @ C

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Returns the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transformation matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    # Normalize the input vectors
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    # Compute the skew-symmetric cross-product matrix of v
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    
    # Compute the rotation matrix
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    
    return rotation_matrix

def get_canonical_3d(world_3d, cam_3d, C, R, fixed_dist=3.5, return_vector_cam_forward=False, canonical_type='same_z'):
    num_frames = len(world_3d)
    canonical_3d = world_3d.copy()
    cam_origin_w = C.copy()
    pelvis_w = world_3d[:, 0].copy()
    if canonical_type == 'same_z':
        pelvis_z_in_cam_frame = cam_3d[:, 0, 2].copy() # (F,)
        mag_cam_origin_to_pelvis = np.expand_dims(pelvis_z_in_cam_frame, axis=1).repeat(3, axis=1) # (F, 3)
    elif canonical_type == 'same_dist':
        #pelvis = world_3d[:, 0].copy() # (F, 3)
        vec_cam_origin_to_pelvis = pelvis_w - cam_origin_w # (F, 3)
        mag_cam_origin_to_pelvis = np.expand_dims(np.linalg.norm(vec_cam_origin_to_pelvis, axis=1), axis=1).repeat(3, axis=1) # (F, 3)
    elif canonical_type == 'fixed_dist':
        mag_cam_origin_to_pelvis = fixed_dist
    
    vec_cam_forward = np.multiply(np.expand_dims(R[2], 0).repeat(num_frames, axis=0),  mag_cam_origin_to_pelvis)
    canonical_pelvis = cam_origin_w + vec_cam_forward
    canonical_3d = canonical_3d - np.expand_dims(pelvis_w, 1) + np.expand_dims(canonical_pelvis, 1)
    
    if return_vector_cam_forward:
        return canonical_3d, vec_cam_forward
    else:
        return canonical_3d