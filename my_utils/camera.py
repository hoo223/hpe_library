from lib_import import *

class Camera:
    def __init__(self, origin, calib_mat, cam_ext=None, R=None, cam_default_R=None, 
                 roll=0, pitch=0, yaw=0,
                 IMAGE_HEIGHT=1000, IMAGE_WIDTH=1000, cam_name='camera'):
        mm_to_m = 0.001
        m_to_mm = 1000
        self.cam_name = cam_name
        if cam_default_R is None:
            forward = [0, 1, 0]
            left = [1, 0, 0]
            up = np.cross(forward, left)
            self.cam_default_R = np.array([left, up, forward]) # default camera orientation
        else:
            self.cam_default_R = cam_default_R
            
        # intrinsic parameter
        self.intrinsic = calib_mat
        # extrinsic parameter
        if cam_ext is None:
            self.C = origin
            if R is None:
                self.rot_z = Rotation.from_euler('z', roll, degrees=True).as_matrix()
                self.rot_y = Rotation.from_euler('y', yaw, degrees=True).as_matrix()
                self.rot_x = Rotation.from_euler('x', pitch, degrees=True).as_matrix()
                #self.R = (self.rot_x @ self.rot_y @ self.rot_z).T @ self.cam_default_R
                self.R = (self.rot_z @ self.rot_y @ self.rot_x).T @ self.cam_default_R 
            else:
                self.R = R @ self.cam_default_R 
            self.t = (- self.R @ self.C).reshape(-1,1)*m_to_mm # [mm]
            self.extrinsic = {'R': self.R, 't': self.t}
        else:
            self.extrinsic = cam_ext
            self.R = np.array(cam_ext['R']) #@ self.cam_default_R 
            self.t = np.array(cam_ext['t']) # [mm]
        
        # camera frame
        self.cam_frame = self.generate_camera_frame(self.extrinsic, name=self.cam_name)
        self.origin = self.cam_frame.origin
        self.C = self.origin

        # projection matrix
        self.ext_mat = np.hstack([self.R, self.t*mm_to_m]) # 3 x 4
        self.extrinsic = np.hstack([self.R, self.t*mm_to_m])
        self.cam_proj = self.intrinsic @ self.ext_mat
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.IMAGE_WIDTH = IMAGE_WIDTH

        fx = self.intrinsic[0][0]
        fy = self.intrinsic[1][1]
        cx = self.intrinsic[0][2]
        cy = self.intrinsic[1][2]
        
        # for drawing
        F = fx*mm_to_m  # focal length
        PX= cx*mm_to_m # principal point x-coordinate
        PY= -cy*mm_to_m # principal point y-coordinate

        # dx, dy, dz = self.cam_default_R
        # THETA_X = 0 #np.pi #/ 2  # roll angle
        # THETA_Y = -np.pi / 2  # pitch angle
        # THETA_Z = 0 #np.pi  # yaw angle
        # #R_image_frame = rotation_matrix_to_vector_align(np.array([-1, 0, 0]), cam_frame.dz)
        # #R = get_rotation_matrix(theta_x=THETA_X, theta_y=THETA_Y, theta_z=THETA_Z)

        self.Z = PrincipalAxis(
            camera_center=self.cam_frame.origin,
            camera_dz=self.cam_frame.dz,
            f=F,
        )

        self.image_frame = ReferenceFrame(
            origin=self.Z.p - self.cam_frame.dx * PX + self.cam_frame.dy * PY, 
            dx=self.R[0], 
            dy=self.R[1],
            dz=self.R[2],
            name="Image",
        )

        # with lines
        # self.image_plane = ImagePlane(
        #     origin=self.image_frame.origin, 
        #     dx=self.image_frame.dx, 
        #     dy=self.image_frame.dy, 
        #     height=self.IMAGE_HEIGHT/100,
        #     width=self.IMAGE_WIDTH/100,
        #     mx=10.0,
        #     my=10.0
        # )

        # without lines
        self.image_plane = ImagePlane(
            origin=self.image_frame.origin, 
            dx=self.image_frame.dx, 
            dy=self.image_frame.dy, 
            height=1,
            width=1,
            mx=1000/self.IMAGE_WIDTH,
            my=1000/self.IMAGE_HEIGHT
        )
        
    def update_camera_parameter(self, calib_mat=None, origin=None, roll=None, pitch=None, yaw=None, H=None, W=None):
        mm_to_m = 0.001
        m_to_mm = 1000
        if H is not None:
            self.IMAGE_HEIGHT = H
        if W is not None:
            self.IMAGE_WIDTH = W
        # intrinsic parameter
        if calib_mat is not None:
            self.intrinsic = calib_mat
        # extrinsic parameter
        if origin is not None:
            self.C = origin
        if roll is not None:
            self.rot_z = Rotation.from_euler('z', roll, degrees=True).as_matrix()
        if yaw is not None:
            self.rot_y = Rotation.from_euler('y', yaw, degrees=True).as_matrix()
        if pitch is not None:
            self.rot_x = Rotation.from_euler('x', pitch, degrees=True).as_matrix()
        #self.R = (self.rot_x @ self.rot_y @ self.rot_z).T @ self.cam_default_R 
        self.R = (self.rot_z @ self.rot_y @ self.rot_x).T @ self.cam_default_R
        self.t = (- self.R @ self.C).reshape(-1,1) * m_to_mm # [mm]
        self.extrinsic = {'R': self.R, 't': self.t}
        self.ext_mat = np.hstack([self.R, self.t*mm_to_m]) # 3 x 4
        self.R = np.array(self.extrinsic['R']) #
        self.t = np.array(self.extrinsic['t']) # [mm]
        
        # camera frame
        self.cam_frame = self.generate_camera_frame(self.extrinsic, name=self.cam_name)
        self.origin = self.cam_frame.origin
        self.C = self.origin

        # projection matrix
        self.cam_proj = self.intrinsic @ self.ext_mat
        
        # for drawing
        fx = self.intrinsic[0][0]
        fy = self.intrinsic[1][1]
        cx = self.intrinsic[0][2]
        cy = self.intrinsic[1][2]
        
        # for drawing
        F = fx*mm_to_m # focal length
        PX= cx*mm_to_m # principal point x-coordinate
        PY= -cy*mm_to_m # principal point y-coordinate
        
        self.Z = PrincipalAxis(
            camera_center=self.cam_frame.origin,
            camera_dz=self.cam_frame.dz,
            f=F,
        )

        self.image_frame = ReferenceFrame(
            origin=self.Z.p - self.cam_frame.dx * PX + self.cam_frame.dy * PY, 
            dx=self.R[0], 
            dy=self.R[1],
            dz=self.R[2],
            name="Image",
        )

        # without lines
        self.image_plane = ImagePlane(
            origin=self.image_frame.origin, 
            dx=self.image_frame.dx, 
            dy=self.image_frame.dy, 
            height=1,
            width=1,
            mx=1000/self.IMAGE_WIDTH,
            my=1000/self.IMAGE_HEIGHT
        )

    def generate_camera_frame(self, cam_ext, mm_to_m=True, name='camera'):
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
            name=f"{name}",
        )
        return cam_frame
    
    def update_torso_projection(self, torsos):
        self.Gs = []
        self.pies = []
        self.xs = []
        self.proj_torsos = []

        for torso in torsos:
            for i in range(5):
                X = torso[i]
                G = GenericPoint(X, name="X")
                X1 = self.image_frame.origin
                X2 = X1 + self.image_frame.dx
                X3 = X1 + self.image_frame.dy
                pi = get_plane_from_three_points(X1, X2, X3)
                self.Gs.append(G)
                self.pies.append(pi)
                self.xs.append(G.get_x(pi, C=self.C))

        
        for i in range(len(torsos)):
            self.proj_torsos.append(np.array(self.xs[i*5:(i+1)*5]))
            
    def update_pose_projection(self, poses):
        self.Gs = []
        self.pies = []
        self.xs = []
        self.proj_torsos = []

        for pose in poses:
            for i in range(len(pose)):
                X = pose[i]
                G = GenericPoint(X, name="X")
                X1 = self.image_frame.origin
                X2 = X1 + self.image_frame.dx
                X3 = X1 + self.image_frame.dy
                pi = get_plane_from_three_points(X1, X2, X3)
                self.Gs.append(G)
                self.pies.append(pi)
                self.xs.append(G.get_x(pi, C=self.C))

    def update_point_projection(self, points):
        self.Gs = []
        self.pies = []
        self.xs = []
        self.proj_points = []

        for point in points:
            X = point
            G = GenericPoint(X, name="X")
            X1 = self.image_frame.origin
            X2 = X1 + self.image_frame.dx
            X3 = X1 + self.image_frame.dy
            pi = get_plane_from_three_points(X1, X2, X3)
            self.Gs.append(G)
            self.pies.append(pi)
            self.xs.append(G.get_x(pi, C=self.C))

        
        for i in range(len(points)):
            self.proj_points.append(np.array(self.xs[i*5:(i+1)*5]))     
            
    def update_line_projection(self, lines):
        self.Gs = []
        self.pies = []
        self.xs = []
        self.proj_lines = []

        for line in lines:
            for i in range(2):
                X = line[i]
                G = GenericPoint(X, name="X")
                X1 = self.image_frame.origin
                X2 = X1 + self.image_frame.dx
                X3 = X1 + self.image_frame.dy
                pi = get_plane_from_three_points(X1, X2, X3)
                self.Gs.append(G)
                self.pies.append(pi)
                self.xs.append(G.get_x(pi, C=self.C))
        
        for i in range(len(lines)):
            self.proj_lines.append(np.array(self.xs[i*5:(i+1)*5]))        
    
    
class BatchCamera:
    def __init__(self, batch_origin, calib_mat=None, batch_roll=None, batch_pitch=None, batch_yaw=None, cam_default_R=None, 
                 W=1000, H=1000, data_type=torch.float32):
        self.data_type = data_type
        self.device = batch_origin.device
        self.batch_size = batch_origin.shape[0]
        self.num_frames = batch_origin.shape[1]
        
        self.batch_origin = batch_origin
        if batch_roll == None:  self.batch_roll = torch.zeros(self.batch_size, self.num_frames, 1)
        else:                   self.batch_roll = batch_roll
        if batch_pitch == None: self.batch_pitch = torch.zeros(self.batch_size, self.num_frames, 1)
        else:                   self.batch_pitch = batch_pitch
        if batch_yaw == None:   self.batch_yaw = torch.zeros(self.batch_size, self.num_frames, 1)
        else:                   self.batch_yaw = batch_yaw
        
        # check device
        if self.batch_origin.device != self.device: self.batch_origin = self.batch_origin.to(self.device)
        if self.batch_roll.device != self.device:   self.batch_roll   = self.batch_roll.to(self.device)
        if self.batch_pitch.device != self.device:  self.batch_pitch  = self.batch_pitch.to(self.device)
        if self.batch_yaw.device != self.device:    self.batch_yaw    = self.batch_yaw.to(self.device)
        
        if calib_mat is None:
            calib_mat = np.array([[1.14504940e+03, 0.00000000e+00, 5.12541505e+02],
                                  [0.00000000e+00, 1.14378110e+03, 5.15451487e+02],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        
        if cam_default_R is None:
            forward = [-1, 0, 0]
            left = [0, -1, 0]
            up = np.cross(left,forward)
            #self.cam_default_R = np.array([left, up, forward]) 
            self.batch_cam_default_R = torch.from_numpy(np.array([left, up, forward])).type(data_type).to(self.device).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.num_frames, 1, 1)
        else:
            #self.cam_default_R = cam_default_R
            self.batch_cam_default_R = torch.from_numpy(cam_default_R).type(data_type).to(self.device).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.num_frames, 1, 1)
        # intrinsic parameter
        self.batch_intrinsic = torch.from_numpy(calib_mat).type(data_type).to(self.device).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.num_frames, 1, 1)
        # extrinsic parameter
        self.batch_origin = batch_origin
        self.batch_C = batch_origin
        self.batch_rot_z = self.batch_rot_z_matrix(batch_roll)
        self.batch_rot_y = self.batch_rot_y_matrix(batch_yaw)
        self.batch_rot_x = self.batch_rot_x_matrix(batch_pitch)
        self.batch_R = (self.batch_rot_z @ self.batch_rot_y @ self.batch_rot_x).transpose(2, 3) @ self.batch_cam_default_R 
        self.batch_t = (- self.batch_R @ self.batch_C.unsqueeze(-1))
        self.batch_extrinsic = {'R': self.batch_R, 't': self.batch_t}

        # projection matrix
        self.batch_cam_proj = self.batch_intrinsic @ torch.cat([self.batch_R, self.batch_t], dim=-1)
        self.W = W
        self.H = H
        
    def batch_rot_z_matrix(self, batch_roll):
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
        return torch.stack([row1, row2, row3], dim=-1).transpose(2, 3).to(self.device) # [B, F, 3, 3]

    def batch_rot_y_matrix(self, batch_yaw):
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
        return torch.stack([row1, row2, row3], dim=-1).transpose(2, 3).to(self.device) # [B, F, 3, 3]

    def batch_rot_x_matrix(self, batch_pitch):
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
        return torch.stack([row1, row2, row3], dim=-1).transpose(2, 3).to(self.device) # [B, F, 3, 3]
        
    def update_camera_parameter(self, calib_mat=None, batch_origin=None, batch_roll=None, batch_pitch=None, batch_yaw=None, H=None, W=None):
        if H is not None:            self.H = H
        if W is not None:            self.W = W
        if calib_mat is not None:    self.batch_intrinsic = torch.from_numpy(calib_mat).type(self.data_type).to(self.device).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.num_frames, 1, 1)
        if batch_origin is not None: self.batch_C = batch_origin
        if batch_roll is not None:   self.batch_rot_z = self.batch_rot_z_matrix(batch_roll)
        if batch_yaw is not None:    self.batch_rot_y = self.batch_rot_y_matrix(batch_yaw)
        if batch_pitch is not None:  self.batch_rot_x = self.batch_rot_x_matrix(batch_pitch)
        self.batch_R = (self.batch_rot_z @ self.batch_rot_y @ self.batch_rot_x).transpose(2, 3) @ self.batch_cam_default_R 
        self.batch_t = (- self.batch_R @ self.batch_C.unsqueeze(-1))
        self.batch_extrinsic = {'R': self.batch_R, 't': self.batch_t}
        
        # camera frame
        self.batch_origin = batch_origin

        # projection matrix
        self.batch_cam_proj = self.batch_intrinsic @ torch.cat([self.batch_R, self.batch_t], dim=-1)