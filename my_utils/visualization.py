from lib_import import *
from .dh import projection, generate_world_frame, rotate_torso_by_R
from .test_utils import get_h36m_keypoint_index

def clear_axes(ax, blacklist=[]):
    if type(ax) == list:
        for ax_ in ax:
            if 'line' not in blacklist:
                for line in ax_.lines[:]:
                    line.remove()  # Remove lines
            if 'collection' not in blacklist:   
                for collection in ax_.collections[:]:
                    collection.remove()  # Remove collections, e.g., scatter plots
            if 'patch' not in blacklist:
                for patch in ax_.patches[:]:
                    patch.remove()
            if 'text' not in blacklist:
                for text in ax_.texts[:]:
                    text.remove()
            if 'image' not in blacklist:
                for image in ax_.images[:]:
                    image.remove()
    else:
        if 'line' not in blacklist:
            for line in ax.lines[:]:
                line.remove()  # Remove lines
        if 'collection' not in blacklist:   
            for collection in ax.collections[:]:
                collection.remove()  # Remove collections, e.g., scatter plots
        if 'patch' not in blacklist:
            for patch in ax.patches[:]:
                patch.remove()
        if 'text' not in blacklist:
            for text in ax.texts[:]:
                text.remove()
        if 'image' not in blacklist:
            for image in ax.images[:]:
                image.remove()

def axes_2d(fig=None, rect=None, loc=111, W=1000, H=1000, xlim=None, ylim=None, xlabel='X', ylabel='Y', title='', axis='on', show_axis=True, normalize=False, ax=None):
    if fig == None:
        fig = plt.gcf()
    if rect != None:
        ax = fig.add_axes(rect)
    else:
        if ax == None:
            ax = fig.add_subplot(loc)
    if xlim != None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim((0, W))
    if ylim != None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim((H, 0))
    if normalize:
        ax.set_xlim((-1, 1))
        ax.set_ylim((1, -1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axis(axis)
    ax.set_aspect('equal', 'box')
    if not show_axis:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    return ax

def axes_3d(fig=None, rect=None, loc=111, xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2), xlabel='X', ylabel='Y', zlabel='Z', title='', view=[0, 0], show_axis=True, ax=None, grid=True, normalize=False):
    if fig == None:
        fig = plt.gcf()
    if rect != None:
        ax = fig.add_axes(rect, projection='3d')
    else:
        if ax == None:
            ax = fig.add_subplot(loc, projection='3d')
    if normalize:
        xlim = (-1, 1)
        ylim = (-1, 1)
        zlim = (-1, 1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.view_init(view[0], view[1])
    if not show_axis:
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # ax.axes.zaxis.set_visible(False)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])

    if not grid:
        ax.grid('off')
    return ax

# https://github.com/Vegetebird/MHFormer/blob/main/demo/vis.py
def draw_3d_pose(ax, pose, dataset='h36m', lw=1, markersize=1, markeredgewidth=0.5, alpha=1.0, color=None):
    if dataset == 'aihub':
        joint_pairs = [[15,12],  [12,17],  [12,16],  [17,19],  [19,21], [16,18], [18,20], [12,0], [0,1], [0,2], [2,1], [2,5], [5,8], [1,4], [4,7], [12,17]]
        joint_pairs_left = [[12,16], [16,18], [18,20], [0,1], [1,4], [4,7]]
        joint_pairs_right = [[12,17], [17,19],[19,21], [0,2], [2,5], [5,8]]
    elif dataset == 'h36m_torso':
        joint_pairs = [[0, 1], [0, 4], [4, 11], [11, 14], [14, 0]]
        joint_pairs_left = [[0, 4], [4, 11]]
        joint_pairs_right = [[0, 1], [14, 1]]
    elif dataset == 'torso':
        joint_pairs = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
        joint_pairs_left = [[0, 1], [1, 2]]
        joint_pairs_right = [[3, 4], [4, 0]]
    elif dataset == 'dhdst_torso':
        joint_pairs = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [5, 6], [4, 7], [4, 8]]
        joint_pairs_left = [[0, 2],[4, 7]]
        joint_pairs_right = [[0, 1], [4, 8]]
    elif dataset == 'torso_small': 
        # 0: pelvis, 1: r_hip, 4: l_shoulder, 3: neck, 5:r_shoulder, 2: l_hip
        joint_pairs = [[0, 1], [1, 5], [5, 3], [3, 4], [4, 2], [2, 0]]
        joint_pairs_left = [[3, 4], [4, 2], [2, 0]]
        joint_pairs_right = [[0, 1], [1, 5], [5, 3]]
    elif dataset == 'limb':
        joint_pairs = [[0, 1], [1, 2]]
        joint_pairs_left = [[0, 1]]
        joint_pairs_right = [[1, 2]]
    elif dataset == 'vector':
        joint_pairs = [[0, 1]]
        joint_pairs_left = []
        joint_pairs_right = []
    elif dataset == 'base':
        # 0: pelvis, 1: l_hip, 2: r_hip
        joint_pairs = [[0, 1], [0, 2]]
        joint_pairs_left = [[0, 1]]
        joint_pairs_right = [[0, 2]]
    elif dataset == 'line':
        joint_pairs = [[0, 1]]
        joint_pairs_left = []
        joint_pairs_right = []
    elif dataset == 'kookmin':
        joint_pairs = [[0, 1], [1, 2], [1, 3], [1, 4], [3, 5], [5, 7], [7, 9], [4, 6], [6, 8], [8, 10], [2, 11], [2, 12], [11, 13], [12, 14], [13, 14], [13, 15], [15, 17], [15, 19], [17, 19], [17, 21], [19, 23], [14, 16], [16, 18], [16, 20], [18, 20], [18, 22], [20, 24]]
        joint_pairs_left = [[1, 3], [3, 5], [5, 7], [7, 9], [2, 11], [11, 13], [13, 15], [15, 17], [15, 19], [17, 19], [17, 21], [19, 23]]
        joint_pairs_right = [[1, 4], [4, 6], [6, 8], [8, 10], [2, 12], [12, 14], [14, 16], [16, 18], [16, 20], [18, 20], [18, 22], [20, 24]]
    elif dataset == 'kookmin2':
        joint_pairs = [[0, 1], [1, 2],  [2, 3],  [2, 4],  [2, 5],  [5, 7],  [7, 9],  [9, 11], [4, 6],  [6, 8],  [8, 10], [3, 12], [3, 13], [12, 14], [13, 15], [14, 15], [14, 16], [16, 18], [16, 20], [18, 20], [18, 22], [20, 24], [15, 17], [17, 19], [17, 21], [19, 21], [19, 23], [21, 25] ]
        joint_pairs_left  = [[2, 4], [4, 6], [6, 8], [8, 10], [3, 12], [12, 14], [14, 16], [16, 18], [16, 20], [18, 20], [18, 22], [20, 24]]
        joint_pairs_right = [[2, 5], [5, 7], [7, 9], [9, 11], [3, 13], [13, 15], [15, 17], [17, 19], [17, 21], [19, 21], [19, 23], [21, 25]]
    elif dataset == 'h36m_without_pelvis':
        joint_pairs = [[1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
        joint_pairs_left = [[8, 11], [11, 12], [12, 13], [4, 5], [5, 6]]
        joint_pairs_right = [[8, 14], [14, 15], [15, 16], [1, 2], [2, 3]]
    elif dataset == 'h36m_without_nose':
        joint_pairs = [[0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [8, 10], [8, 13], [10, 11], [11, 12], [13, 14], [14, 15]]
        joint_pairs_left = [[8, 10], [10, 11], [11, 12], [4, 5], [5, 6]]
        joint_pairs_right = [[8, 13], [13, 14], [14, 15], [1, 2], [2, 3]]
    else: # 'h36m'
        joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
        joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
        joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

    if type(color) != type(None):
        color_mid = color
        color_left = color
        color_right = color
    else:
        color_mid = '0.1' # "#00457E"
        color_left = 'r' # "#02315E"
        color_right = 'b' # "#2F70AF"
    
    j3d = pose
    assert j3d.shape[1] > 2, 'Only single 3d pose is supported.'
    
    # plt.tick_params(left = True, right = True , labelleft = False ,
    #                 labelbottom = False, bottom = False)
    # 좀 더 보기 좋게 하기 위해 y <-> z, - 붙임
    for i in range(len(joint_pairs)):
        limb = joint_pairs[i]
        if dataset == 'h36m_cam':
            xs, zs, ys = [-np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
            # xs *= -1
            # ys *= -1
            # zs *= -1
        elif dataset in ['aihub', 'h36m_world', 'h36m_torso', 'torso', 'base', 'fit3d', 'h36m', 'kookmin', 'kookmin2', 'dhdst_torso', 'limb', 'h36m_without_pelvis', 'h36m_without_nose', 'vector', 'torso_small']:
            xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
        if joint_pairs[i] in joint_pairs_left:
            ax.plot(xs, ys, zs, color=color_left, lw=lw, marker='o', markerfacecolor='w', markersize=markersize, markeredgewidth=markeredgewidth, alpha=alpha) 
        elif joint_pairs[i] in joint_pairs_right:
            ax.plot(xs, ys, zs, color=color_right, lw=lw, marker='o', markerfacecolor='w', markersize=markersize, markeredgewidth=markeredgewidth, alpha=alpha) 
        else:
            ax.plot(xs, ys, zs, color=color_mid, lw=lw, marker='o', markerfacecolor='w', markersize=markersize, markeredgewidth=markeredgewidth, alpha=alpha) 
            

# https://github.com/Vegetebird/MHFormer/blob/main/demo/vis.py
def get_2d_pose_image(kps, img=None, H=1080, W=1920, box=None, thickness=10, dataset='h36m'):
    if dataset == 'h36m':
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], 
                       [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], 
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]] 
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool) 
    elif dataset == 'h36m_torso':
        connections = [[0, 1], [0, 4], [4, 11], [11, 14], [14, 0]]
        LR = np.array([0, 0, 1, 1, 0], dtype=bool)
    elif dataset == 'torso': # pelvis l_hip l_shoulder r_shoulder r_hip
        connections = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
        LR = np.array([1, 1, 2, 0, 0], dtype=int)
    elif dataset == 'limb':
        connections = [[0, 1], [1, 2]]
        LR = np.array([2, 2], dtype=int)
    elif dataset == 'base':
        connections = [[0, 1], [0, 2]]
        LR = np.array([1, 0], dtype=int)
    elif dataset == 'twolines': # pelvis l_hip l_shoulder r_shoulder r_hip
        connections = [[1, 4], [2, 3]]
        LR = np.array([0, 1], dtype=int)
    elif dataset == 'line':
        connections = [[0, 1]]
        LR = np.array([2], dtype=int)
    elif dataset == 'aihub':
        connections = [[15,12], [12,17], [12,16], [17,19], [19,21], [16,18],
                       [18,20], [12,0], [0,1], [0,2], [2,1], [2,5], [5,8], [1,4], [4,7]]
        LR = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
    elif dataset == 'coco':
        connections = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], # head
                       [5, 6], [6, 12], [12, 11], [11, 5], # torso
                       [5, 7], [7, 9], # left arm
                       [6, 8], [8, 10], # right arm
                       [11, 13], [13, 15], # left leg
                       [12, 14], [14, 16], # right leg
                      ]
        LR = np.array([0, 1, 2, 0, 1, 0, 1, # head
                       2, 1, 2, 0, # torso
                       0, 0, # left arm
                       1, 1, # right arm
                       0, 0, # left leg
                       1, 1, # right leg
                    ], dtype=bool)

    if img is None:
        img = np.ones((H, W, 3), dtype=np.uint8)*255

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    mcolor = (0, 0, 0)
    colors = [lcolor, rcolor, mcolor]

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), color=colors[LR[j]], thickness=thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)
        if box is not None:
            box = box.astype(int)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    return img

def draw_2d_pose(ax, pose2d, img=None, H=1920, W=1080, box=None, thickness=10, dataset='h36m', normalize=False, color=None):
    # if not normalize:
    #     img = get_2d_pose_image(pose2d, img=img, H=H, W=W, box=box, thickness=thickness, dataset=dataset)
    #     ax.imshow(img)
    # else:
    if dataset == 'h36m':
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], 
                    [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], 
                    [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]] 
        LR = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0], dtype=bool) 
    elif dataset == 'h36m_torso':
        connections = [[0, 1], [0, 4], [4, 11], [11, 14], [14, 0]]
        LR = np.array([0, 0, 1, 1, 0], dtype=bool)
    elif dataset == 'h36m_without_nose':
        # 9 nose -> head
        # 11~16 -> 10~15
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], 
                    [5, 6], [0, 7], [7, 8], [8, 9], 
                    [8, 10], [10, 11], [11, 12], [8, 13], [13, 14], [14, 15]] 
        LR = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0], dtype=bool) 
    elif dataset == 'torso': # pelvis l_hip l_shoulder r_shoulder r_hip
        connections = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
        LR = np.array([1, 1, 2, 0, 0], dtype=int)
    elif dataset == 'limb':
        connections = [[0, 1], [1, 2]]
        LR = np.array([2, 2], dtype=int)
    elif dataset == 'base':
        connections = [[0, 1], [0, 2]]
        LR = np.array([1, 0], dtype=int)
    elif dataset == 'twolines': # pelvis l_hip l_shoulder r_shoulder r_hip
        connections = [[1, 4], [2, 3]]
        LR = np.array([0, 1], dtype=int)
    elif dataset in ['line', 'vector']:
        connections = [[0, 1]]
        LR = np.array([2], dtype=int)
    elif dataset == 'aihub':
        connections = [[15,12], [12,17], [12,16], [17,19], [19,21], [16,18],
                    [18,20], [12,0], [0,1], [0,2], [2,1], [2,5], [5,8], [1,4], [4,7]]
        LR = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
    elif dataset == 'coco':
        connections = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], # head
                    [5, 6], [6, 12], [12, 11], [11, 5], # torso
                    [5, 7], [7, 9], # left arm
                    [6, 8], [8, 10], # right arm
                    [11, 13], [13, 15], # left leg
                    [12, 14], [14, 16], # right leg
                    ]
        LR = np.array([0, 1, 2, 0, 1, 0, 1, # head
                    2, 1, 2, 0, # torso
                    0, 0, # left arm
                    1, 1, # right arm
                    0, 0, # left leg
                    1, 1, # right leg
                    ], dtype=bool)
        
    if type(color) != None:
        colors = [color, color, color]
    else:
        lcolor = 'b' # (255, 0, 0)
        rcolor = 'r' # (0, 0, 255)
        mcolor = 'k' # (0, 0, 0)
        colors = [lcolor, rcolor, mcolor]

    if type(img) != type(None):
        ax.imshow(img)

    kps = pose2d
    for j,c in enumerate(connections):
        start = map(float, kps[c[0]])
        end = map(float, kps[c[1]])
        start = list(start)
        end = list(end)
        ax.plot([start[0], end[0]], [start[1], end[1]], '-', color=colors[LR[j]], linewidth=1.5)


def show_2d_3d(fig_idx, 
               camera, 
               torsos, 
               xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), 
               H=1000, W=1000,
               plot_2d=True, 
               plot_3d=True,
               title='', 
               dataset='torso',
               show=True, 
               proj_line=False, 
               save=None):
    
    torsos = np.array(torsos).reshape(-1, 5, 3)
    camera.update_torso_projection(torsos)
    torsos_projected = projection(torsos, camera.cam_proj)

    fig = plt.figure(fig_idx)
    fig.clear()

    # create axes
    if plot_3d and plot_2d:
        ax1 = axes_3d(fig=fig, loc=121, 
                      xlim=xlim, ylim=ylim, zlim=zlim, 
                      xlabel='X', ylabel='Y', zlabel='Z', 
                      title=title)
        ax2 = axes_2d(fig=fig, loc=122, W=W, H=H)
    elif plot_2d:
        ax2 = axes_2d(fig=fig, loc=111, W=W, H=H)
    elif plot_3d:
        ax1 = axes_3d(fig=fig, loc=111, 
                      xlim=xlim, ylim=ylim, zlim=zlim, 
                      xlabel='X', ylabel='Y', zlabel='Z', 
                      title=title)

    # draw 3d pose
    if plot_3d:
        plt.sca(ax1)
        for torso in torsos:
            draw_3d_pose(ax1, torso, dataset=dataset)

        world_frame = generate_world_frame()
        world_frame.draw3d()
        camera.cam_frame.draw3d()
        camera.image_frame.draw3d()
        camera.image_plane.draw3d()
        camera.Z.draw3d()

        if proj_line:
            for i in range(len(camera.Gs)):
                camera.Gs[i].draw3d(camera.pies[i], C=camera.C)
        else:
            for torso in camera.proj_torsos:
                draw_3d_pose(ax1, torso, dataset=dataset)
    
    # draw 2d pose
    if plot_2d:
        plt.sca(ax2)
        img = None
        for torso_projected in torsos_projected:
            img = get_2d_pose_image(torso_projected, img=img, H=H, W=W, dataset=dataset)
            ax2.imshow(img)

    plt.suptitle(title, fontsize=10)
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    if plot_3d and plot_2d:
        return ax1, ax2
    elif plot_2d:
        return ax2
    elif plot_3d:
        return ax1

def draw_rotation_matrix(ax, origin, rot_mat):
    forward = rot_mat[:, 0]
    left = rot_mat[:, 1]
    up = rot_mat[:, 2]
    draw3d_arrow(origin, forward, head_length=0.1, color="tab:red", ax = ax)
    draw3d_arrow(origin, left, head_length=0.1, color="tab:green", ax = ax)
    draw3d_arrow(origin, up, head_length=0.1, color="tab:orange", ax = ax)

def show3Dtrajectory(traj, ax, final_step, recent=200):
    if final_step < recent:
        ax.plot(traj[:final_step, 0], traj[:final_step, 1], traj[:final_step, 2], color='0.0', label='recent 200 step')
    elif final_step >= recent:
        ax.plot(traj[:(final_step-recent), 0], traj[:(final_step-recent), 1], traj[:(final_step-recent), 2], color='0.5', label='total traj')
        ax.plot(traj[(final_step-recent):final_step, 0], traj[(final_step-recent):final_step, 1], traj[(final_step-recent):final_step, 2], color='0.0', label='recent {} step'.format(recent))

        # if i > 0:
        #     ax.quiver(traj[i-1], traj[i][0])
    #ax.legend()
    return ax

def show2Dtrajectory(traj, ax, final_step, recent=200):
    if final_step < recent:
        ax.plot(traj[:final_step, 0], traj[:final_step, 1], color='0.0', label='recent 200 step')
    elif final_step >= recent:
        ax.plot(traj[:(final_step-recent), 0], traj[:(final_step-recent), 1], color='0.5', label='total traj')
        ax.plot(traj[(final_step-recent):final_step, 0], traj[(final_step-recent):final_step, 1], color='0.0', label='recent 200 step')

        # if i > 0:
        #     ax.quiver(traj[i-1], traj[i][0])
    #ax.legend()
    return ax

def draw_trajectory(ax, traj, final_step, recent=200, dim='3d'):
    if final_step < recent:
        if dim == '3d':
            ax.plot(traj[:final_step, 0], traj[:final_step, 1], traj[:final_step, 2], color='0.0', label='recent 200 step')
        elif dim == '2d':
            ax.plot(traj[:final_step, 0], traj[:final_step, 1], color='0.0', label='recent 200 step')
    elif final_step >= recent:
        if dim == '3d':
            ax.plot(traj[:(final_step-recent), 0], traj[:(final_step-recent), 1], traj[:(final_step-recent), 2], color='0.5', label='total traj')
            ax.plot(traj[(final_step-recent):final_step, 0], traj[(final_step-recent):final_step, 1], traj[(final_step-recent):final_step, 2], color='0.0', label='recent {} step'.format(recent))
        elif dim == '2d':
            ax.plot(traj[:(final_step-recent), 0], traj[:(final_step-recent), 1], color='0.5', label='total traj')
            ax.plot(traj[(final_step-recent):final_step, 0], traj[(final_step-recent):final_step, 1], color='0.0', label='recent 200 step')
        # if i > 0:
        #     ax.quiver(traj[i-1], traj[i][0])
    #ax.legend()
    return ax

def draw_segment(ax, segment, dim='3d', color='r', linewidth=0.15):
    points = segment
    if dim == '3d':
        ax.plot(points[0, 0], points[0, 1], points[0, 2], 'bx',  label='start point')
        ax.plot(points[1:, 0], points[1:, 1], points[1:, 2], color, label='inside', linewidth=linewidth)
        ax.plot(points[-1, 0], points[-1, 1], points[-1, 2], 'kx', label='end point')
    elif dim == '2d':
        ax.plot(points[0, 0], points[0, 1], 'bx',  label='start point')
        ax.plot(points[1:, 0], points[1:, 1], color, label='inside', linewidth=linewidth)
        ax.plot(points[-1, 0], points[-1, 1], 'kx', label='end point')

def draw_segments(ax, segments, dim='3d', color='r', linewidth=0.15):
    for i in range(len(segments)):
        points = segments[i]['points']
        draw_segment(ax, points, dim, color, linewidth)

def draw_multiple_3d_pose(ax, pose3d, dataset='', period=1, final_step=-1):
    if final_step == -1:
        final_step = len(pose3d)
    for i in range(0, final_step, period):
        draw_3d_pose(ax, pose3d[i], dataset=dataset)

def draw_multiple_2d_pose(ax, pose2d, H=1000, W=1000, dataset='', period=1, final_step=-1, img=None):
    if final_step == -1:
        final_step = len(pose2d)
    for i in range(0, final_step, period):
        img = get_2d_pose_image(pose2d[i], img=img, H=H, W=W, dataset=dataset)
    ax.imshow(img)
    return img

def axes_to_compare_pred_gt(fig_idx, xlim=(1.5, 4.5), ylim=(-1.5, 1.5), zlim=(0, 3), view=(90, 180), W=1000, H=1000):
    # --------------------------------------------- Figure setting
    fig = plt.figure(fig_idx, figsize=(10, 10))
    fig.clear()
    ax_gt3d = axes_3d(fig=fig, loc=221, 
                      xlim=xlim, ylim=ylim, zlim=zlim, view=view,
                      xlabel='X', ylabel='Y', zlabel='Z', title='GT 3D')

    ax_gt2d = axes_2d(fig=fig, loc=222, W=W, H=H, 
                      xlabel='X', ylabel='Y', title='GT 2D')

    ax_pred3d = axes_3d(fig=fig, loc=223, 
                        xlim=xlim, ylim=ylim, zlim=zlim, view=view,
                        xlabel='X', ylabel='Y', zlabel='Z', title='Pred 3D')

    ax_pred2d = axes_2d(fig=fig, loc=224, W=W, H=H, 
                        xlabel='X', ylabel='Y', title='Pred 2D')
    
    return ax_gt3d, ax_gt2d, ax_pred3d, ax_pred2d

def plot_to_compare_pred_gt(fig_idx, preds_3d, preds_2d, gts_3d, gts_2d, start=0, length=-1, save=None):
    
    ax_gt3d, ax_gt2d, ax_pred3d, ax_pred2d = axes_to_compare_pred_gt(fig_idx)
    # --------------------------------------------- Plot
    plt.sca(ax_gt3d)
    draw_multiple_3d_pose(ax_gt3d, gts_3d[start:start+length], dataset='torso')
    plt.sca(ax_gt2d)
    draw_multiple_2d_pose(ax_gt2d, gts_2d[start:start+length], dataset='torso')
    plt.sca(ax_pred3d)
    draw_multiple_3d_pose(ax_pred3d, preds_3d[start:start+length], dataset='torso')
    plt.sca(ax_pred2d)
    draw_multiple_2d_pose(ax_pred2d, preds_2d[start:start+length], dataset='torso')

    plt.show()
    if save is not None:
        plt.savefig(save)

def show_whole_segment_trajectories(fig_idx, traj_segment_dataset):
    train_segments = traj_segment_dataset['train']['cam1']['traj_segment']
    test_segments = traj_segment_dataset['test']['cam1']['traj_segment']

    # ---------------------------------------------- Figure setting
    fig = plt.figure(fig_idx)
    fig.clear()
    ax1 = axes_3d(fig=fig, loc=121, 
                xlim=(1.5, 4.5), ylim=(-1.5, 1.5), zlim=(0, 3),
                xlabel='X', ylabel='Y', zlabel='Z', title='train segments', 
                view=(90, 0))
    ax2 = axes_3d(fig=fig, loc=122, 
                xlim=(1.5, 4.5), ylim=(-1.5, 1.5), zlim=(0, 3),
                xlabel='X', ylabel='Y', zlabel='Z', title='test segments', 
                view=(90, 0))
    # ---------------------------------------------- Plot trajectories
    draw_segments(ax1, train_segments)
    draw_segments(ax2, test_segments)

    # ---------------------------------------------- Plot torsos
    #draw_multiple_3d_pose(ax1, train_segments[0]['torsos'], dataset='torso', period=1, final_step=-1)
    #draw_multiple_3d_pose(ax2, test_segments[0]['torsos'], dataset='torso', period=1, final_step=-1)

    plt.suptitle('{} train / {} test segments'.format(len(train_segments), len(test_segments)))
    plt.show()
    
# Remove duplicate labels in legend
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    
def draw_one_segment(fig_idx, segment, cam_proj, period=1, W=1000, H=1000, show_3d_pose=True, show_2d_pose=True):
    # Plot one segment
    points = segment['points'].copy()
    torsos = segment['torsos'].copy()
    rots = segment['rots'].copy()
    torso_projected = segment['torsos_projected'].copy()

    # ---------------------------------------------- Figure setting

    fig = plt.figure(fig_idx)
    fig.clear()
    ax1 = axes_3d(fig=fig, loc=121, 
                xlim=(1.5, 4.5), ylim=(-1.5, 1.5), zlim=(0, 3),
                xlabel='X', ylabel='Y', zlabel='Z', title='3D trajectory',
                view=(45, 180))
    ax2 = axes_2d(fig=fig, loc=122, W=W, H=H, 
                xlabel='X', ylabel='Y', title='2D trajectory')

    # ---------------------------------------------- 3D plot
    plt.sca(ax1)
    draw_segment(ax1, points, dim='3d')
    if show_3d_pose: draw_multiple_3d_pose(ax1, torsos, dataset='torso', period=period) 
    legend_without_duplicate_labels(ax1)
    # ---------------------------------------------- 2D plot
    plt.sca(ax2)
    points_projected = projection(points, cam_proj)
    draw_segment(ax2, points_projected, dim='2d')
    if show_2d_pose: draw_multiple_2d_pose(ax2, torso_projected, dataset='torso', period=period)
    
    plt.show()
    
def draw_bbox(ax, bbox, box_type='xyxy', color="red", linestyle='-', linewidth=1, dim_type='2d'):
    if box_type == 'xyxy':
        x1, y1, x2, y2 = bbox
    elif box_type == 'xxyy':
        x1, x2, y1, y2 = bbox
    elif box_type == 'xywh':
        cx, cy, w, h = bbox
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
    #ax.plot([x1, y1], [x2, y2], color='r')
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width/2
    cy = y1 + height/2
    if dim_type == '2d':
        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=linewidth, linestyle=linestyle, edgecolor=color, facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # ax.plot(x1, y1, 'yx')
        # ax.plot(x2, y2, 'bx')
        # ax.plot(cx, cy, 'kx')
    elif dim_type == '3d':
        upper_left = [x1, y1, 0]
        upper_right = [x2, y1, 0]
        lower_left = [x1, y2, 0]
        lower_right = [x2, y2, 0]
        ax.plot([upper_left[0], upper_right[0]], [upper_left[1], upper_right[1]], [upper_left[2], upper_right[2]], color=color, linewidth=linewidth)
        ax.plot([upper_right[0], lower_right[0]], [upper_right[1], lower_right[1]], [upper_right[2], lower_right[2]], color=color, linewidth=linewidth)
        ax.plot([lower_right[0], lower_left[0]], [lower_right[1], lower_left[1]], [lower_right[2], lower_left[2]], color=color, linewidth=linewidth)
        ax.plot([lower_left[0], upper_left[0]], [lower_left[1], upper_left[1]], [lower_left[2], upper_left[2]], color=color, linewidth=linewidth)
    
def save_h36m_pose_video(pose_list, video_path, dataset='h36m', pose_2d_list=None, gt=[], W=None, H=None, pose_type='3d', fps=30,
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(0, 1), view=(0, 45),
                         centered_xy=False, cam_space=False, on_ground=False, refine_tilt=False,  
                         dynamic_view=False, dual_view=False,
                         imgs=None,
                         show_axis=False, grid=True):
    # pose_list : [N, 17, 3]
    fig = plt.figure()
    fig.clear()
    if pose_type == '3d':
        if dual_view:
            ax1 = axes_3d(fig, loc=121, xlim=xlim, ylim=ylim, zlim=zlim, view=(0, 45), show_axis=show_axis, grid=grid)
            ax2 = axes_3d(fig, loc=122, xlim=xlim, ylim=ylim, zlim=zlim, view=(0, -45), show_axis=show_axis, grid=grid)
        elif dynamic_view: 
            ax = axes_3d(fig, xlim=xlim, ylim=ylim, zlim=zlim, view=(0, 0), show_axis=show_axis, grid=grid)# (0, 10*(sin(2*radians(frame)))+45))
        else:
            ax = axes_3d(fig, xlim=xlim, ylim=ylim, zlim=zlim, view=view, show_axis=show_axis, grid=grid)
    elif pose_type == '2d':
        ax = axes_2d(fig, normalize=True)
    elif pose_type == '2d3d':
        #assert pose_2d_list.all() != None, 'pose_2d_list should be provided'
        assert len(pose_2d_list) == len(pose_list), 'pose_2d_list should have same length as pose_list'
        assert dual_view == False, 'dual_view is not supported in 2d3d mode'
        if dynamic_view: 
            ax = axes_3d(fig, loc=121, xlim=xlim, ylim=ylim, zlim=zlim, view=(0, 0), show_axis=show_axis, grid=grid)# (0, 10*(sin(2*radians(frame)))+45))
        else:
            ax = axes_3d(fig, loc=121, xlim=xlim, ylim=ylim, zlim=zlim, view=view, show_axis=show_axis, grid=grid)
        ax2 = axes_2d(fig, loc=122, normalize=True)

    if type(gt) != type(None):
        assert gt.shape == pose_list.shape, 'gt should have same shape as pose_list'

    videowriter = imageio.get_writer(video_path, fps=fps)
    for frame in tqdm(range(len(pose_list))):
        if type(pose_2d_list) != type(None):
            pose_2d = pose_2d_list[frame].copy() # 1 frame
        pose = pose_list[frame].copy() # 1 frame
        if type(gt) != type(None): pose_gt = gt[frame].copy()
        else: pose_gt = []

        if '3d' in pose_type:
            if centered_xy:
                pose[:, 0] -= pose[get_h36m_keypoint_index('Pelvis'), 0]
                pose[:, 1] -= pose[get_h36m_keypoint_index('Pelvis'), 1]
            if cam_space:
                R1 = Rotation.from_rotvec([-np.pi/2, 0, 0]).as_matrix() # -90 around x-axis
                R2 = Rotation.from_rotvec([0, 0, -np.pi/2]).as_matrix() # -90 around z-axis
                pose = rotate_torso_by_R(pose, R2 @ R1) #+ np.array([0, 0, 0.5])
            if on_ground or refine_tilt:
                l_ankle = pose[get_h36m_keypoint_index('L_Ankle')]
                r_ankle = pose[get_h36m_keypoint_index('R_Ankle')]
                c_ankle = (l_ankle + r_ankle) / 2
                if on_ground:
                    pose -= c_ankle
                if refine_tilt:
                    head = pose[get_h36m_keypoint_index('Head')]
                    tilt = degrees(np.arctan2(head[2] - c_ankle[2], head[0] - c_ankle[0])) - 90
                    R3 = Rotation.from_rotvec([0, np.radians(tilt), 0]).as_matrix() # tilt around y-axis
                    pose = rotate_torso_by_R(pose, R3)
            #pose = get_rootrel_pose(pose) # root-relative
            
            if len(pose_gt) != 0:
                if centered_xy:
                    pose_gt[:, 0] -= pose_gt[get_h36m_keypoint_index('Pelvis'), 0]
                    pose_gt[:, 1] -= pose_gt[get_h36m_keypoint_index('Pelvis'), 1]
                if cam_space:
                    R3 = Rotation.from_rotvec([0, 0, np.pi/2]).as_matrix() # 
                    pose_gt = rotate_torso_by_R(pose_gt, R3 @ R1)
                if on_ground or refine_tilt:
                    l_ankle_gt = pose_gt[get_h36m_keypoint_index('L_Ankle')]
                    r_ankle_gt = pose_gt[get_h36m_keypoint_index('R_Ankle')]
                    c_ankle_gt = (l_ankle_gt + r_ankle_gt) / 2
                    if on_ground:
                        pose_gt -= c_ankle_gt
                    if refine_tilt:
                        head = pose_gt[get_h36m_keypoint_index('Head')]
                        tilt = degrees(np.arctan2(head[2] - c_ankle_gt[2], head[0] - c_ankle_gt[0])) - 90
                        R3 = Rotation.from_rotvec([0, np.radians(tilt), 0]).as_matrix() # tilt around y-axis
                        pose_gt = rotate_torso_by_R(pose_gt, R3)

        if pose_type == '3d':
            if dual_view:
                clear_axes(ax1)
                clear_axes(ax2)
                draw_3d_pose(ax1, pose, dataset=dataset)
                draw_3d_pose(ax2, pose, dataset=dataset)
                ax1.set_title('frame {}'.format(frame))
                ax2.set_title('frame {}'.format(frame))
            else:
                if dynamic_view: 
                    ax.view_init(0, frame/fps*30)
                clear_axes(ax)
                draw_3d_pose(ax, pose, dataset=dataset)
                if type(gt) != type(None):
                    draw_3d_pose(ax, pose_gt, dataset=dataset)
                ax.set_title('frame {}'.format(frame)) 
        elif pose_type == '2d':
            clear_axes(ax)
            if type(imgs) != type(None):
                assert len(imgs) == len(pose_list), 'imgs should have same length as pose_list'
                #img = get_2d_pose_image(pose, img=imgs[frame], W=W, H=H, dataset=dataset)   
                draw_2d_pose(ax, pose, normalize=True, img=imgs[frame])
            else: 
                draw_2d_pose(ax, pose, normalize=True)
        elif pose_type == '2d3d':
            clear_axes(ax)
            if dynamic_view: 
                ax.view_init(0, frame)
            draw_3d_pose(ax, pose, dataset=dataset)
            if len(pose_gt) != 0:
                draw_3d_pose(ax, pose_gt, dataset=dataset)
            ax.set_title('frame {}'.format(frame))
            clear_axes(ax2)
            draw_2d_pose(ax2, pose_2d, normalize=True, dataset=dataset)

        canvas = FigureCanvas(fig)
        canvas.draw()
        image_from_plot = np.array(canvas.renderer._renderer)
        image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)       
        videowriter.append_data(image_from_plot)
    videowriter.close()