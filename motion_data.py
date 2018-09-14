from sklearn.preprocessing import StandardScaler
import numpy as np
from h36m_data import load_data
from h36m_data import convert_exp_rot_to_abs_pos_joints
file_names = ["harald_music", "harald_no_music", "jacob_cartwheel", "jacob_clean_jerk", "jacob_dance",
                  "jacob_snatch"]
bike_names = ["rec7_valerio_bike0010_ref","rec7_valerio_bike0011_asym","rec7_valerio_bike0012_shoulders",
              "rec7_valerio_bike0013_asynch"]
#100 FPS
def motion_load_qualisys_tsv(name, skip_rows=10, skip_columns=2):
    with open('data/IPEM/'+name+".tsv") as tsvfile:
        all_text = [line.split('\t') for line in tsvfile.readlines()]

        header=all_text[skip_rows]
        # marker_names = header[1:]
        marker_values= np.array(all_text[skip_rows + 1:]).astype(float)[:,skip_columns:]
        marker_pos=marker_values.reshape(marker_values.shape[0],-1,3)
        #normalisation can be done in different ways and strongly affects the interface
        #average all joint positions according to 8,9 14,15 (shoulder and hips)
        # center_pos=(marker_pos[:,8,:]+marker_pos[:,9,:]+marker_pos[:,14,:]+marker_pos[:,15,:])/4
    return marker_pos,marker_pos.shape[1]

from functools import reduce
def center_norm_data_h36m(motion_xyz_data, center_idxs=[8, 9, 14, 15]):
    center_pos= reduce(lambda x,y: x+y,[motion_xyz_data[:, i, :] for i in center_idxs])/ len(center_idxs)
    motion_xyz_data= motion_xyz_data - np.repeat(center_pos, motion_xyz_data.shape[1], axis=0).reshape((motion_xyz_data.shape[0], -1, 3))
    motion_xyz_data= (motion_xyz_data-np.min(motion_xyz_data)) / (np.max(motion_xyz_data)-np.min(motion_xyz_data))
    return motion_xyz_data
#25 fps
actions = ["directions",  "eating","discussion", "greeting", "phoning",
               "posing", "purchases", "sitting", "sittingdown", "smoking",
               "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

def load_motion_joint_exp_rot(actions=actions,subjects=[5],data_dir="./data/h3.6m/dataset"):
    # Read data from .txt file
    one_hot = False
    data, _ = load_data(data_dir, subjects, actions, one_hot)

    subject = 5
    exp_map_data=dict([(kv[0][:3],kv[1])  for kv in data.items()])
    return exp_map_data
def motion_load_h36m(action_name):
    # actions=np.random.choice(lm.actions,3,replace=False)
    actions = ["directions",  "eating","discussion", "greeting", "phoning",
               "posing", "purchases", "sitting", "sittingdown", "smoking",
               "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
    # data_dir="./sensor_learning/data/h3.6m/dataset"
    data_dir = "data/h3.6m"
    exp_maps = load_motion_joint_exp_rot(actions=actions, data_dir=data_dir)
    # extract only the usable joints from the data and start to work in channel format
    import os
    import pickle
    file_name = "h36m_motion_xyz.pkl"
    if os.path.isfile(file_name):
        motion_xyz = pickle.load(open(file_name, 'rb'))
        print("motion file exists")
    else:
        motion_xyz = {}
        for action in exp_maps.keys():
            motion_xyz[action] = convert_exp_rot_to_abs_pos_joints(exp_maps[action])
        pickle.dump(motion_xyz, open(file_name, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
    motion_keys=[key for key in motion_xyz.keys() if key[1] == action_name]
    key=motion_keys[np.random.choice(range(len(motion_keys)))]
    return motion_xyz[key].reshape((motion_xyz[key].shape[0],-1,3)), list(motion_xyz.values())[0].shape[1]

class IPEM_plot():

    def __init__(self,ax_mot=None,motion_xyz=None,parent_child_joint=None):
        #for dots on the joints
        # self.graph, = ax_mot.plot(*motion_xyz[0].T, linestyle="", marker="o",markersize=2)
        # lines for bones
        self.ax_mot=ax_mot
        self.bones_index = []
        self.plots = []
        if motion_xyz is not None:
            motion_xyz = motion_xyz.reshape(motion_xyz.shape[0], -1, 3)

            init_row = motion_xyz[0]
        if parent_child_joint == None:
            parent_child_joint = dict([(16, [19, 14, 15]), (15, [12]), (14, [13]), (12, [10]), (13, [11]), (10, [0]), (11, [1]),
                                   (19, [17, 18, 22, 23]), (17, [9]), (18, [8]), (9, [6]), (8, [7]), (7, [4])
                                      , (6, [5]), (4, [3]), (22, [23])])
        for connection in list(parent_child_joint.items()):
            parent = connection[0]
            for child in connection[1]:
                self.bones_index.append((parent,child))
                if (ax_mot is not None):
                    p_pos = init_row[parent]
                    c_pos = init_row[child]
                    line = np.stack((p_pos, c_pos), axis=1)
                    # ax.plot returns list, the first element is the line
                    self.plots.append(ax_mot.plot(*line, lw=2, c="b")[0])
        # if a global view is set
        # min_coord = np.array([np.min(motion_xyz.reshape(-1, 3)[:, a]) for a in range(3)])
        # max_coord = np.array([np.max(motion_xyz.reshape(-1, 3)[:, a]) for a in range(3)])
        # ax_mot.set_xlim3d([min_coord[0], max_coord[0]])
        # ax_mot.set_zlim3d([min_coord[2], max_coord[2]])
        # ax_mot.set_ylim3d([min_coord[1], max_coord[1]])
        # ax_mot.set_aspect('equal')
    def update(self,motion_xyz_frame,radius=2):
        # motion
        # self.graph.set_data(*motion_xyz_frame.T[0:2])
        # self.graph.set_3d_properties(motion_xyz_frame.T[2])
        bones=[]
        motion_xyz_frame = motion_xyz_frame.reshape(-1, 3)
        for bone_index in self.bones_index:
            bone = [motion_xyz_frame[bone_index[0]],motion_xyz_frame[bone_index[1]]]
            bones.append(bone)
        if self.ax_mot  is not None:
            for line,bone in zip(self.plots,bones):
                line.set_xdata([bone[0][0], bone[1][0]])
                line.set_ydata([bone[0][1], bone[1][1]])
                line.set_3d_properties([bone[0][2], bone[1][ 2]])
        r = radius;
        if self.ax_mot is not None:
            xroot, yroot, zroot =np.average(motion_xyz_frame,axis=0)

            self.ax_mot.set_xlim3d([-r + xroot, r + xroot])
            self.ax_mot.set_zlim3d([-r + zroot, r + zroot])
            self.ax_mot.set_ylim3d([-r + yroot, r + yroot])

            # self.ax.set_aspect('equal')

            self.ax_mot.set_aspect('equal')
        return bones


class h36m_plot():
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c", limbs_to_visualize=[]):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        self.J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
        # Left / right indicator
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)

        if len(limbs_to_visualize) > 0:
            self.I = self.I[limbs_to_visualize]
            self.J = self.J[limbs_to_visualize]
            self.LR = self.LR[limbs_to_visualize]

        self.ax = ax

        vals = np.zeros((32, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def highlight_limb(self, limb):
        test = np.zeros(self.LR.shape[0])
        test[limb] = 1
        self.LR = test.astype(bool)

    def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
        for i in np.arange(len(self.I)):
            x = np.array([channels[self.I[i], 0], channels[self.J[i], 0]])
            y = np.array([channels[self.I[i], 1], channels[self.J[i], 1]])
            z = np.array([channels[self.I[i], 2], channels[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

        r = 0.4;
        xroot, yroot, zroot = channels[0, 0], channels[0, 1], channels[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])

        self.ax.set_aspect('equal')


