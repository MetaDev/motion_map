#add autoencoder for first layers of z
#encode the kernels of a CNN trained on imagenet with a CPPN and use than that CPPN to generate the kernels used in the deconv
#add evolution: see https://github.com/jinyeom/tf-dppn/blob/master/dppn.py

#for text: text(0.5, 0.5, 'matplotlib', horizontalalignment='center',
# ...      verticalalignment='center', transform=ax.transAxes)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import motion_data as mmd


import CPPN as cn
import time
def interactive_viz():
    # motion_loading=mmd.motion_load_h36m
    # motion_ax=mmd.h36m_plot
    # motion_names=["walking","smoking","eating"]

    motion_loading=mmd.motion_load_qualisys_tsv
    motion_names=["jacob_walking","jacob_walking_1","jacob_walking_2"]
    #motion load returns motion_xyz,motion_xyz_norm,frame_size
    motion_dict = dict([(motion_name, motion_loading(motion_name,skip_rows=12,skip_columns=0)) for motion_name in motion_names])
    # #filter outliers
    # motions_data = [md[0] for md in motion_dict.values()]
    # std=[np.std(md,axis=(0,1)) for md in motions_data]
    # mean=[np.mean(md,axis=(0,1)) for md in motions_data]
    # # motions_data[0][:,:,]
    # print(mean)
    # print(std)
    # def outlier_default(elt,mean,var):
    #     index=np.logical_or(elt <mean-var, elt> mean + var)
    #     elt[index]=mean[index]
    #     return elt
    # print(np.max(test))
    # print(np.min(test))
    # [np.apply_along_axis(lambda coord: outlier_default(coord,mean[0],std[0]) ,2,md )for md in motions_data]
    frames=[slice(0,-1),slice(550,1150),slice(1500,2000)]
    frames = [slice(1500, 2000), slice(550,1150), slice(1500, 2000)]
    motions_data = [md[0][frame] for md,frame in zip(motion_dict.values(),frames)]
    from functools import reduce
    def center_norm_data(motion_xyz_data ,center_idxs=[3,5,13,14]):
        center_pos = reduce(lambda x, y: x + y, [motion_xyz_data[:, i, :] for i in center_idxs]) / len(center_idxs)
        motion_xyz_data = motion_xyz_data - np.repeat(center_pos, motion_xyz_data.shape[1], axis=0).reshape(
            (motion_xyz_data.shape[0], -1, 3))
        range = [np.min(motion_xyz_data, axis=(0, 1)), np.max(motion_xyz_data, axis=(0, 1))]
        motion_xyz_data = (motion_xyz_data - np.min(motion_xyz_data)) / (
                    np.max(motion_xyz_data) - np.min(motion_xyz_data))

        # motion_xyz_data=np.apply_along_axis(lambda coord: (coord-range[0]) / (range[1] - range[0]), 2, motion_xyz_data)
        return motion_xyz_data
    motions_data = [center_norm_data(md[0][frame],center_idxs=[0]) for md,frame in zip(motion_dict.values(),frames)]

    motions_data = [md.reshape(md.shape[0], -1) for md in motions_data]

    nx, ny = (20, 20)
    #get frame size of motion_xyz
    frame_size=motions_data[0].shape[1]
    n_frames=list(motion_dict.values())[0][0].shape[0]
    model_params = frame_size, list(cn.Fun.funs.keys()),3

    model_constr = lambda model_params: cn.Node_CPPN(frame_size, list(cn.Fun.funs.keys()),3,0.05)

    viz_models = [model_constr(model_params).to(cn.device) for i in range(3)]
    for model in viz_models:
        model.random_init_weights()
        model.to(cn.device)
    ims=[]
    axes={}

    fig=plt.figure()
    plt.axis('off')

    for i in range(4):
        for j in range(3):
            if i ==0:
                axes[(i,j)] = fig.add_subplot(4,3,i*3+j+1,projection='3d')
            else:
                axes[(i, j)] = fig.add_subplot(4, 3, i*3 + j + 1)
            axes[(i, j)].set_axis_off()
    #test with random output
    ims=dict([((i,j),axes[1+i,j].imshow(np.ones((nx,ny,3)),animated=True))for i in range(3) for j in range(3)])

    make_3d_graph = lambda i : axes[(0,i)].plot(*motions_data[0].reshape(-1,3).T, linestyle="", marker="o", markersize=2)[0]
    poses = [make_3d_graph(i) for i in range(3)]
    #choose between 0 and 9

    torch_frames={}
    #def init torch frames
    skip_frames = 0
    #should be set to 10 somthing for FC CPPN and maybe different for Node CPPN
    #the scale of z strongly influences the output depending on the weight scale
    for mt,motion in enumerate(motion_names):
        torch_frames[motion]=[]
        for frame in motions_data[mt][::skip_frames + 1]:
                torch_frames[motion].append(cn._tnsr(frame)*2-1)

    frames_text = plt.text(0, 0, "frames: 0")
    last_ind=-1
    def on_key(event):
        nonlocal viz_models
        nonlocal last_ind
        #mutate that are not selected by index
        idx=-1
        try:
            idx=int(event.key) - 1
        except ValueError:
            pass
        if (idx in range(len(viz_models))):
            last_ind = idx
            for i in range(len(viz_models)):
                if i is not idx:
                    viz_models[i] = viz_models[idx].mutate(copy_model=True)
        if (event.key=="r"):
            # print([list(m.graph.nodes)[30:] for m in viz_models])
            [m.random_init_weights() for m in viz_models]
        if (event.key=="d"):
            import datetime
            if last_ind in range(len(viz_models)):
                viz_models[last_ind].save("output/")
                print("model saved")
            else:
                print("Select a model before saving")

        if (event.key=="l"):
            #for loading the model is loaded in the first row and the other are mutations from that first model
            #find model names and load the latest saved
            model_name = "2018-08-23 16:45:34.046992"
            viz_models[0]=cn.Node_CPPN.load("output/"+model_name)
            viz_models[1:]=[viz_models[0].mutate(copy_model=True) for i in range(len(viz_models[1:]))]
            print("model loaded in first row and mutations created")
        if (event.key=="r"):
            for model in viz_models:
                model.random_init_weights()
    def update(f):
        f=f%max([len(motions_data[i]) for i in range(1,3)])
        angle=f%360
        frames_text.set_text("frames: "+str(f))
        if viz_models!=None:
            for j,act in enumerate(list(torch_frames.keys())[:3]):
                if f < len(motions_data[j]):
                    images=[model.render_image(torch_frames[act][f],nx,ny) for model in viz_models]
                    for i in range(3):
                        ims[(i,j)].set_data(images[i])
                    xyz = motions_data[j][::skip_frames+1][f]
                    axes[(0, j)].view_init(10, 200)
                    poses[j].set_data(*xyz.reshape(-1,3).T[0:2])
                    poses[j].set_3d_properties(xyz.reshape(-1,3).T[2])


    # np.save("test",generate_image(torch_frames[actions[0]][0],model))
    anim=FuncAnimation(fig, update, frames=np.arange(0, max([len(motions_data[i]) for i in range(1,3)])), interval=1)

    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

#TODO render to raw video
def render_viz():
    folder="output/"
    model_name="2018-09-12 16:05:10.533266"

    motion_loading = mmd.motion_load_qualisys_tsv
    motion_names = ["jacob_walking", "jacob_walking_1", "jacob_walking_2"]
    # motion load returns motion_xyz,motion_xyz_norm,frame_size
    motion_dict = dict(
        [(motion_name, motion_loading(motion_name, skip_rows=12, skip_columns=0)) for motion_name in motion_names])

    frames = [slice(1500, 2000), slice(550, 1150), slice(1500, 2000)]

    motions_data = [md[0][frame] for md, frame in zip(motion_dict.values(), frames)]
    from functools import reduce
    def center_norm_data(motion_xyz_data, center_idxs=[3, 5, 13, 14]):
        center_pos = reduce(lambda x, y: x + y, [motion_xyz_data[:, i, :] for i in center_idxs]) / len(center_idxs)
        motion_xyz_data = motion_xyz_data - np.repeat(center_pos, motion_xyz_data.shape[1], axis=0).reshape(
            (motion_xyz_data.shape[0], -1, 3))
        range = [np.min(motion_xyz_data, axis=(0, 1)), np.max(motion_xyz_data, axis=(0, 1))]
        motion_xyz_data = (motion_xyz_data - np.min(motion_xyz_data)) / (
                np.max(motion_xyz_data) - np.min(motion_xyz_data))

        # motion_xyz_data=np.apply_along_axis(lambda coord: (coord-range[0]) / (range[1] - range[0]), 2, motion_xyz_data)
        return motion_xyz_data

    motions_data = [center_norm_data(md[0][frame], center_idxs=[0]) for md, frame in zip(motion_dict.values(), frames)]

    motions_data = [md.reshape(md.shape[0], -1) for md in motions_data]

    model_type = [cn.FC_CPPN, cn.Node_CPPN][1]
    model=model_type.load("output/" + model_name)
    model.to(cn.device)
    torch_frames = {}
    # def init torch frames
    skip_frames = 0

    for mt, motion in enumerate(motion_names):
        torch_frames[motion] = []
        for frame in motions_data[mt][::skip_frames + 1]:
            torch_frames[motion].append(cn._tnsr(frame) * 2 - 1)
    width=10
    height=10
    fps=25
    for mt, motion in enumerate(motion_names):
        n_frames=len(torch_frames[motion])
        frames = range(0, min(10,n_frames))

        render_name = model_name+"_"+motion+"_"+str(frames)+"_" + str(width) + "," + str(height)
        import cv2
        out = cv2.VideoWriter(folder+ render_name + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'),fps , (width, height))
        #convert all motion to images
        # iamges=
        if cn.use_cuda:
            batch_size=10000
        else:
            batch_size=-1
        for i in frames:
            print(i)
            # writing to a image array  .
            out.write(np.uint8(model.render_image(torch_frames[motion][i], width,height,batch_size) * 255))
        out.release()
        render_motion=True
        if render_motion:
            motion_data=motions_data[mt]
            fig = plt.figure()

            ax=plt.axes(projection='3d')
            ax.set_axis_off()
            ax.view_init(10, 200)
            pose=ax.plot(*motion_data.reshape(-1, 3).T, linestyle="", marker="o", markersize=2)[0]


            def update(f):
                xyz = motion_data[::skip_frames + 1][f]
                pose.set_data(*xyz.reshape(-1, 3).T[0:2])
                pose.set_3d_properties(xyz.reshape(-1, 3).T[2])

            anim = FuncAnimation(fig, update, frames=frames, interval=1)
            anim.save(folder+motion+"_"+str(frames)+'.gif', dpi=80, writer='imagemagick')

# render_viz()
if __name__ == "__main__": interactive_viz()

