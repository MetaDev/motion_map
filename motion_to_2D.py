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
    motion_loading=mmd.motion_load_h36m
    motion_ax=mmd.h36m_plot
    motion_names=["walking","smoking","eating"]
    #motion load returns motion_xyz,motion_xyz_norm,frame_size
    motion_dict = dict([(motion_name, motion_loading(motion_name)) for motion_name in motion_names])
    motions_data = [mmd.center_norm_data(md[0]) for md in motion_dict.values()]
    motions_data = [md.reshape(md.shape[0], -1) for md in motions_data]


    nx, ny = (20, 20)
    #get frame size of motion_xyz
    frame_size=motions_data[0].shape[1]
    n_frames=list(motion_dict.values())[0][0].shape[0]
    model_params = frame_size, list(cn.Fun.funs.keys()),3
    model_type=[cn.FC_CPPN,cn.Node_CPPN][1]
    model_constr = lambda model_params: model_type(*model_params)

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


    poses = [motion_ax(axes[(0,i)]) for i in range(3)]
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
            viz_models[0]=model_type.load("output/"+model_name)
            viz_models[1:]=[viz_models[0].mutate(copy_model=True) for i in range(len(viz_models[1:]))]
            print("model loaded in first row and mutations created")
        if (event.key=="r"):
            for model in viz_models:
                model.random_init_weights()
    def update(f):
        frames_text.set_text("frames: "+str(f))
        if viz_models!=None:
            for j,act in enumerate(list(torch_frames.keys())[:3]):
                images=[model.render_image(torch_frames[act][f],nx,ny) for model in viz_models]
                for i in range(3):
                    ims[(i,j)].set_data(images[i])
                xyz = motions_data[j][::skip_frames+1][f]
                poses[j].update(xyz.reshape(-1,3))

    # np.save("test",generate_image(torch_frames[actions[0]][0],model))
    anim=FuncAnimation(fig, update, frames=np.arange(0, min([len(motions_data[i]) for i in range(3)])), interval=1)

    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

#TODO render to raw video
def render_viz():
    folder="output/"
    model_name="2018-08-21 11:27:22.347153"
    motion_loading = mmd.motion_load_h36m
    motion_ax = mmd.h36m_plot

    motion_names = ["walking", "walkingdog", "greeting"]
    # motion load returns motion_xyz,motion_xyz_norm,frame_size
    motion_dict = dict([(motion_name, motion_loading(motion_name)) for motion_name in motion_names])
    motions_data = [mmd.center_norm_data(md[0]) for md in motion_dict.values()]
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
    width=100
    height=100
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
        render_motion=False
        if render_motion:
            motion_data=motions_data[mt]
            fig = plt.figure()

            ax=plt.axes(projection='3d')
            ax.set_axis_off()

            pose = motion_ax(ax, motion_data)
            def update(f):
                xyz = motion_data[::skip_frames + 1][f]
                pose.update(xyz.reshape(-1, 3))

            anim = FuncAnimation(fig, update, frames=frames, interval=1)
            anim.save(folder+motion+"_"+str(frames)+'.gif', dpi=80, writer='imagemagick')

# render_viz()
if __name__ == "__main__": interactive_viz()


