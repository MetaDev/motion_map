import motion_data as md
import numpy as np
import distance_encode_movement as dem
import torch
from os.path import join
import motion_map.CPPN as cn

from pythonosc import osc_message_builder

from pythonosc import udp_client

def init_train(motion_names):
    ############# encoding model
    motions_data=[md.motion_load_qualisys_tsv(mn,skip_rows=10, skip_columns=2) for mn in motion_names]
    #the method motion_load_... returns a  tuple of the shape and the actual data
    motions_data = [md.center_norm_data(motion_data[0]) for motion_data in motions_data]
    motions_data=[motion_data.reshape(motion_data.shape[0],-1) for motion_data in motions_data]

    #we need a list of movements we want to differentiate
    dem.cross_corr_loss_weight=0
    dem.class_loss_weight=0
    dem.disentangled_VAE_loss_weight=5
    dem.reconstruction_loss_weight=1
    dem.n_epochs=10
    X,Y=dem.generate_windows_and_labels(motions_data)
    model=dem.train_mov_enc(X,Y,"cross_fit_enc")
    return model
def load_enc_model(name):
    model = dem.EncDec(motion_feature_size=60, n_classes=2, hidden_size=10)
    model.load_state_dict(torch.load(join("", name), map_location='cpu'))
    return model


def encode_motion(motion_data,enc_model):
    data = np.array(motion_data[len(motion_data)-dem.window_length:len(motion_data)])

    data = torch.tensor(data.transpose(1,0), dtype=torch.float, device=dem.device).unsqueeze(0)
    enc_motion = enc_model.encode(data)
    return enc_motion.squeeze(0)

#########sonification model

def generate_random_son_model():
    #random model
    n_parameters=5

    model_params = dem.hidden_size, ["sin", "hat","id"],n_parameters

    model_type = [lambda model_params: cn.FC_CPPN(*model_params,z_scale=0.1),
                      lambda model_params: cn.Node_CPPN(*model_params,z_scale=0.1)][1]

    model = model_type(model_params).to(cn.device)
    model.random_init_weights()
    return model

# enc_dir="enc_models"
# enc_name=""
# model.load_state_dict(torch.load(join(enc_dir,enc_name),map_location='cpu'))
# #give file path without file extension
# son_model_name=""
# cn.NodeCPPN.load(son_model_name)


############ sonification stuff

def init_sonification():
    ip="127.0.0.1"
    port=8999
    client = udp_client.SimpleUDPClient(ip, port)
    return client

def send_sonification_params(motion_data, enc_model,son_model,client):
    enc_frame=encode_motion(motion_data,enc_model)
    sound = son_model.render_image(enc_frame, 1, 1)[0, 0]
    msg = osc_message_builder.OscMessageBuilder(address="/parameters")

    for s in sound:
        msg.add_arg(float(s))

    msg = msg.build()
    client.send(msg)

def test(motion_names,enc_model_name):
    motions_data=[md.motion_load_qualisys_tsv(mn,skip_rows=10, skip_columns=2) for mn in motion_names]

    #the method motion_load_... returns a  tuple of the shape and the actual data
    motions_data = [md.center_norm_data(motion_data[0]) for motion_data in motions_data]
    motions_data=[motion_data.reshape(motion_data.shape[0],-1) for motion_data in motions_data]

    son_model=generate_random_son_model()
    enc_model=load_enc_model(enc_model_name)
    client=init_sonification()
    for i in range(100):
        send_sonification_params(motions_data[0][:100+i],enc_model,son_model,client)
def main():
    model_motion_files=["jacob_deadlift0034","jacob_powersnatch0037"]
    enc_model_name="cross_fit_enc_60_markers_test.pth"
    # test(model_motion_files,enc_model_name)
    # init_train(model_motion_files)
if __name__ == "__main__": main()
