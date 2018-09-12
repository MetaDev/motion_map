#add autoencoder for first layers of z
#encode the kernels of a CNN trained on imagenet with a CPPN and use than that CPPN to generate the kernels used in the deconv
#add evolution: see https://github.com/jinyeom/tf-dppn/blob/master/dppn.py

import numpy as np
import motion_data as mmd

import CPPN as cn
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
import UI_utils as ui

import argparse
from pythonosc import osc_message_builder

import torch

from pythonosc import udp_client
import distance_encode_movement as dem
#### motion data
data_type=["bike","crossfit"][0]
if data_type=="bike":
    motion_names=mmd.bike_names[:3]
    motion_dict=dict([(motion_name,mmd.motion_load_qualisys_tsv(motion_name, skip_rows=10, skip_columns=2)) for motion_name in motion_names])
    motions_data =[mmd.center_norm_data_h36m(md[0]) for md in motion_dict.values()]
    motions_data=[md.reshape(md.shape[0],-1) for md in motions_data]
elif data_type=="crossfit":
    pass
model = dem.EncDec(motion_feature_size=84, n_classes=4, hidden_size=10)
from os.path import isfile, join
from os import listdir
enc_dir="enc_models"
enc_name=[f for f in listdir(enc_dir) if isfile(join(enc_dir, f))][3]
print(enc_name)
model.load_state_dict(torch.load(join(enc_dir,enc_name),map_location='cpu'))


enc_motions_data=[]
for motion_data in motions_data:
    data=np.array([motion_data[i - dem.window_length:i] for i in range(dem.window_length,len(motion_data))])
    data = torch.tensor(data.transpose(0, 2, 1), dtype=torch.float, device=dem.device)
    #the input data nees to have the window dimension the last
    enc_motion = model.encode(data)
    enc_motions_data.append(enc_motion)

################## models
n_parameters=5

model_params = dem.hidden_size, ["sin", "hat","id"],n_parameters

model_type = [lambda model_params: cn.FC_CPPN(*model_params,z_scale=0.5),
                  lambda model_params: cn.Node_CPPN(*model_params,z_scale=0.1)][0]

nx, ny = (1, 1)

models = [model_type(model_params).to(cn.device) for i in range(3)]
for model in models:
    model.random_init_weights()


############ sonification stuff

parser = argparse.ArgumentParser()
# parser.add_argument("--ip", default="valerioMac.local",
parser.add_argument("--ip", default="127.0.0.1",
                    help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=8999,
                    help="The port the OSC server is listening on")
args = parser.parse_args()
client = udp_client.SimpleUDPClient(args.ip, args.port)

############# GUI stuff


frame = 0
ipm = mmd.IPEM_plot()

viz_type=["bones","joints"][0]
#bones only works if the joint connection tree is known and configure in motion_data script
if viz_type=="bones":
    bone_plot_fn = lambda motion_data: gl.GLLinePlotItem(pos=np.array(ipm.update(motion_data[frame])),
                                                     color=(1, 0, 1, 1), width=15, mode="lines", antialias=True)
elif viz_type=="joints":
    bone_plot_fn = lambda motion_data: gl.GLScatterPlotItem(pos=np.array(motion_data[frame]),
                                                         color=(1, 0, 1, 1), width=15)

bone_plots = [bone_plot_fn(md) for md in motions_data]



def set_motion_frame(frame):
    for bone_plot, motion_data in zip(bone_plots, motions_data):
        i_ = min(frame, len(motion_data) - 1)
        if viz_type=="bones":
            pos=np.array(ipm.update(motion_data[i_]))
        elif viz_type=="joints":
            pos=motion_data[i_]
        bone_plot.setData(pos=pos)


def update():
    global frame
    global bone_plots
    if play:
        frame += 1
        set_motion_frame(frame)
    b_i = b_group.checkedId()
    if b_i is not -1:
        mtmd = b_id_to_mm[b_i]
        send_pd_motion(frame, enc_motions_data[mtmd[0]], models[mtmd[1]])


### main GUI

app = QtGui.QApplication([])
window = QtGui.QMainWindow()
window.showMaximized()
window.setWindowTitle("Motion to sound")
window.show()
def newOnkeyPressEvent(e):
    global play
    if e.key() == QtCore.Qt.Key_Space:
        play= not play
window.installEventFilter(window)

def newEventFilter(source, event):
    global play
    if event.type() == QtCore.QEvent.ShortcutOverride:
        if event.key() == QtCore.Qt.Key_Space:
            play= not play
            return True
    return QtGui.QMainWindow.eventFilter(window,source,event)
window.eventFilter=newEventFilter

w = pg.GraphicsWindow()
window.setCentralWidget(w)
w.setMinimumSize(384, 360)
w.setBackground("w")
layout = QtGui.QGridLayout()
w.setLayout(layout)
ui.add_close(window)
# create an instance of menu bar
menubar = window.menuBar()

#
# #TOD0
# ####### Menu stuff
#
# def add_model():
#     global models
#     model = model_type().to(cn.device)
#     model.random_init_weights()
#     models.append(model)
# def delete_model():
#     pass
# # same as add model only not with random weights
# def load_model():
#     pass
# def show_model_params():
#     pass
# model_menu = menubar.addMenu('&Model')
# # file menu actions
# def create_menu_item(name, method):
#     action = QtGui.QAction('&' + name, w)
#     action.triggered.connect(method)
#     model_menu.addAction(action)
#
#
# create_menu_item("Add", add_model)
# create_menu_item("Delete", delete_model)
# create_menu_item("Load", load_model)
# create_menu_item("Settings", show_model_params)

### 3D views
views = [gl.GLViewWidget() for i in range(3)]
xgrid = gl.GLGridItem()

for i, (view, bone_plot) in enumerate(zip(views, bone_plots)):
    view.addItem(bone_plot)
    view.addItem(xgrid)
    view.opts['distance'] = 2
    layout.addWidget(view, 0, i + 1)

# buttons
# for model selection
import itertools as it

mm_selected = dict([((mt, md), False) for mt, md in
                    it.product(range(len(motions_data)), range(len(models)))])


def model_motion_select():
    global mm_selected
    global play_btn
    play_btn.setChecked(True)
    play_mode(play_btn)


play = False


def play_mode(play_btn):
    global play
    if play_btn.isChecked():
        play_btn.setText("playing")
    else:
        play_btn.setText("paused")
    play = play_btn.isChecked()


import datetime


def popup(message):
    msg = QtGui.QMessageBox()
    msg.setIcon(QtGui.QMessageBox.Information)
    msg.setText(message)
    msg.setWindowTitle("")
    msg.setStandardButtons(QtGui.QMessageBox.Ok)
    msg.exec_()


def save_model():
    md_sel = [md for md, cb in enumerate(model_select_cb) if cb.isChecked()]
    if len(md_sel) > 1:
        popup("Only one model should be selected when saving.")
        return
    if len(md_sel) == 1:
        name, _ = QtGui.QFileDialog.getSaveFileName(None, 'Save model, add .pickle to file name', "."+
                                                    models[0].file_extension(),
                                                    "Python pickle model(*.{fe})".format(fe=models[0].file_extension()))
        if name is not "":
            models[md_sel[0]].save(name,fe="",dated=False)
    else:
        popup("Select one model for saving.")
    [cb.setChecked(False) for cb in model_select_cb]


## visualisation of parameters


x = np.arange(n_parameters)
y = np.zeros(n_parameters)
in_out_layout = QtGui.QGridLayout()
layout.addLayout(in_out_layout,0, 0, 1, 1)

played_model_param_bar = pg.BarGraphItem(x=x + 0.5, width=0.95,
                                         height=y, brush='r')
param_in_w = pg.PlotWidget()
p_label= QtGui.QLabel("Output Parameters")
in_out_layout.addWidget(p_label, 0, 0, 1, 1)
param_in_w.setMenuEnabled(False)
param_in_w.setXRange(min=0, max=5, padding=0)
param_in_w.setYRange(min=0, max=1, padding=0)
param_in_w.setMaximumSize(100, 10000)
param_in_w.addItem(played_model_param_bar)

in_out_layout.addWidget(param_in_w, 1, 0, 1, 1)

### graphics element for the encoding
x = np.arange(dem.hidden_size)
y = np.zeros(dem.hidden_size)

played_enc_motion_bar = pg.BarGraphItem(x=x + 0.5, width=0.95,
                                         height=y, brush='r')
em_label= QtGui.QLabel("Encoded Motion")
in_out_layout.addWidget(em_label, 2, 0, 1, 1)
enc_in_w = pg.PlotWidget()
#html='<font size="1">Encoded Movement</font>'


enc_in_w.setMenuEnabled(False)

enc_in_w.setXRange(min=0, max=dem.hidden_size - 1, padding=0)
enc_in_w.setYRange(min=-1, max=1, padding=0)
enc_in_w.setMaximumSize(100, 10000)

enc_in_w.addItem(played_enc_motion_bar)

in_out_layout.addWidget(enc_in_w, 3, 0, 1, 1)

import os

from pathlib import Path

def load_model():
    md_sel = [md for md, cb in enumerate(model_select_cb) if cb.isChecked()]
    if len(md_sel) <= 0:
        popup("Select at least one row to load the model in.")

    global models
    file_name, _ = QtGui.QFileDialog.getOpenFileName(None, "Choose a model file", os.getcwd(),
                                                     "Python pickle model(*.{fe})"
                                                     .format(fe=models[0].file_extension()))
    if file_name is not "":
        for md, cb in enumerate(model_select_cb):
            if cb.isChecked():
                models[md].load(file_name,fe="")
                p = Path(file_name)
                cb.setText(p.stem)
                cb.setChecked(False)


def reset_models():
    global models
    for md, cb in enumerate(model_select_cb):
        if cb.isChecked():
            models[md].random_init_weights()


# TODO make proper evolution behaviour mutation + crossover
# add evolution parameters to gui
def evolve():
    global models
    # for now only mutation
    for md, cb in enumerate(model_select_cb):
        if cb.isChecked():
            models[md] = models[md].mutate()


model_select_cb = [QtGui.QCheckBox(w) for i in range(len(models))]
for m_id, model in enumerate(models):
    cb = model_select_cb[m_id]
    cb.setText("model " + str(m_id))
    layout.addWidget(cb, m_id + 1, 0)

btn_layout = QtGui.QGridLayout()
layout.addLayout(btn_layout, len(models) + 2, 0, 1, 4)
load_btn = QtGui.QPushButton("Load")
load_btn.clicked.connect(load_model)
btn_layout.addWidget(load_btn, 0, 0)

save_btn = QtGui.QPushButton("Save")
save_btn.clicked.connect(save_model)
btn_layout.addWidget(save_btn, 0, 1)

evolve_btn = QtGui.QPushButton("Evolve")
evolve_btn.clicked.connect(evolve)
btn_layout.addWidget(evolve_btn, 0, 2)

reset_btn = QtGui.QPushButton("Reset")
reset_btn.clicked.connect(reset_models)
btn_layout.addWidget(reset_btn, 0, 3)

b_group = QtGui.QButtonGroup()
b_id_to_mm = {}
i = 0
for mt, motion in enumerate(motion_dict.keys()):
    for md, model in enumerate(models):
        mm_btn = QtGui.QRadioButton(w)
        mm_btn.setText(model_select_cb[md].text() + " " + motion_names[mt])
        # the identifying tuple is replaced by an int index because it has to
        b_id_to_mm[i] = (mt, md)
        b_group.addButton(mm_btn, i)

        mm_btn.toggled.connect(lambda state: model_motion_select)
        layout.addWidget(mm_btn, md + 1, mt + 1)
        i += 1

# play and pause motion

hor_l = QtGui.QHBoxLayout()
layout.addLayout(hor_l, len(models) + 3, 0, 1, len(motion_dict.keys()))
from PyQt5.QtCore import Qt

sl = QtGui.QSlider(Qt.Horizontal)
sl.setMinimum(dem.window_length)
max_frames = np.max([len(motion)-1 for motion in motions_data])
sl.setMaximum(max_frames)
sl.setValue(0)
# sl.setTickPosition(QtGui.QSlider.TicksBelow)
sl.setTickInterval(1)

hor_l.addWidget(sl)

def valuechange(value):
    global frame
    frame = value
    set_motion_frame(value)


sl.valueChanged.connect(valuechange)
# sl.setTickPosition(QtGui.QSlider.TicksBelow)

play_btn = QtGui.QCheckBox(w)
play_btn.stateChanged.connect(lambda: play_mode(play_btn))
play_btn.setText("paused")
hor_l.addWidget(play_btn)

# TODO stop button

w.show()
# start animation
timer = pg.QtCore.QTimer()

timer.timeout.connect(update)
timer.start(10)

# Set up a timer to call updateGL() every 0 ms
update_gl_timer = pg.QtCore.QTimer()
update_gl_timer.setInterval(0)
update_gl_timer.start()
for view in views:
    update_gl_timer.timeout.connect(view.updateGL)
# start play loop to pure data
import torch


def send_pd_motion(frame, frame_data, model):
    # rescale from 0,1 to -1,1
    # and add a z scale
    frame=min(frame, len(frame_data) - 1)
    torch_frame = frame_data[frame]
    sound = model.render_image(torch_frame, nx, ny)[0, 0]
    msg = osc_message_builder.OscMessageBuilder(address="/parameters")
    played_model_param_bar.setOpts(height=sound)
    played_enc_motion_bar.setOpts(height=frame_data[frame].data.cpu())
    for s in sound:
        msg.add_arg(float(s))

    msg = msg.build()
    client.send(msg)


QtGui.QApplication.instance().exec_()