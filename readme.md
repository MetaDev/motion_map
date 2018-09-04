This folder contains the work in progress of the motion map
An algorithm which maps motion data fr0m different sources onto 2D animations.

The main script is motion_to_2D.py
it contains 2 methods:
    -interactive_viz
    this method starts an interactive plotting of visualisations of motions
    arranged in a matrix, where the rows are the same models and the columns same motions
    models can be evolved (only mutation for now) by selecting one of the rows
    the last selected model can be saved
    and if the filename is correct a model can be loaded
    -render_viz:
    in an interactive setting the visualisations are of low resolution
    this method is preferably used on gpu to render an animation and motion in high resolution

The motion data is read and visualised in the motion_data.py script

To parse motion data from a video file run the demo.py script in motion_map/pytorch-pose-hg-3d/src
This script is written for python2
if run on my gpu node the py27 environment needs to be activated
source activate py27
Needs to be run on cuda enabled devices, a test can be run with command below:

CUDA_VISIBLE_DEVICES=1 python demo.py -demo walk.mp4 -loadModel hgreg-3d.pth


CPPN.py contains the evolution and neural network code

There are two additional experimental architectures for CPPN, FC_block_CPPN_test.py and simple_CPPN_test.py
These scripts can also be run to test the visualisations of these models
but these architectures have not been integrated yet in the CPPN.py script.

Other scripts in the folder contain mostly snippets, which could later be useful to make the motion map more usable
such as the client/server for webcam and GUI with plotly