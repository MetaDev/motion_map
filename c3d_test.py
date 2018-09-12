import c3d
import c3d_viewer
import numpy as np
reader = c3d.Reader(open('FMfootre0001verw.c3d', 'rb'))
# for i, points, analog in reader.read_frames():
#     print('frame {}: point {}, analog {}'.format(
#         i, points.shape, analog.shape))
#     print('Frame {}: {}'.format(i, points.round(2)))

c3d_viewer.Viewer(c3d_reader=reader).mainloop()