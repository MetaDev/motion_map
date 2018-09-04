
def add_close(window=None):
    '''
    adds close keyboard short for any of the following: \n
    Ctrl+w \n
    Ctrl+F4 \n
    Ctrl+q \n
    \n
    `(for PyQtGraph)`
    '''
    import pyqtgraph as pg

    if window == None:
        pass
    else:
        # Create close shortcut
        cl_keyseq = pg.QtGui.QKeySequence("Ctrl+w")
        cl_short = pg.QtGui.QShortcut(cl_keyseq, window, window.close)

        cl_keyseq_alt = pg.QtGui.QKeySequence("Ctrl+F4")
        cl_short_alt = pg.QtGui.QShortcut(cl_keyseq_alt, window, window.close)

        cl_keyseq_alt2 = pg.QtGui.QKeySequence("Ctrl+q")
        cl_short_alt2 = pg.QtGui.QShortcut(cl_keyseq_alt2, window, window.close)
    return