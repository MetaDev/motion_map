from __future__ import print_function

import qtm
from qtm.protocol import QRTCommandException, QRTEvent
import xml.etree.cElementTree as ET

class QScript:

    def __init__(self):
        self.qrt = qtm.QRT("10.100.20.174", 22223)
        self.qrt.connect(on_connect=self.on_connect, on_disconnect=self.on_disconnect, on_event=self.on_event)

    def on_connect(self, connection, version):
        print('Connected to QTM with {}'.format(version))
        # Connection is the object containing all methods/commands you can send to qtm
        self.connection = connection
        # Try to start rt from file

        self.connection.start(rtfromfile=False, on_ok=lambda result: self.start_stream(), on_error=self.on_error)

    def on_disconnect(self, reason):
        print(reason)
        # Stops main loop and exits script
        qtm.stop()

    def on_event(self, event):
        # Print event type
        print(event)

    def on_error(self, error):
        error_message = error.getErrorMessage()
        if error_message == "'RT from file already running'":
            # If rt already is running we can start the stream anyway
            self.start_stream()
        else:
            # On other errors we fail
            print(error_message)
            self.connection.disconnect()

    def on_packet(self, packet):
        # All packets has a Framenumber and a timestamp
        print('Framenumber: %d\t Timestamp: %d' % (packet.framenumber, packet.timestamp))

        # all components have some sort of header
        # both header and components are named tuples
        header, markers = packet.get_3d_markers()

        frame = [marker_pos for marker in markers for marker_pos in [marker.x, marker.y, marker.z]]
        self.frames.append(frame)

    def start_stream(self):
        # Start streaming 2d data and register packet callback
        self.connection.stream_frames(on_packet=self.on_packet, components=['2d'])

        # Schedule a call for later to shutdown connection
        # qtm.call_later(5, self.connection.disconnect)

def main():
    # Instantiate our script class
    # We don't need to create a class, you could also store the connection object in a global variable
    QScript()

    # Start the processing loop
    qtm.start()


if __name__ == '__main__':
    main()