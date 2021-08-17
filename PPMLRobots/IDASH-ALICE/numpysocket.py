#!/usr/bin/env python
# https://github.com/sabjorn/NumpySocket
import socket
import numpy as np
from io import BytesIO, StringIO


class NumpySocket():
    def __init__(self):
        self.address = 0
        self.port = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.type = None  # server or client

    # def __del__(self):
    #     if self.type is "client":
    #         self.endClient()
    #     if self.type is "server":
    #         self.endServer()
    #     print self.id, 'closed'

    def startServer(self, address, port):
        self.type = "server"
        self.address = address
        self.port = port
        try:
            self.socket.connect((self.address, self.port))
            print('Connected to %s on port %s' % (self.address, self.port))
        except socket.error as e:
            print('Connection to %s on port %s failed: %s' % (self.address, self.port, e))
            return

    def endServer(self):
        self.socket.shutdown(1)
        self.socket.close()

    def sendNumpy(self, image):
        if self.type != "server":
            print("Not setup as a server")
            return

        if not isinstance(image, np.ndarray):
            print('not a valid numpy image')
            return
        f = BytesIO()
        #print(type(f))
        np.savez_compressed(f, frame=image)
        f.seek(0)
        out = f.read()
        val = "{0}:".format(len(f.getvalue()))  # prepend length of array
        val = str.encode(val)
        out = val + out
        try:
            self.socket.sendall(out)
        except Exception:
            exit()
        print('np array sent')

    def startClient(self, port):
        self.type = "client"
        self.address = ''
        self.port = port

        self.socket.bind((self.address, self.port))
        self.socket.listen(1)
        print('waiting for a connection...')
        self.client_connection, self.client_address = self.socket.accept()
        print('connected to ', self.client_address[0])

    def endClient(self):
        self.client_connection.shutdown(1)
        self.client_connection.close()

    def recieveNumpy(self):
        if self.type != "client":
            print("Not setup as a client")
            return

        length = None
        ultimate_buffer = b""
        while True:
            data = self.client_connection.recv(1024)
            ultimate_buffer += data
            if len(ultimate_buffer) == length:
                break
            while True:
                if length is None:
                    if b':' not in ultimate_buffer:
                        break
                    # remove the length bytes from the front of ultimate_buffer
                    # leave any remaining bytes in the ultimate_buffer!
                    #print(type(ultimate_buffer))
                    length_str, ignored, ultimate_buffer = ultimate_buffer.partition(b':')
                    length = int(length_str)
                    #print(length)
                if len(ultimate_buffer) < length:
                    break
                # split off the full message from the remaining bytes
                # leave any remaining bytes in the ultimate_buffer!
                ultimate_buffer = ultimate_buffer[length:]
                length = None
                break
        print(len(ultimate_buffer))
        final_image = np.load(BytesIO(ultimate_buffer))['frame']
        print('np array received')

        return final_image
