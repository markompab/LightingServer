# echo-server.py

import socket
import _thread
from _thread import *

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)



def onNewClient(conn, addr):
    with conn:
        print(f"Connected by {addr}")

        #while True:
        data = conn.recv(1024)

        result ='{\n'\
                +'\"azimuth\" : \"{}\"\n'.format("0") \
                +'\"elevation\" : \"{}\"\n'.format("0") \
                +'"\"EXRImageURL\" : \"{}\"\n '.format("0")\
                +"}";

        bresult =  str.encode(result)
        conn.sendall(bresult)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        _thread.start_new_thread(onNewClient, (conn, addr))
    s.close()