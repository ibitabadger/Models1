import time
import socket 
from sklearn.datasets import load_iris
data = load_iris()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 9999))

server.listen()

while True:
    client, addr = server.accept()
    print("connection from", addr)
    client.send("you are connected".encode())
    client.send(f"{data['data'][:, 0]}".encode())
    time.sleep(2)
    client.send("you are being disconnected".encode())
    client.close()