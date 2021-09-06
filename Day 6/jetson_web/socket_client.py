import socketio
import time

if __name__ == "__main__":
    sio = socketio.Client()

    @sio.event
    def connect():
        print("Connection established")
    
    @sio.event
    def disconnect():
        print("Disconnected from server")

    sio.connect("http://localhost:8000")

    while True:
        sio.emit('sensor', {'temp': 32.3, 'photo': 20.0, 'vr': 3.8})

        time.sleep(1);