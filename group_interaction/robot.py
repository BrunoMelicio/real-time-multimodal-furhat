"""Bounded SDK calls and gaze/head attention; keeps the configured lip-sync voice."""
import asyncio
from concurrent.futures import TimeoutError
from math import hypot, radians, tan


class Robot:
    def __init__(self,host='127.0.0.1'):
        from furhat_realtime_api import FurhatClient
        self.client, self.connected = FurhatClient(host), False

    def call(self,coroutine,timeout=5):
        future = asyncio.run_coroutine_threadsafe(coroutine,self.client._loop)
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise RuntimeError('Furhat SDK timed out. Check the virtual robot and its voice.')

    def start(self):
        self.call(self.client.async_client.connect(),timeout=10)
        self.connected = True
        self.call(self.client.async_client.request_face_headpose(0,0,0,False))
        self.attend(0.,0.)

    def attend(self,yaw,pitch):
        x = tan(radians(float(yaw)))
        y = -hypot(x,1.)*tan(radians(float(pitch)))
        self.call(self.client.async_client.send_event(dict(type='request.attend.location',
            x=float(x),y=float(y),z=1.,slack_yaw=5.,slack_pitch=5.,slack_timeout=800,speed='medium')),timeout=2)

    def nod(self):
        self.call(self.client.async_client.request_gesture_start('Nod',intensity=.4,duration=.7,wait=False))

    def say(self,text):
        self.call(self.client.async_client.request_speak_text(text,wait=True,abort=True),timeout=35)

    def stop_speech(self):
        if self.connected:
            self.call(self.client.async_client.request_speak_stop(),timeout=2)

    def close(self):
        try:
            if self.connected:
                self.call(self.client.async_client.disconnect(),timeout=3)
        finally:
            self.connected = False
            self.client._loop.call_soon_threadsafe(self.client._loop.stop)
            self.client._thread.join(timeout=2)
            if not self.client._thread.is_alive():
                self.client._loop.close()
