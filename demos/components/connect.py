"""Connect to Virtual Furhat on this computer and say hello."""

from furhat_realtime_api import FurhatClient


furhat = FurhatClient("127.0.0.1")

try:
    furhat.connect()
    print("Connected to Furhat!")
    furhat.request_speak_text(
        "Hello! Your Python connection to Furhat works.", wait=True
    )
    print("Speech finished. Demo complete!")
except Exception as error:
    print(f"Demo failed: {error}")
    print("Start Virtual Furhat and enable Realtime API in its web settings.")
    raise SystemExit(1)
finally:
    furhat.disconnect()
