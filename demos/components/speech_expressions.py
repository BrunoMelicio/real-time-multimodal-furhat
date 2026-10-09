"""A simple greeting with a smile and raised eyebrows."""

from furhat_realtime_api import FurhatClient


# 127.0.0.1 means Virtual Furhat is running on this same computer.
furhat = FurhatClient("127.0.0.1")

try:
    furhat.connect()
    print("Connected to Furhat!")

    # Start a smile, then speak without waiting for the gesture to finish.
    print("Smiling...")
    furhat.request_gesture_start("Smile", duration=3.0, wait=False)
    furhat.request_speak_text("Hello! It is nice to see you.", wait=True)

    # Raise the eyebrows to illustrate surprise, then say the next line.
    print("Raising eyebrows...")
    furhat.request_gesture_start("BrowRaise", duration=3.0, wait=False)
    furhat.request_speak_text("Wow! I can speak and show expressions!", wait=True)

    print("Demo v2 complete!")
except Exception as error:
    print(f"Demo failed: {error}")
    print("Check that Virtual Furhat is running and Realtime API is enabled.")
    raise SystemExit(1)
finally:
    # Close the connection even if something goes wrong.
    furhat.disconnect()
