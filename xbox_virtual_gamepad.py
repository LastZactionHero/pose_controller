import websocket
import json
import sys

# Define the WebSocket URL of your server
WEBSOCKET_URL = 'ws://144.202.24.235:5555'

# Try to import pyvjoy, if not available, create a stub for dry run mode
try:
    import pyvjoy
    PYVJOY_AVAILABLE = True
except ImportError:
    PYVJOY_AVAILABLE = False
    print("pyvjoy is not available. Running in dry run mode.")

    # Create a stub for pyvjoy.VJoyDevice
    class VJoyDeviceStub:
        def __init__(self, device_id=1):
            self.device_id = device_id

        def set_axis(self, axis, value):
            print(f"Stub: set_axis({axis}, {value})")

        def set_button(self, button_id, value):
            print(f"Stub: set_button({button_id}, {value})")

        def reset(self):
            print("Stub: reset()")

        def data(self):
            return self

        def update(self):
            pass

        @property
        def data(self):
            return self

        @data.setter
        def data(self, value):
            pass

        @property
        def hdata(self):
            return self

        @hdata.setter
        def hdata(self, value):
            pass

    # Assign VJoyDeviceStub to pyvjoy.VJoyDevice for consistency
    pyvjoy = type('pyvjoy', (), {'VJoyDevice': VJoyDeviceStub,
                                 'HID_USAGE_X': 0x30, 'HID_USAGE_Y': 0x31,
                                 'HID_USAGE_Z': 0x32, 'HID_USAGE_RX': 0x33,
                                 'HID_USAGE_RY': 0x34, 'HID_USAGE_RZ': 0x35,
                                 'HID_USAGE_SL0': 0x36, 'HID_USAGE_SL1': 0x37,
                                 'HID_USAGE_WHL': 0x38, 'HID_USAGE_POV': 0x39})

# Initialize virtual joystick or stub
j = pyvjoy.VJoyDevice(1)  # Device ID 1

def on_message(ws, message):
    controller_state = json.loads(message)
    axes = controller_state.get('axes', [])
    buttons = controller_state.get('buttons', [])
    hats = controller_state.get('hats', [])

    # Reset the joystick to neutral position
    j.reset()

    # Update axes
    # vJoy axes values range from 0 to 0x8000 (0 to 32768)
    # We'll map input values from [-1.0, 1.0] to [0, 32768]

    def map_axis(value):
        return int((value + 1) * 0x4000)  # 0x4000 = 16384

    axis_mapping = [
        (pyvjoy.HID_USAGE_X, 0),  # Axis 0 -> X
        (pyvjoy.HID_USAGE_Y, 1),  # Axis 1 -> Y
        (pyvjoy.HID_USAGE_Z, 2),  # Axis 2 -> Z
        (pyvjoy.HID_USAGE_RX, 3), # Axis 3 -> Rx
        (pyvjoy.HID_USAGE_RY, 4), # Axis 4 -> Ry
        (pyvjoy.HID_USAGE_RZ, 5), # Axis 5 -> Rz
    ]

    for axis, idx in axis_mapping:
        if idx < len(axes):
            value = map_axis(axes[idx])
            j.set_axis(axis, value)

    # Update buttons
    # vJoy supports up to 128 buttons, numbered from 1
    for i, button_state in enumerate(buttons):
        button_id = i + 1  # vJoy buttons start at 1
        j.set_button(button_id, int(button_state))

    # Update POV hats
    # vJoy supports up to 4 POV hats
    # Hats are represented as integers:
    # -1: neutral, 0: north, 4500: north-east, 9000: east, ..., 31500: north-west
    pov_mapping = {
        (0, 1): 0,      # North
        (1, 1): 4500,   # North-East
        (1, 0): 9000,   # East
        (1, -1): 13500, # South-East
        (0, -1): 18000, # South
        (-1, -1): 22500,# South-West
        (-1, 0): 27000, # West
        (-1, 1): 31500, # North-West
        (0, 0): -1      # Neutral
    }

    if hats:
        hat = hats[0]  # Assuming single hat
        pov_value = pov_mapping.get(tuple(hat), -1)
        # vJoy uses 0xFFFFFFFF for neutral position
        if pov_value >= 0:
            j.data.wHat = pov_value * 100  # vJoy expects values in hundredths of a degree
        else:
            j.data.wHat = 0xFFFFFFFF  # Neutral position
        j.update()
    else:
        # No hat input; set to neutral
        j.data.wHat = 0xFFFFFFFF
        j.update()

def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Connection Closed")

def on_open(ws):
    print("WebSocket Connection Opened")

if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        WEBSOCKET_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()
