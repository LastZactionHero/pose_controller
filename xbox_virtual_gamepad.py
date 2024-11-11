import websocket
import json
import sys

# Define the WebSocket URL of your server
WEBSOCKET_URL = 'ws://144.202.24.235:5555'

# Try to import vgamepad, if not available, create a stub for dry run mode
try:
    import vgamepad as vg
    VGAMEPAD_AVAILABLE = True
except ImportError:
    VGAMEPAD_AVAILABLE = False
    print("vgamepad is not available. Running in dry run mode.")

    # Create a stub for vg.VX360Gamepad
    class GamepadStub:
        def __init__(self):
            pass

        def right_joystick_float(self, x_value_float, y_value_float):
            print(f"Stub: right_joystick_float({x_value_float}, {y_value_float})")

        def left_joystick_float(self, x_value_float, y_value_float):
            print(f"Stub: left_joystick_float({x_value_float}, {y_value_float})")

        def left_trigger_float(self, value_float):
            print(f"Stub: left_trigger_float({value_float})")

        def right_trigger_float(self, value_float):
            print(f"Stub: right_trigger_float({value_float})")

        def press_button(self, button):
            print(f"Stub: press_button({button})")

        def release_button(self, button):
            print(f"Stub: release_button({button})")

        def press_dpad_button(self, direction):
            print(f"Stub: press_dpad_button({direction})")

        def release_dpad(self):
            print("Stub: release_dpad()")

        def update(self):
            pass  # No action needed in dry run mode

    # Assign GamepadStub to vg.VX360Gamepad for consistency
    vg = type('vg', (), {'VX360Gamepad': GamepadStub, 'XUSB_BUTTON': type('XUSB_BUTTON', (), {}), 'XUSB_DPAD': type('XUSB_DPAD', (), {})})

    # Define constants for buttons and D-pad directions in the stub
    vg.XUSB_BUTTON.XUSB_GAMEPAD_A = 'A'
    vg.XUSB_BUTTON.XUSB_GAMEPAD_B = 'B'
    vg.XUSB_BUTTON.XUSB_GAMEPAD_X = 'X'
    vg.XUSB_BUTTON.XUSB_GAMEPAD_Y = 'Y'
    vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER = 'LEFT_SHOULDER'
    vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER = 'RIGHT_SHOULDER'
    vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK = 'BACK'
    vg.XUSB_BUTTON.XUSB_GAMEPAD_START = 'START'
    vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB = 'LEFT_THUMB'
    vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB = 'RIGHT_THUMB'

    vg.XUSB_DPAD.UP = 'UP'
    vg.XUSB_DPAD.DOWN = 'DOWN'
    vg.XUSB_DPAD.LEFT = 'LEFT'
    vg.XUSB_DPAD.RIGHT = 'RIGHT'
    vg.XUSB_DPAD.UP_LEFT = 'UP_LEFT'
    vg.XUSB_DPAD.UP_RIGHT = 'UP_RIGHT'
    vg.XUSB_DPAD.DOWN_LEFT = 'DOWN_LEFT'
    vg.XUSB_DPAD.DOWN_RIGHT = 'DOWN_RIGHT'
    vg.XUSB_DPAD.NONE = 'NONE'

# Initialize virtual gamepad or stub
gamepad = vg.VX360Gamepad()

def on_message(ws, message):
    controller_state = json.loads(message)
    axes = controller_state.get('axes', [])
    buttons = controller_state.get('buttons', [])
    hats = controller_state.get('hats', [])

    # Update axes (assuming standard mapping)
    # Axes mapping may vary depending on the controller
    # Here we assume axes[0] and axes[1] are left joystick
    # and axes[2] and axes[3] are right joystick
    if len(axes) >= 4:
        # Left joystick
        left_x = axes[0]
        left_y = axes[1]
        gamepad.left_joystick_float(x_value_float=left_x, y_value_float=left_y)

        # Right joystick
        right_x = axes[2]
        right_y = axes[3]
        gamepad.right_joystick_float(x_value_float=right_x, y_value_float=right_y)

    # Update triggers (if axes[4] and axes[5] are triggers)
    if len(axes) >= 6:
        # Triggers typically range from -1.0 to 1.0, adjust as needed
        left_trigger = (axes[4] + 1) / 2  # Convert from [-1, 1] to [0, 1]
        right_trigger = (axes[5] + 1) / 2  # Convert from [-1, 1] to [0, 1]
        gamepad.left_trigger_float(value_float=left_trigger)
        gamepad.right_trigger_float(value_float=right_trigger)

    # Update buttons
    # Map the buttons to virtual gamepad buttons
    # The mapping depends on the controller; here's an example for Xbox controller
    # Assuming buttons[0] is 'A', buttons[1] is 'B', buttons[2] is 'X', buttons[3] is 'Y', etc.
    button_mapping = {
        0: vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
        1: vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
        2: vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
        3: vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
        4: vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
        5: vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
        6: vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
        7: vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
        8: vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
        9: vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
    }

    for i, button_state in enumerate(buttons):
        if i in button_mapping:
            button = button_mapping[i]
            if button_state:
                gamepad.press_button(button=button)
            else:
                gamepad.release_button(button=button)

    # Update D-pad (hats)
    # Assuming hats is a list of tuples, e.g., [(0, 1)]
    # Map hat values to D-pad directions
    dpad_mapping = {
        (0, 1): vg.XUSB_DPAD.UP,
        (1, 1): vg.XUSB_DPAD.UP_RIGHT,
        (1, 0): vg.XUSB_DPAD.RIGHT,
        (1, -1): vg.XUSB_DPAD.DOWN_RIGHT,
        (0, -1): vg.XUSB_DPAD.DOWN,
        (-1, -1): vg.XUSB_DPAD.DOWN_LEFT,
        (-1, 0): vg.XUSB_DPAD.LEFT,
        (-1, 1): vg.XUSB_DPAD.UP_LEFT,
        (0, 0): vg.XUSB_DPAD.NONE,
    }

    if hats:
        hat = hats[0]  # Assuming single hat
        direction = dpad_mapping.get(hat, vg.XUSB_DPAD.NONE)
        if direction != vg.XUSB_DPAD.NONE:
            gamepad.press_dpad_button(direction=direction)
        else:
            gamepad.release_dpad()
    else:
        gamepad.release_dpad()

    # Send updates to the virtual gamepad
    gamepad.update()

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
