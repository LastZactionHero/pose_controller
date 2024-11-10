import pygame
import time

# Initialize Pygame
pygame.init()

# Initialize the joystick module
pygame.joystick.init()

# Check if any joysticks are connected
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("No joystick detected. Please connect a joystick and try again.")
    exit()
else:
    print(f"Number of joysticks detected: {joystick_count}")

# Get the first joystick (adjust the index if necessary)
joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Joystick name: {joystick.get_name()}")

# Get the number of axes, buttons, and hats
num_axes = joystick.get_numaxes()
num_buttons = joystick.get_numbuttons()
num_hats = joystick.get_numhats()

print(f"Number of axes: {num_axes}")
print(f"Number of buttons: {num_buttons}")
print(f"Number of hats: {num_hats}")

try:
    while True:
        # Process event queue
        pygame.event.pump()

        # Read axis values
        axes = []
        for i in range(num_axes):
            axis = joystick.get_axis(i)
            axes.append(axis)
        print(f"Axes: {axes}")

        # Read button states
        buttons = []
        for i in range(num_buttons):
            button = joystick.get_button(i)
            buttons.append(button)
        print(f"Buttons: {buttons}")

        # Read hat (D-pad) positions
        hats = []
        for i in range(num_hats):
            hat = joystick.get_hat(i)
            hats.append(hat)
        print(f"Hats: {hats}")

        # Wait a short time before the next read
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    # Clean up
    joystick.quit()
    pygame.joystick.quit()
    pygame.quit()
