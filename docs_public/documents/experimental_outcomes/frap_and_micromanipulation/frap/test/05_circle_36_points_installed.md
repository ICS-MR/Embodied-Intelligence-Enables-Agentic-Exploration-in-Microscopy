# FRAP Tool Interaction Record: Sequential Bleaching of 36 Points on a Circle (Installed State)

## User Request
> Imaging task: Use the FRAP tool to sequentially photobleach 36 evenly spaced points along a circle centered at the field-of-view origin (0, 0) with a radius of 50 micrometers.

## Planner Output
- **Module:** `frap`
- **Command:** `Activate the fluorescence recovery system; #Perform precision photobleaching at 36 evenly spaced points along a circle centered at (0, 0) with a radius of 50 micrometers; #Deactivate the fluorescence recovery system`

## User Confirmation
> confirm

## Executor-Generated Code
```python
import math

laser_on()
say("Starting precision photobleaching on a circle.")

center_x = 0
center_y = 0
radius = 50
num_points = 36

for i in range(num_points):
    angle = 2 * math.pi * i / num_points
    x = center_x + radius * math.cos(angle)
    y = center_y + radius * math.sin(angle)
    
    # Round to nearest integer as laser_position expects int
    x_int = int(round(x))
    y_int = int(round(y))
    
    laser_position(x_int, y_int)
    say(f"Bleached point {i+1}/{num_points} at ({x_int}, {y_int})")

say("Photobleaching sequence completed.")
laser_off()
```
