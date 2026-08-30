# FRAP Tool Interaction Record: Drawing an Archimedean Spiral

## User Request
> Draw a spiral line in the center of the field of view

## Planner Output
- **Module:** `frap`
- **Command:** `Start the active FRAP workflow state.; #Targeted bleaching: Draw an Archimedean spiral line centered at the field-of-view center coordinate (0, 0) microns; use 3 full turns, maximum radius 20 microns, clockwise direction, and 1 micron point spacing; bleach each sampled coordinate once in spiral order.; #Stop the active FRAP workflow state while keeping the session available.`

## User Confirmation
> confirm

## Executor-Generated Code
```python
import math

turns = 3
max_radius = 20
point_spacing = 1
center_x = 0
center_y = 0
clockwise = True

if turns <= 0:
    say("[ERROR] turns must be positive.")
    raise RuntimeError("turns must be positive")
if max_radius <= 0:
    say("[ERROR] max_radius must be positive.")
    raise RuntimeError("max_radius must be positive")
if point_spacing <= 0:
    say("[ERROR] point_spacing must be positive.")
    raise RuntimeError("point_spacing must be positive")

theta_max = 2 * math.pi * turns
b = max_radius / theta_max

points = []
theta = 0.0
last_point = None

while theta < theta_max:
    r = b * theta
    x = center_x + r * math.cos(theta)
    y_sign = -1 if clockwise else 1
    y = center_y + y_sign * r * math.sin(theta)

    point = (int(round(x)), int(round(y)))
    if point != last_point:
        points.append(point)
        last_point = point

    dtheta = point_spacing / (b * math.sqrt(theta * theta + 1))
    theta += dtheta

end_x = center_x + max_radius * math.cos(theta_max)
end_y = center_y + (-1 if clockwise else 1) * max_radius * math.sin(theta_max)
end_point = (int(round(end_x)), int(round(end_y)))
if points[-1] != end_point:
    points.append(end_point)

say(f"[ACTION] Starting FRAP workflow for clockwise Archimedean spiral bleaching.")
say(f"[INFO] Generated {len(points)} sampled laser positions.")

laser_on()
try:
    for x, y in points:
        laser_position(x, y)
    say("[INFO] Spiral bleaching sequence completed.")
finally:
    laser_off()
    say("[ACTION] FRAP workflow stopped; session remains available.")
```
