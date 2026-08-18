# FRAP Tool Interaction Record: Drawing Geometric Patterns (Circle, Square, Triangle)

## User Request
> Draw a circle in the center of the field of view

## Planner Output
- **Module:** `frap`
- **Command:** `Start the active FRAP workflow state.; #Targeted bleaching: Draw a circle centered at field-centered coordinates (0, 0) microns, with radius 10 microns, using a circular trajectory sampled every 2 microns along the circumference.; #Stop the active FRAP workflow state while keeping the session available.`

## User Confirmation
> confirm

## Executor-Generated Code
```python
import math

center_x = 0
center_y = 0
radius = 10
spacing = 2

if radius <= 0:
    say("[ERROR] Radius must be positive.")
    raise RuntimeError("Invalid radius.")

if spacing <= 0:
    say("[ERROR] Sampling spacing must be positive.")
    raise RuntimeError("Invalid spacing.")

circumference = 2 * math.pi * radius
num_points = max(3, int(math.ceil(circumference / spacing)))

points = []
for i in range(num_points):
    theta = (2 * math.pi * i) / num_points
    x = int(round(center_x + radius * math.cos(theta)))
    y = int(round(center_y + radius * math.sin(theta)))
    points.append((x, y))

say(f"[ACTION] Starting FRAP workflow for circular bleaching trajectory with {num_points} points.")
laser_on()

try:
    for x, y in points:
        say(f"[ACTION] Bleaching at ({x}, {y}) microns.")
        laser_position(x, y)
finally:
    say("[ACTION] Stopping FRAP workflow while keeping session available.")
    laser_off()
```

## User Request
> Draw a square a little bit to the left of the circle

## Planner Output
- **Module:** `frap`
- **Command:** `Start the active FRAP workflow state.; #Targeted bleaching: Draw a square to the left of the existing circle centered at (0, 0) microns with radius 10 microns. Use a square perimeter centered at (-25, 0) microns with side length 20 microns, vertices (-35, -10), (-15, -10), (-15, 10), (-35, 10), and close back to (-35, -10). Sample the square trajectory every 2 microns along each edge.; #Stop the active FRAP workflow state while keeping the session available.`

## User Confirmation
> confirm

## Executor-Generated Code
```python
step_um = 2
vertices = [(-35, -10), (-15, -10), (-15, 10), (-35, 10), (-35, -10)]

if step_um <= 0:
    say("[ERROR] Step size must be positive.")
    raise RuntimeError("Invalid step size.")

path = []
for i in range(len(vertices) - 1):
    x1, y1 = vertices[i]
    x2, y2 = vertices[i + 1]
    dx = x2 - x1
    dy = y2 - y1

    if dx != 0 and dy != 0:
        say("[ERROR] Square edges must be horizontal or vertical.")
        raise RuntimeError("Invalid square edge.")

    length = abs(dx) + abs(dy)
    if length % step_um != 0:
        say("[ERROR] Edge length must be divisible by step size.")
        raise RuntimeError("Invalid trajectory spacing.")

    steps = length // step_um
    sx = 0 if dx == 0 else (step_um if dx > 0 else -step_um)
    sy = 0 if dy == 0 else (step_um if dy > 0 else -step_um)

    start_index = 0 if i == 0 else 1
    for j in range(start_index, steps + 1):
        path.append((x1 + sx * j, y1 + sy * j))

say("[ACTION] Starting active FRAP workflow state.")
laser_on()

try:
    say(f"[INFO] Bleaching square perimeter centered at (-25, 0) with {len(path)} points.")
    for x, y in path:
        laser_position(x, y)
    say("[INFO] Square perimeter bleaching complete.")
finally:
    say("[ACTION] Stopping active FRAP workflow state and keeping session available.")
    laser_off()
```

## User Request
> Draw a triangle a little bit to the right of the circle

## Planner Output
- **Module:** `frap`
- **Command:** `Start the active FRAP workflow state.; #Targeted bleaching: Draw an equilateral triangle a little bit to the right of the existing circle centered at (0, 0) microns. Use a triangle centered at field-centered coordinates (25, 0) microns with side length 20 microns, vertices (25, 11.55), (15, -5.77), (35, -5.77), and close back to (25, 11.55). Sample the triangle trajectory every 2 microns along each edge.; #Stop the active FRAP workflow state while keeping the session available.`

## User Confirmation
> confirm

## Executor-Generated Code
```python
laser_on()
say("[ACTION] FRAP workflow started.")

vertices = [
    (25.0, 11.55),
    (15.0, -5.77),
    (35.0, -5.77),
    (25.0, 11.55),
]

spacing_microns = 2.0
sampled_points = []

say("[ACTION] Sampling equilateral triangle trajectory every 2 microns.")

for edge_index in range(len(vertices) - 1):
    x1, y1 = vertices[edge_index]
    x2, y2 = vertices[edge_index + 1]
    dx = x2 - x1
    dy = y2 - y1
    edge_length = (dx * dx + dy * dy) ** 0.5
    steps = max(1, int(round(edge_length / spacing_microns)))

    start_step = 0 if edge_index == 0 else 1
    for step in range(start_step, steps + 1):
        t = step / steps
        x = x1 + dx * t
        y = y1 + dy * t
        sampled_points.append((int(round(x)), int(round(y))))

say(f"[INFO] Generated {len(sampled_points)} bleaching points.")

for x, y in sampled_points:
    laser_position(x, y)

say("[ACTION] Triangle bleaching sequence complete.")
laser_off()
say("[ACTION] FRAP workflow stopped; session remains available.")
```
