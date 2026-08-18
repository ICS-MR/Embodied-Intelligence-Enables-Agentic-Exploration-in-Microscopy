# FRAP Tool Interaction Record: Writing the Letters "HIT"

## User Request
> Write the letters "HIT" in the field of view.

## Planner Output
- **Module:** `frap`
- **Command:** `Start the active FRAP workflow state.; #Targeted bleaching: Write the letters 'HIT' in the current field of view using field-centered micron coordinates, with letter height 40 µm, letter width 18 µm, inter-letter spacing 8 µm, stroke sampling spacing 2 µm, and bleaching performed along generated line trajectories. Bleach letter H with vertical strokes from (-32,-20) to (-32,20) and (-14,-20) to (-14,20), and crossbar from (-32,0) to (-14,0). Bleach letter I with top bar from (-5,20) to (13,20), center stroke from (4,20) to (4,-20), and bottom bar from (-5,-20) to (13,-20). Bleach letter T with top bar from (22,20) to (40,20), and center stroke from (31,20) to (31,-20).; #Stop the active FRAP workflow state while keeping the session available.`

## User Confirmation
> confirm

## Executor-Generated Code
```python
stroke_spacing_um = 2

strokes = [
    ("H left vertical", (-32, -20), (-32, 20)),
    ("H right vertical", (-14, -20), (-14, 20)),
    ("H crossbar", (-32, 0), (-14, 0)),
    ("I top bar", (-5, 20), (13, 20)),
    ("I center stroke", (4, 20), (4, -20)),
    ("I bottom bar", (-5, -20), (13, -20)),
    ("T top bar", (22, 20), (40, 20)),
    ("T center stroke", (31, 20), (31, -20)),
]

def generate_line_points(start, end, spacing):
    if spacing <= 0:
        say("[ERROR] Stroke sampling spacing must be positive.")
        raise RuntimeError("Invalid stroke sampling spacing.")

    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    length = (dx * dx + dy * dy) ** 0.5
    steps = int(length // spacing)

    points = []
    for i in range(steps + 1):
        distance = i * spacing
        if distance > length:
            distance = length
        fraction = 0 if length == 0 else distance / length
        x = int(round(x1 + dx * fraction))
        y = int(round(y1 + dy * fraction))
        if not points or points[-1] != (x, y):
            points.append((x, y))

    endpoint = (int(round(x2)), int(round(y2)))
    if points[-1] != endpoint:
        points.append(endpoint)

    return points

laser_on()
try:
    say("[ACTION] Started active FRAP workflow state.")
    say("[INFO] Generating HIT bleaching trajectories with 2 um sampling.")

    total_points = 0
    for stroke_name, start, end in strokes:
        points = generate_line_points(start, end, stroke_spacing_um)
        say(f"[ACTION] Bleaching {stroke_name}: {len(points)} sampled positions.")
        for x, y in points:
            laser_position(x, y)
        total_points += len(points)

    say(f"[INFO] Completed HIT targeted bleaching across {total_points} laser positions.")
finally:
    laser_off()
    say("[ACTION] Stopped active FRAP workflow state; session remains available.")
```
