# FRAP Tool Interaction Record: Photobleaching a Cell and Repeat at Lower Intensity

## User Request
> Photobleach the entire area of the cell closest to the center of the field of view.

## Planner Output
- **Module:** `frap`
- **Command:** `Start the active FRAP workflow state.; #Cell discovery: Detect usable cells in the current field of view and return each cell identifier with field-centered micron coordinates.; #Cell contour mapping: For all detected usable cells, extract fitted ellipse boundary point sequences; select the cell whose center coordinate has the smallest distance to the field center coordinate (0, 0) microns.; #Targeted bleaching: Photobleach the entire selected cell area by generating and executing a filled raster trajectory over the interior of the selected cell contour using 1.0 micron point spacing, including boundary points and all interior points, with coordinates expressed as field-centered microns.; #Stop the active FRAP workflow state while keeping the session available.`

## User Confirmation
> confirm

## Executor-Generated Code
```python
def point_on_segment(px, py, ax, ay, bx, by, tolerance=1e-9):
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > tolerance:
        return False

    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= tolerance


def point_in_or_on_polygon(px, py, polygon):
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1

    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[j]

        if point_on_segment(px, py, ax, ay, bx, by):
            return True

        intersects = ((ay > py) != (by > py)) and (
            px < (bx - ax) * (py - ay) / (by - ay) + ax
        )
        if intersects:
            inside = not inside

        j = i

    return inside


def rounded_target(x, y):
    return int(round(x)), int(round(y))


laser_on()

try:
    say("[ACTION] Starting active FRAP workflow state.")

    detection_result = cell_detection()
    detected_cells = detection_result.get("cells", [])

    if not detected_cells:
        say("[ERROR] No usable cells detected in the current field of view.")
        raise RuntimeError("No usable cells detected.")

    say(f"[INFO] Detected {len(detected_cells)} usable cells.")
    for cell in detected_cells:
        say(
            f"[INFO] Cell {cell.get('cell_id')}: "
            f"x={cell.get('x')} microns, y={cell.get('y')} microns."
        )

    contour_result = cell_contour_extraction()
    contour_cells = contour_result.get("cells", [])

    if not contour_cells:
        say("[ERROR] No usable cell contours extracted.")
        raise RuntimeError("No usable cell contours extracted.")

    contours_by_id = {}
    for cell in contour_cells:
        cell_id = cell.get("cell_id")
        points = cell.get("points", [])
        if cell_id is not None and len(points) >= 3:
            contours_by_id[cell_id] = points

    usable_detected_cells = [
        cell for cell in detected_cells if cell.get("cell_id") in contours_by_id
    ]

    if not usable_detected_cells:
        say("[ERROR] No detected cells have matching usable contours.")
        raise RuntimeError("No detected cells have matching usable contours.")

    selected_cell = min(
        usable_detected_cells,
        key=lambda cell: (cell.get("x", 0) * cell.get("x", 0))
        + (cell.get("y", 0) * cell.get("y", 0)),
    )

    selected_cell_id = selected_cell.get("cell_id")
    selected_x = selected_cell.get("x")
    selected_y = selected_cell.get("y")
    contour = contours_by_id[selected_cell_id]

    say(
        f"[ACTION] Selected cell {selected_cell_id}, nearest to field center: "
        f"x={selected_x} microns, y={selected_y} microns."
    )

    polygon = []
    for point in contour:
        if len(point) >= 2:
            polygon.append((float(point[0]), float(point[1])))

    if len(polygon) < 3:
        say("[ERROR] Selected cell contour has fewer than 3 valid points.")
        raise RuntimeError("Selected cell contour is invalid.")

    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)

    spacing = 1.0
    start_x = int(min_x // spacing)
    end_x = int(max_x // spacing)
    if start_x * spacing < min_x:
        start_x += 1
    if end_x * spacing < max_x:
        end_x += 1

    start_y = int(min_y // spacing)
    end_y = int(max_y // spacing)
    if start_y * spacing < min_y:
        start_y += 1
    if end_y * spacing < max_y:
        end_y += 1

    targets = []
    seen_targets = set()

    for point in polygon:
        target = rounded_target(point[0], point[1])
        if target not in seen_targets:
            seen_targets.add(target)
            targets.append(target)

    for y_index in range(start_y, end_y + 1):
        row_targets = []
        y = y_index * spacing

        for x_index in range(start_x, end_x + 1):
            x = x_index * spacing
            if point_in_or_on_polygon(x, y, polygon):
                target = rounded_target(x, y)
                if target not in seen_targets:
                    seen_targets.add(target)
                    row_targets.append(target)

        if y_index % 2 == 0:
            targets.extend(row_targets)
        else:
            targets.extend(reversed(row_targets))

    if not targets:
        say("[ERROR] No bleaching targets generated for selected cell.")
        raise RuntimeError("No bleaching targets generated.")

    say(
        f"[ACTION] Photobleaching selected cell {selected_cell_id} with "
        f"{len(targets)} boundary/interior targets at 1.0 micron spacing."
    )

    for x, y in targets:
        laser_position(x, y)

    say("[INFO] Targeted bleaching complete.")

finally:
    laser_off()
    say("[ACTION] Stopped active FRAP workflow state; session remains available.")
```

## User Request
> Carry out the same cell photobleaching operation with lower intensity.

## Planner Output
- **Module:** `frap`
- **Command:** `Start the active FRAP workflow state.; #Cell discovery: Detect usable cells in the current field of view and return each cell identifier with field-centered micron coordinates.; #Cell contour mapping: For all detected usable cells, extract fitted ellipse boundary point sequences; select the cell whose center coordinate has the smallest distance to the field center coordinate (0, 0) microns.; #Targeted bleaching: Photobleach the entire selected cell area at lower intensity by generating and executing a filled raster trajectory over the interior of the selected cell contour using 2.0 micron point spacing, including boundary points and all interior points, with coordinates expressed as field-centered microns.; #Stop the active FRAP workflow state while keeping the session available.`

## User Confirmation
> confirm

## Executor-Generated Code
```python
laser_on()
try:
    say("[ACTION] Starting active FRAP workflow state.")

    detection_result = cell_detection()
    detected_cells = detection_result.get("cells", [])
    say(f"[INFO] Detected {len(detected_cells)} usable cells.")

    if not detected_cells:
        say("[ERROR] No usable cells detected in the current field of view.")
        raise RuntimeError("No usable cells detected.")

    contour_result = cell_contour_extraction()
    contour_cells = contour_result.get("cells", [])
    say(f"[INFO] Extracted contours for {len(contour_cells)} usable cells.")

    if not contour_cells:
        say("[ERROR] No usable cell contours extracted.")
        raise RuntimeError("No usable cell contours extracted.")

    contour_by_id = {}
    for contour_cell in contour_cells:
        contour_by_id[contour_cell.get("cell_id")] = contour_cell

    selected_cell = None
    selected_distance_sq = None
    for cell in detected_cells:
        cell_id = cell.get("cell_id")
        if cell_id not in contour_by_id:
            continue
        x = cell.get("x")
        y = cell.get("y")
        if x is None or y is None:
            continue
        distance_sq = (x * x) + (y * y)
        if selected_cell is None or distance_sq < selected_distance_sq:
            selected_cell = cell
            selected_distance_sq = distance_sq

    if selected_cell is None:
        say("[ERROR] No detected cell has a matching extracted contour.")
        raise RuntimeError("No detected cell has a matching extracted contour.")

    selected_id = selected_cell["cell_id"]
    selected_contour = contour_by_id[selected_id]
    points = selected_contour.get("points", [])

    if len(points) < 3:
        say("[ERROR] Selected cell contour has fewer than three boundary points.")
        raise RuntimeError("Selected cell contour is invalid.")

    say(
        f"[INFO] Selected cell {selected_id} at "
        f"({selected_cell['x']}, {selected_cell['y']}) microns."
    )

    spacing = 2.0
    tolerance = 1e-9

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    def point_on_segment(px, py, ax, ay, bx, by):
        cross = ((px - ax) * (by - ay)) - ((py - ay) * (bx - ax))
        if abs(cross) > tolerance:
            return False
        dot = ((px - ax) * (px - bx)) + ((py - ay) * (py - by))
        return dot <= tolerance

    def point_inside_or_on_polygon(px, py, polygon):
        inside = False
        count = len(polygon)
        j = count - 1
        for i in range(count):
            xi, yi = polygon[i][0], polygon[i][1]
            xj, yj = polygon[j][0], polygon[j][1]

            if point_on_segment(px, py, xi, yi, xj, yj):
                return True

            intersects = ((yi > py) != (yj > py))
            if intersects:
                x_intersect = xi + ((py - yi) * (xj - xi) / (yj - yi))
                if px < x_intersect:
                    inside = not inside
            j = i
        return inside

    targets = []
    seen = set()

    def add_target(x, y):
        ix = int(round(x))
        iy = int(round(y))
        key = (ix, iy)
        if key not in seen:
            seen.add(key)
            targets.append(key)

    for point in points:
        add_target(point[0], point[1])

    y = min_y
    while y <= max_y + tolerance:
        row_targets = []
        x = min_x
        while x <= max_x + tolerance:
            if point_inside_or_on_polygon(x, y, points):
                row_targets.append((x, y))
            x += spacing

        if row_targets:
            if len(targets) % 2 == 0:
                for tx, ty in row_targets:
                    add_target(tx, ty)
            else:
                for tx, ty in reversed(row_targets):
                    add_target(tx, ty)

        y += spacing

    if not targets:
        say("[ERROR] Raster trajectory contains no bleaching targets.")
        raise RuntimeError("Raster trajectory is empty.")

    say(
        f"[ACTION] Photobleaching selected cell {selected_id} with "
        f"{len(targets)} raster targets at 2.0 micron spacing."
    )

    for x, y in targets:
        laser_position(x, y)

    say(f"[INFO] Completed targeted bleaching for cell {selected_id}.")

finally:
    laser_off()
    say("[ACTION] Stopped active FRAP workflow state; session remains available.")
```
