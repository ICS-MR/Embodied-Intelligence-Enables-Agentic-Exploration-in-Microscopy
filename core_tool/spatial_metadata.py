from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import tifffile

logger = logging.getLogger(__name__)


def load_ome_spatial_metadata(
    file_path: str | Path,
    *,
    require_stage_positions: bool = False,
) -> dict[str, float | bool]:
    resolved = Path(file_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Source image file does not exist: {resolved}")

    center_x_um = 0.0
    center_y_um = 0.0
    center_z_um = 0.0
    pixel_size_x_um = 1.0
    pixel_size_y_um = 1.0
    stage_positions_present = False

    with tifffile.TiffFile(str(resolved)) as tif:
        ome_xml = tif.ome_metadata or ""

    if ome_xml:
        root = ET.fromstring(ome_xml)
        namespace = {}
        if root.tag.startswith("{") and "}" in root.tag:
            namespace["ome"] = root.tag.split("}", 1)[0][1:]
            pixels = root.find(".//ome:Pixels", namespace)
            plane = root.find(".//ome:Plane", namespace)
        else:
            pixels = root.find(".//Pixels")
            plane = root.find(".//Plane")

        if pixels is not None:
            pixel_size_x_um = float(pixels.attrib.get("PhysicalSizeX") or 1.0)
            pixel_size_y_um = float(pixels.attrib.get("PhysicalSizeY") or 1.0)
        if plane is not None:
            has_x = "PositionX" in plane.attrib
            has_y = "PositionY" in plane.attrib
            stage_positions_present = bool(has_x and has_y)
            if has_x:
                center_x_um = float(plane.attrib["PositionX"])
            if has_y:
                center_y_um = float(plane.attrib["PositionY"])
            if "PositionZ" in plane.attrib:
                center_z_um = float(plane.attrib["PositionZ"])

    if require_stage_positions and not stage_positions_present:
        raise ValueError(
            "OME stage position metadata is missing Plane PositionX/PositionY; "
            f"cannot convert image-space detections into stage coordinates for {resolved}"
        )

    if not stage_positions_present:
        logger.warning("OME stage position metadata is missing for %s", resolved)

    return {
        "center_x_um": center_x_um,
        "center_y_um": center_y_um,
        "center_z_um": center_z_um,
        "pixel_size_x_um": pixel_size_x_um,
        "pixel_size_y_um": pixel_size_y_um,
        "stage_positions_present": stage_positions_present,
    }
