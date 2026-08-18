from __future__ import annotations

from .common import *


class Frap(BaseTool):
    """Simulation implementation of the public FRAP API."""

    _active_instance: Frap | None = None

    planning_hint = (
        "Use for FRAP workflows that start and stop FRAP, detect cells, extract contours, "
        "and photobleach selected field-centered micron coordinates. Spatial patterns can be "
        "executed as sampled coordinate trajectories derived from fitted contours or generated "
        "paths. Bleaching intensity is controlled by sampling point spacing."
    )
    execution_hint = (
        "Call laser_on before cell_detection, cell_contour_extraction, or laser_position; "
        "call laser_off after the full bleaching sequence is complete. Treat laser_position "
        "x and y as microns relative to the field center. Use detected cell centers, fitted "
        "contour points, or sampled trajectories as laser_position targets. Adjust relative "
        "bleaching dose by changing point spacing: increase point spacing to lower the dose; "
        "decrease point spacing to raise the dose."
    )

    def __init__(self, storage_manager=None, output_dir: str = "./output") -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir
        self._laser_enabled = False
        type(self)._active_instance = self

    @tool_func
    def laser_on(self) -> None:
        """
        Turn on the FRAP operation switch.

        This method must be called before laser_position(), cell_detection(),
        or cell_contour_extraction(). It starts the FRAP operation.

        """
        self._laser_enabled = True

    @tool_func
    def laser_off(self) -> None:
        """
        Turn off the FRAP operation switch.

        Call this method after completing the laser_position() bleaching
        sequence. This stops FRAP operation but does not release the session.
        """
        self._laser_enabled = False

    @tool_func
    @staticmethod
    def laser_position(x: int, y: int) -> None:
        """
        Position the laser at the specified coordinates and perform one
        bleaching operation at that location.

        FRAP must be turned on with laser_on() before calling this method.
        The method may be called repeatedly to bleach multiple positions.

        Args:
            x: X-axis position in microns relative to the center of the current field of view.
            y: Y-axis position in microns relative to the center of the current field of view.
        """
        del x, y
        instance = Frap._require_active_instance()
        if not instance._laser_enabled:
            raise RuntimeError("laser_position requires FRAP to be started first.")

    @tool_func
    @staticmethod
    def cell_detection() -> dict:
        """
        Detect all usable cells in the current field of view.

        FRAP must be turned on with laser_on() before calling this method.

        Returns:
            Dictionary containing a ``cells`` list. Each item contains ``cell_id``
            plus ``x`` and ``y`` coordinates in microns relative to the field center.
            The list is empty when no usable cells are detected.
        """
        instance = Frap._require_active_instance()
        instance._require_laser_enabled("cell_detection")
        return {"cells": [{"cell_id": 1, "x": 0.0, "y": 0.0}]}

    @tool_func
    @staticmethod
    def cell_contour_extraction() -> dict:
        """
        Extract all usable cell contours from the current field of view.

        FRAP must be turned on with laser_on() before calling this method.

        Returns:
            Dictionary containing a ``cells`` list. Each item contains ``cell_id``
            and fitted ellipse ``points`` represented as ``[x, y]`` pairs in
            field-centered microns. The list is empty when no usable contours
            are extracted.
        """
        instance = Frap._require_active_instance()
        instance._require_laser_enabled("cell_contour_extraction")
        return {
            "cells": [
                {
                    "cell_id": 1,
                    "points": [
                        [-10.0, -10.0],
                        [10.0, -10.0],
                        [10.0, 10.0],
                        [-10.0, 10.0],
                    ],
                }
            ]
        }

    def release_session(self) -> None:
        """Reset simulated FRAP session state."""
        self._laser_enabled = False

    @classmethod
    def _require_active_instance(cls) -> Frap:
        instance = cls._active_instance
        if instance is None:
            raise RuntimeError("Frap must be instantiated before using this static method.")
        return instance

    def _require_laser_enabled(self, operation_name: str) -> None:
        if not self._laser_enabled:
            raise RuntimeError(f"{operation_name} requires FRAP to be started first.")
