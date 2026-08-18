# FRAP Tool Interaction Record: Sequential Bleaching of 36 Points on a Circle (Uninstalled State)

## User Request
> Imaging task: Use the FRAP tool to sequentially photobleach 36 evenly spaced points along a circle centered at the field-of-view origin (0, 0) with a radius of 50 micrometers.

## Planner Output
- **Status:** `unsupported`
- **Reason:** The current system modules (Microscope Operation Platform, Image Analysis Platform, Cell Segmentation Platform, mp285_controller) do not support FRAP (Fluorescence Recovery After Photobleaching) operations or specific photobleaching point pattern generation. The Microscope Operation Platform supports standard image acquisition and stage movement, but lacks commands for controlling a photobleaching laser or executing FRAP protocols. The mp285_controller is for mechanical manipulation and fluid handling, not optical photobleaching.
