"""Compatibility wrapper for the canonical user-tool prompt generator.

The maintained implementation now lives in ``utils.tool_generation``.
This module remains only to preserve older imports such as::

    from tool_generation import ToolProcessingPipeline
"""

from utils.tool_generation import *  # noqa: F401,F403
