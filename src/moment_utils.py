"""Backwards-compatible shim. New code should import from .physics.moments directly."""
from .physics.moments import calc_moment, invert, moment_eq, moments

__all__ = ["calc_moment", "invert", "moment_eq", "moments"]
