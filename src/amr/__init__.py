"""0-D velocity-space AMR machinery: tree node, sample fitting, and initial refinement."""
from .group import VelocityGroup, fit_maxent_weights, initial_refine

__all__ = ["VelocityGroup", "fit_maxent_weights", "initial_refine"]
