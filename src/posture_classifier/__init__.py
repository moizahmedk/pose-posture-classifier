"""Standing / sitting / lying classification from 2D human pose."""

from .posture import Posture, PostureResult, PostureRules, classify_posture

__all__ = ["Posture", "PostureResult", "PostureRules", "classify_posture"]
__version__ = "0.1.0"
