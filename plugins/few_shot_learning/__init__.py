"""Few-Shot Learning Plugin for FiftyOne Labs."""

from .panel import FewShotLearningPanel


def register(p):
    p.register(FewShotLearningPanel)
