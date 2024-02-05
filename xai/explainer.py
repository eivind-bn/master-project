from typing import *

from .feed_forward import FeedForward

class Explainer:
    
    def __init__(self, feed_forward: FeedForward) -> None:
        self._feed_forward = feed_forward