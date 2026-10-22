"""
SmartBus TN - Crowd Status Engine
Stage 1: Standalone AI People Counter

This module is the SINGLE SOURCE OF TRUTH for calculating crowd status
based on current occupancy and bus capacity.
"""


def get_status(occupancy: int, capacity: int) -> tuple[str, float]:
    """
    Calculate crowd status and occupancy percentage.

    Thresholds:
      - Below 50%          : GREEN
      - 50% to below 80%   : YELLOW
      - 80% to 100%        : RED
      - Above capacity     : OVERFLOW (occupancy > capacity)

    Args:
        occupancy: Current passenger count on board.
        capacity: Maximum rated passenger capacity.

    Returns:
        (status, percent): e.g. ("GREEN", 49.0)
    """
    if capacity <= 0:
        percent = 0.0
    else:
        percent = (occupancy / capacity) * 100.0

    if occupancy > capacity:
        status = "OVERFLOW"
    elif percent < 50.0:
        status = "GREEN"
    elif percent < 80.0:
        status = "YELLOW"
    else:
        status = "RED"

    return status, round(percent, 1)
