"""
Backward compatibility shim for losses module.

DEPRECATED: This module is maintained for backward compatibility.
New code should import from 'registries.losses' instead.

The losses registry has been refactored to use the BaseRegistry pattern
and moved to registries/losses.py. This file re-exports everything
to maintain compatibility with existing code.

Example (old):
    from losses import Losses
    loss_fn = Losses.get('focal_loss')

Example (new - recommended):
    from registries.losses import Losses
    loss_fn = Losses.get('focal_loss')
"""

# Import everything from the new location
from registries.losses import (
    Losses,
    focal_loss,
    dice_loss,
    combined_direction_loss,
    compute_dynamic_alpha,
    point_huber,
    local_trend_loss,
    extended_trend_loss,
    custom_loss,
)

# Re-export everything for backward compatibility
__all__ = [
    'Losses',
    'focal_loss',
    'dice_loss',
    'combined_direction_loss',
    'compute_dynamic_alpha',
    'point_huber',
    'local_trend_loss',
    'extended_trend_loss',
    'custom_loss',
]
