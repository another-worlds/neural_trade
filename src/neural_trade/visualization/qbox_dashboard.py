"""HTML block of the physics (T-perp / QBOX) loss terms for the epoch dashboard (moved in B12)."""
from __future__ import annotations


def _qbox_dashboard_html(logs):
    """Return an HTML string for the T_⊥ / QBOX section of the epoch dashboard.

    Only renders if at least one QBOX component has a non-trivial value (> 1e-6),
    so the section stays hidden when all QBOX lambdas are 0.
    """
    components = [
        ('t_perp_loss',       'T⊥ calib',   'val_t_perp_loss'),
        ('casimir_loss',      'Casimir',     'val_casimir_loss'),
        ('vac_loss',          'Vac BW',      'val_vac_loss'),
        ('hd_loss',           'Hyper-Dec',   'val_hd_loss'),
        ('ife_loss',          'Info-Flow',   'val_ife_loss'),
        ('vac_overflow_loss', 'T⊥ Overflow', 'val_vac_overflow_loss'),
    ]
    rows = []
    for train_key, label, val_key in components:
        train_val = float(logs.get(train_key, 0.0))
        val_val   = float(logs.get(val_key,   0.0))
        if train_val > 1e-6 or val_val > 1e-6:
            rows.append(
                f'<span style="display: inline-block; width: 180px;">{label}:</span>'
                f' <span style="color: #CE93D8;">{train_val:.6f}</span>'
                f' <span style="color: #888; font-size: 11px;">val: {val_val:.6f}</span><br>'
            )
    if not rows:
        return ''
    inner = '\n                            '.join(rows)
    return f"""
                    <div style="margin-bottom: 15px;">
                        <div style="color: #CE93D8; font-weight: bold; margin-bottom: 8px;">⚛ T&#x22A5; / QBOX LOSSES</div>
                        <div style="margin-left: 15px;">
                            {inner}
                        </div>
                    </div>"""


def qbox_dashboard_html(data, config=None, **_):
    """Registry entry: ``data`` is a Keras logs dict; empty string when every term is ~0."""
    return _qbox_dashboard_html(data)
