from __future__ import annotations

import os
from pathlib import Path


def default_g1_mjcf_xml_path() -> Path:
    """Return the preferred G1 MJCF path for Prior_Recon tools.

    Resolution order:
    1. ``PRIOR_RECON_G1_XML`` environment variable
    2. vendored Prior_Recon asset path
    """

    override = os.environ.get("PRIOR_RECON_G1_XML")
    if override:
        path = Path(override).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(
            f"PRIOR_RECON_G1_XML points to a missing file: {path}"
        )

    repo_root = Path(__file__).resolve().parents[3]
    path = (
        repo_root
        / "Prior_Recon"
        / "Masked_Flow"
        / "assets"
        / "g1_29dof.xml"
    )
    if path.exists():
        return path

    raise FileNotFoundError(
        "Could not locate g1_29dof.xml. Set PRIOR_RECON_G1_XML or place the asset under "
        "Prior_Recon/Masked_Flow/assets/g1_29dof.xml."
    )
