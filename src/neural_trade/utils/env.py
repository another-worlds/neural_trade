"""Environment fingerprint for reproducibility (plan section C6)."""
from __future__ import annotations

import hashlib
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

REPO = Path(__file__).resolve().parents[3]


def git_sha(repo: Optional[Path] = None) -> str:
    """Short HEAD sha, with '-dirty' when tracked files have uncommitted changes."""
    repo = repo or REPO
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=repo,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def fingerprint(include_devices: bool = True) -> Dict[str, object]:
    """Versions, CUDA build, devices, git state and a HASHED hostname (never the raw name)."""
    import numpy
    import pandas
    import sklearn

    info: Dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "git": git_sha(),
        "host_sha1": hashlib.sha1(socket.gethostname().encode()).hexdigest()[:12],
    }
    try:
        import tensorflow as tf

        build = tf.sysconfig.get_build_info()
        info.update({"tensorflow": tf.__version__, "cuda_build": build.get("cuda_version"),
                     "cudnn_build": build.get("cudnn_version")})
        if include_devices:
            info["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        info["tensorflow"] = None
    return info
