"""Test-time PYTHONPATH so pytest can find the package without colcon."""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
for p in (_here, _here.parent / "humanoid_pose_estimator"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
