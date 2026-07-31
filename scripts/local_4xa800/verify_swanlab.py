import os

from verl.utils.tracking import Tracking


mode = os.environ.get("SWANLAB_MODE", "online")
if mode == "online" and not os.environ.get("SWANLAB_API_KEY"):
    raise RuntimeError("SwanLab online verification requires SWANLAB_API_KEY.")

tracker = Tracking(
    project_name="search-r1",
    experiment_name="integration-check",
    default_backend=["console", "swanlab"],
    config={"verification": True},
)
tracker.log({"verification/value": 1.0}, step=1)
tracker.finish()
print(f"SwanLab {mode} integration: OK")
