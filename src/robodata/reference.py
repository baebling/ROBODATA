"""Import calibrated independent marker trajectories without using prediction alignment."""

import csv
import hashlib
from pathlib import Path

import numpy as np
import yaml
from pydantic import Field, field_validator
from scipy.spatial.transform import Rotation

from .export import save_episode
from .geometry import validate_transform
from .pipeline import software_provenance
from .schema import Canonical, Contract, Episode, FrameRecord, Quality, Source
from .temporal import add_labels


class Calibration(Contract):
    episode_id: str
    task_name: str
    surface_frame_id: str
    instrument: str
    T_surface_tracking: list[list[float]]
    T_marker_tool: list[list[float]]
    time_offset_sec: float = 0
    horizon_sec: float = Field(default=0.2, gt=0)
    timestamp_tolerance_sec: float = Field(default=0.04, ge=0)

    @field_validator("T_surface_tracking", "T_marker_tool")
    @classmethod
    def matrix(cls, value):
        return validate_transform(value).tolist()


def import_reference(csv_path, calibration_path, output_path) -> Episode:
    path, calibration, out = Path(csv_path), Path(calibration_path), Path(output_path)
    if out.exists():
        raise FileExistsError(out)
    cfg = Calibration.model_validate(yaml.safe_load(calibration.read_text(encoding="utf-8")))
    frames = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = ["timestamp_sec", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"]
        if reader.fieldnames is None or not set(required) <= set(reader.fieldnames):
            raise ValueError("Reference CSV needs metric position and xyzw quaternion columns")
        for i, row in enumerate(reader):
            values = np.array([float(row[k]) for k in required])
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite reference measurement")
            if not np.isclose(np.linalg.norm(values[4:8]), 1, atol=0.01):
                raise ValueError("Reference quaternion must be normalized")
            G_M = np.eye(4)
            G_M[:3, :3] = Rotation.from_quat(values[4:8]).as_matrix()
            G_M[:3, 3] = values[1:4]
            ST = np.array(cfg.T_surface_tracking) @ G_M @ np.array(cfg.T_marker_tool)
            frames.append(
                FrameRecord(
                    frame_index=i,
                    timestamp_sec=float(values[0] + cfg.time_offset_sec),
                    canonical=Canonical(
                        T_surface_tool=ST.tolist(),
                        tip_position_surface_m=ST[:3, 3].tolist(),
                        tool_axis_surface=ST[:3, 2].tolist(),
                        standoff_distance_m=float(abs(ST[2, 3])),
                        signed_surface_distance_m=float(ST[2, 3]),
                        incidence_angle_rad=float(np.arccos(np.clip(-ST[2, 2], -1, 1))),
                    ),
                    quality=Quality(
                        valid=True, full_pose_valid=True, confidence_kind="instrument_reference"
                    ),
                )
            )
    provenance = software_provenance(calibration)
    provenance.update(
        {
            "instrument": cfg.instrument,
            "calibration": cfg.model_dump(),
            "measurement_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
    episode = Episode(
        episode_id=cfg.episode_id,
        task_name=cfg.task_name,
        surface_frame_id=cfg.surface_frame_id,
        source=Source(
            mode="external_pose", metric_scale_source="calibrated_external", data_kind="real"
        ),
        frames=frames,
        provenance=provenance,
    )
    add_labels(episode, cfg.horizon_sec, cfg.timestamp_tolerance_sec)
    save_episode(episode, out)
    return episode
