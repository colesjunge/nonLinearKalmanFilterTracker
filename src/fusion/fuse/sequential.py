import numpy as np
from tracking.models.measurement import MeasurementModel

def fuse_sequential(filt, updates: list[tuple[MeasurementModel, np.ndarray]]) -> None:
    """
    Sequentially fuses multiple measurements into the filter.

    In: filt; the filter object (e.g., KalmanFilter, ExtendedKalmanFilter, etc.)
        updates; a list of tuples, where each tuple contains a MeasurementModel and the corresponding measurement vector (z).
    Out: None; the filter is updated in place.
    """

    for model, z in updates:
        filt.measurement = model
        filt.update(z)

    