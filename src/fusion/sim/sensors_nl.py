import numpy as np
from tracking.sim.sensors import Sensor
from tracking.types import Detection
from fusion.models.radar import RadarRangeBearing
from fusion.sim.noise import contaminated_gaussian_noise


class RadarSensor(Sensor):
    """
    Nonlinear Radar Sensor that wraps RadarRangeBearing measurement model
    Generates per frame detections
    """

    def __init__(self, radar_model: RadarRangeBearing, p_detect: float, clutter_rate: float, range_max: float, bearing_range: tuple):
        
        self.radar_model = radar_model
        self.p_detect = p_detect
        self.clutter_rate = clutter_rate
        self.range_max = range_max
        self.bearing_range = bearing_range

        self.dim_z = 2

    def observe(self, true_states, t, rng) -> list[Detection]:
        """
        Adds noise to true positions and returns a list of Detection objects.
        Same as PositionSensor.observe(), but uses radar model to generate measurements.
        In: true_states shape (n_targets, dim_x), t; current time, rng; random number generator
        Out: list of Detection objects (both true detections and clutter)
        """

        # Get the number of targets
        n_targets = true_states.shape[0]

        # Determine which targets are detected
        detected_mask = rng.uniform(size=n_targets) < self.p_detect

        # Generate range/bearing measurements for detected targets
        measurements = []
        for i in range(n_targets):
            if detected_mask[i]:
                # Generate ideal range/bearing measurement
                measurement = self.radar_model.h(true_states[i])

                # Keep measurements within the sensor's measurement bounds
                if 0 <= measurement[0] <= self.range_max:
                    if self.bearing_range[0] <= measurement[1] <= self.bearing_range[1]:
                        # Add Contaminated Gaussian noise in range/bearing space
                        noise = contaminated_gaussian_noise(rng, self.radar_model.R())
                        measurement = measurement + noise

                        measurements.append(Detection(measurement, t, sensor_id="radar", meta={"target_id": i}))

        # Add clutter to measurements
        n_clutter = rng.poisson(self.clutter_rate)
        for _ in range(n_clutter):
            # Clutter sampled uniformly in range/bearing space
            clutter_measurement = np.array([
                rng.uniform(0, self.range_max),
                rng.uniform(self.bearing_range[0], self.bearing_range[1])
            ])

            measurements.append(Detection(clutter_measurement, t, sensor_id="clutter", meta={"clutter": True}))

        return measurements