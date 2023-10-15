import numpy as np
from tools.snowfall.simulation import my_augment
from tools.snowfall.sampling import dart_throwing, compute_occupancy, snowfall_rate_to_rainfall_rate

class SnowGenerator:
    snowflakes = {}

    def __init__(self, seed=42):
        self.snowfall_rate = 0.5         # 0.5-2.5 mm/h
        self.terminal_velocity = 0.2     # 0.2-2 m/s
        self.snow_density = 0.2          # 0.01-0.2 g/cm^3
        self.snowflake_diameter = 0.003  # m
        self.R_0 = 50. # radius in meters, where snow particles are located
        self.distribution = 'sekhon' #'sekhon', 'gunn'
        self.rng = np.random.default_rng(seed)
        self.update_occupancy_ratio()
        self.update_precipitation_rate()
    
    def update_occupancy_ratio(self):
        self.occupancy_ratio = compute_occupancy(
            self.snowfall_rate, 
            self.terminal_velocity, 
            self.snow_density)

    def update_precipitation_rate(self):
        self.precipitation_rate = snowfall_rate_to_rainfall_rate(
            self.snowfall_rate, 
            self.terminal_velocity, 
            self.snow_density, 
            self.snowflake_diameter)

    def set_distribution(self, distribution="sekhon"):
        """Set distribution

        Args:
            distribution (str, optional): options: ["sekhon", "gunn"]. Defaults to "sekhon".
        """
        self.distribution = distribution

    def set_max_sampling_radius(self, max_sampling_radius: float):
        self.R_0 = max_sampling_radius

    def set_snowfall_rate(self, snowfall_rate: float):
        self.snowfall_rate = snowfall_rate
        self.update_occupancy_ratio()
        self.update_precipitation_rate()

    def set_terminal_velocity(self, terminal_velocity: float):
        self.terminal_velocity = terminal_velocity
        self.update_occupancy_ratio()
        self.update_precipitation_rate()
    
    def set_snow_density(self, snow_density: float):
        self.snow_density = snow_density
        self.update_occupancy_ratio()
        self.update_precipitation_rate()

    def set_snowflake_diameter(self, snowflake_diameter: float):
        self.snowflake_diameter = snowflake_diameter
        self.update_precipitation_rate()

    
    def generate_snowflakes(self, unique_channels_list: list):
        for c in unique_channels_list:
            self.snowflakes[c] = dart_throwing(self.occupancy_ratio, 
                                            self.precipitation_rate, 
                                            self.R_0, 
                                            self.rng, 
                                            self.distribution, 
                                            show_progessbar=False)

    def augment_cloud(self, cloud: np.ndarray, 
                      sensor_info: dict,
                      beam_divergence_rad: float = 0.003, 
                      noise_floor: float = 0.7,
                      shuffle=True, 
                      show_progressbar=True) -> np.ndarray:
        beam_divergence = float(np.degrees(beam_divergence_rad))  # (deg)
        stats, aug_cloud = my_augment(cloud, particles=self.snowflakes, 
                            sensor_info=sensor_info, 
                            beam_divergence=beam_divergence, 
                            shuffle=shuffle, 
                            show_progressbar=show_progressbar, 
                            noise_floor=noise_floor)
        return stats, aug_cloud
    