import numpy as np

class DopplerSimulator:
    SPEED_OF_SOUND = 343.0

    def compute_positions(self, t, src_params, obs_params):
        src_type, x0, y0, speed, direction = src_params
        obs_type, ox0, oy0, ospeed, odirection = obs_params

        if src_type == 'moving':
            theta_s = np.radians(direction)
            src_x = x0 + speed * np.cos(theta_s) * t
            src_y = y0 + speed * np.sin(theta_s) * t
        else:
            src_x, src_y = x0, y0

        if obs_type == 'moving':
            theta_o = np.radians(odirection)
            obs_x = ox0 + ospeed * np.cos(theta_o) * t
            obs_y = oy0 + ospeed * np.sin(theta_o) * t
        else:
            obs_x, obs_y = ox0, oy0

        return src_x, src_y, obs_x, obs_y

    def compute_perceived_frequency(self, f_emit, src_x, src_y, obs_x, obs_y,
                                    src_type, src_speed, src_dir,
                                    obs_type, obs_speed, obs_dir):
        dx = obs_x - src_x
        dy = obs_y - src_y
        distance = np.sqrt(dx**2 + dy**2)

        v_src_rad = 0.0
        v_obs_rad = 0.0

        if distance > 1e-6:
            ur_x = dx / distance
            ur_y = dy / distance
            if src_type == 'moving':
                v_sx = src_speed * np.cos(np.radians(src_dir))
                v_sy = src_speed * np.sin(np.radians(src_dir))
                v_src_rad = v_sx * ur_x + v_sy * ur_y
            if obs_type == 'moving':
                v_ox = obs_speed * np.cos(np.radians(obs_dir))
                v_oy = obs_speed * np.sin(np.radians(obs_dir))
                v_obs_rad = -(v_ox * ur_x + v_oy * ur_y)

        denominator = self.SPEED_OF_SOUND - v_src_rad
        if abs(denominator) < 1e-6:
            f_perceived = f_emit
        else:
            f_perceived = f_emit * (self.SPEED_OF_SOUND + v_obs_rad) / denominator

        return max(20, min(20000, round(f_perceived, 1)))