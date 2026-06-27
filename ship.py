import numpy as np

class Ship:
    _counter = 0

    def __init__(self, x, y, psi_deg, speed_ms, name=None):
        Ship._counter += 1
        self.id = Ship._counter
        self.name = name or f"Ship_{self.id}" # Name of the ship for the GUI
        
        self.x = float(x) # Current 2D spatial coordinates of the vessel in meters
        self.y = float(y) # Current 2D spatial coordinates of the vessel in meters
        
        self.base_x = float(x) # Recording an initial coordinates for future return to base course
        self.base_y = float(y) # Recording an initial coordinates for future return to base course
        self.psi = np.deg2rad(float(psi_deg)) # Current heading (yaw angle) of the ship, converted from degrees to radians
        self.u = float(speed_ms) # Current forward speed of the ship in meters per second
        self.r = 0.0 # Current rate of turn (yaw rate) in radians per second
        self.T_psi = 30.0
        self.K_psi = 0.01
        self.T_v = 50.0
        self.K_v = 1.0
        self.rudder_cmd = 0.0
        self.rpm_cmd = 50.0
        self.rudder_max = 35.0
        self.rpm_max = 100.0
        self.history_x = [self.x]
        self.history_y = [self.y]
        self.max_history = 2000
        self.length = 110.0
        self.width = 20.0
        self.bow_length = 10.0
        self.hull_length = 100.0
        self.color = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'][(self.id - 1) % 8]
        self.llm_controlled = False
        self.llm_decision = None
        self.base_heading_deg = float(psi_deg)
        self.in_maneuver = False # True if ship's current heading (self.psi) deviates by more than 1 degrees from its base_heading_deg
        self.llm_reasoning = ""  # LLM desicion reasoning

    def update(self, dt):
        tau_c = np.clip(self.rudder_cmd * np.pi / 180 * (20 / 35), -20, 20)
        u_c = np.clip(self.rpm_cmd, 0, self.rpm_max) / 100.0
        r_dot = -(1 / self.T_psi) * self.r + (self.K_psi / self.T_psi) * tau_c
        u_dot = -(1 / self.T_v) * self.u + (self.K_v / self.T_v) * u_c * 5.0
        self.r = self.r + r_dot * dt
        self.u = max(0.0, self.u + u_dot * dt)
        x_dot = self.u * np.sin(self.psi)
        y_dot = self.u * np.cos(self.psi)
        psi_dot = self.r
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.psi = np.arctan2(np.sin(self.psi + psi_dot * dt), np.cos(self.psi + psi_dot * dt))
        self.history_x.append(self.x)
        self.history_y.append(self.y)
        if len(self.history_x) > self.max_history:
            self.history_x.pop(0)
            self.history_y.pop(0)

    def get_heading_deg(self): # Convert ship's heading from radians into a standard compass heading (in degrees)
        return np.mod(np.rad2deg(self.psi), 360)

    def distance_to(self, px, py): # calculates the exact straight-line distance between the ship's current position and any other specific point on the map
        return np.sqrt((self.x - px) ** 2 + (self.y - py) ** 2)

    def get_pentagon_vertices(self): # UI function
        half_width = self.width / 2
        bow_x = self.length / 2
        stern_x = -self.length / 2
        bow_shoulder_x = bow_x - self.bow_length
        local_vertices = np.array([
            [bow_x, 0], [bow_shoulder_x, half_width],
            [stern_x, half_width], [stern_x, -half_width],
            [bow_shoulder_x, -half_width]
        ])
        sin_psi = np.sin(self.psi)
        cos_psi = np.cos(self.psi)
        rotation = np.array([[sin_psi, cos_psi], [cos_psi, -sin_psi]])
        return np.array([rotation @ v + np.array([self.x, self.y]) for v in local_vertices])

    def toggle_llm_control(self, enable=True):
        self.llm_controlled = enable

    def apply_llm_command(self, rudder_deg, rpm_percent): # Converts LLM rudder commands into actionalble changes (with phisical boundaries)
        rudder_diff = rudder_deg - self.rudder_cmd
        if abs(rudder_diff) > 5:
            self.rudder_cmd += np.sign(rudder_diff) * 5
        else:
            self.rudder_cmd = rudder_deg
        rpm_diff = rpm_percent - self.rpm_cmd
        if abs(rpm_diff) > 10:
            self.rpm_cmd += np.sign(rpm_diff) * 10
        else:
            self.rpm_cmd = rpm_percent

    def get_llm_status_text(self): # UI function, generate  pop-up text attached to the ship
        if not self.llm_controlled:
            return ""
        if self.llm_decision:
            rudder = self.llm_decision.get('rudder_deg', 0)
            rpm = self.llm_decision.get('rpm_percent', 50)
            return f"\nLLM: R={rudder:.0f}° RPM={rpm:.0f}%"
        return "\nLLM: waiting..."

    def get_log_row(self, time_s):
        return [
            f"{time_s:.2f}",
            f"{self.x:.2f}",
            f"{self.y:.2f}",
            f"{self.get_heading_deg():.2f}",
            f"{self.u:.3f}",
            f"{self.rudder_cmd:.2f}",
            f"{self.rpm_cmd:.1f}"
        ]
    def set_base_heading(self, heading_deg):
        """Set a new base heading degree and position (e.g., after Load Task)"""
        self.base_x = self.x
        self.base_y = self.y
        self.base_heading_deg = float(heading_deg)