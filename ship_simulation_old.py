import sys
import os
import json
import csv
from datetime import datetime
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QGroupBox, QStatusBar, QMessageBox, QInputDialog,
                             QDialog, QSlider, QSpinBox, QFormLayout, QDialogButtonBox,
                             QCheckBox, QComboBox, QTextBrowser, QMenu, QAction)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QCursor
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
import warnings
warnings.filterwarnings("ignore")

from collision_analyzer import launch_collision_analysis, CollisionAnalyzer
from llm_controller import LLMCoordinator


class LLMWorker(QThread):
    """Отдельный поток для выполнения LLM-запросов"""
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, coordinator, ships_data, collision_data):
        super().__init__()
        self.coordinator = coordinator
        self.ships_data = ships_data
        self.collision_data = collision_data
        self.is_running = True
    
    def run(self):
        try:
            commands = self.coordinator.get_coordinated_commands(
                self.ships_data, 
                self.collision_data
            )
            if self.is_running:
                self.result_ready.emit(commands)
        except Exception as e:
            if self.is_running:
                self.error_occurred.emit(str(e))
    
    def stop(self):
        self.is_running = False


class Ship:
    _counter = 0

    def __init__(self, x, y, psi_deg, speed_ms, name=None):
        Ship._counter += 1
        self.id = Ship._counter
        self.name = name or f"Ship_{self.id}"
        self.x = float(x)
        self.y = float(y)
        self.psi = np.deg2rad(float(psi_deg))
        self.u = float(speed_ms)
        self.r = 0.0
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

    def get_heading_deg(self):
        return np.mod(np.rad2deg(self.psi), 360)

    def distance_to(self, px, py):
        return np.sqrt((self.x - px) ** 2 + (self.y - py) ** 2)

    def get_pentagon_vertices(self):
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

    def apply_llm_command(self, rudder_deg, rpm_percent):
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

    def get_llm_status_text(self):
        if not self.llm_controlled:
            return ""
        if self.llm_decision:
            return (f"\nLLM: R={self.llm_decision.get('rudder_deg', 0):.0f}° "
                    f"RPM={self.llm_decision.get('rpm_percent', 50):.0f}%")
        return "\nLLM: ожидание..."

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


class AddShipDialog(QDialog):
    def __init__(self, x, y, parent=None, default_course=0, default_speed=5, default_name=None):
        super().__init__(parent)
        self.setWindowTitle("Add Vessel")
        self.setFixedSize(350, 200)
        self.x = x
        self.y = y
        layout = QVBoxLayout()
        info_label = QLabel(f"<b>Position:</b> ({x:.0f}, {y:.0f}) m")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        form_layout = QFormLayout()
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("Ship (auto)")
        if default_name:
            self.txt_name.setText(default_name)
        form_layout.addRow("Name:", self.txt_name)
        self.txt_course = QLineEdit(str(default_course))
        self.txt_course.setPlaceholderText("0° = North")
        form_layout.addRow("Course (°):", self.txt_course)
        self.txt_speed = QLineEdit(str(default_speed))
        self.txt_speed.setPlaceholderText("m/s")
        form_layout.addRow("Speed (m/s):", self.txt_speed)
        layout.addLayout(form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.txt_course.setFocus()

    def get_values(self):
        name = self.txt_name.text().strip()
        try:
            course = float(self.txt_course.text())
            speed = float(self.txt_speed.text())
            return name if name else None, course, speed
        except ValueError:
            return None, None, None


class ControlDialog(QDialog):
    def __init__(self, ship, parent=None):
        super().__init__(parent)
        self.ship = ship
        self.setWindowTitle(f"Control: {ship.name}")
        self.setFixedSize(400, 350)
        layout = QVBoxLayout()
        llm_status = "🤖 LLM" if ship.llm_controlled else " Manual"
        info_label = QLabel(
            f"<b>{ship.name}</b> [{llm_status}]<br>"
            f"Course: {ship.get_heading_deg():.1f}° | Speed: {ship.u:.2f} m/s<br>"
            f"Position: ({ship.x:.0f}, {ship.y:.0f})"
        )
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        llm_group = QGroupBox("🤖 LLM Control")
        llm_layout = QHBoxLayout()
        self.llm_checkbox = QCheckBox("Enable LLM")
        self.llm_checkbox.setChecked(ship.llm_controlled)
        self.llm_checkbox.stateChanged.connect(self.on_llm_toggled)
        llm_layout.addWidget(self.llm_checkbox)
        llm_group.setLayout(llm_layout)
        layout.addWidget(llm_group)
        rudder_group = QGroupBox("🧭 Rudder")
        rudder_layout = QHBoxLayout()
        self.rudder_slider = QSlider(Qt.Horizontal)
        self.rudder_slider.setRange(-35, 35)
        self.rudder_slider.setValue(int(ship.rudder_cmd))
        self.rudder_slider.setTickPosition(QSlider.TicksBelow)
        self.rudder_slider.setTickInterval(5)
        self.rudder_spin = QSpinBox()
        self.rudder_spin.setRange(-35, 35)
        self.rudder_spin.setValue(int(ship.rudder_cmd))
        self.rudder_spin.setSuffix("°")
        self.rudder_label = QLabel(f"{int(ship.rudder_cmd)}°")
        self.rudder_label.setMinimumWidth(50)
        self.rudder_label.setAlignment(Qt.AlignCenter)
        rudder_layout.addWidget(QLabel("Port"))
        rudder_layout.addWidget(self.rudder_slider, stretch=1)
        rudder_layout.addWidget(self.rudder_label)
        rudder_layout.addWidget(self.rudder_spin)
        rudder_layout.addWidget(QLabel("Starboard"))
        rudder_group.setLayout(rudder_layout)
        layout.addWidget(rudder_group)
        rpm_group = QGroupBox("️ Engine RPM")
        rpm_layout = QHBoxLayout()
        self.rpm_slider = QSlider(Qt.Horizontal)
        self.rpm_slider.setRange(0, 100)
        self.rpm_slider.setValue(int(ship.rpm_cmd))
        self.rpm_slider.setTickPosition(QSlider.TicksBelow)
        self.rpm_slider.setTickInterval(10)
        self.rpm_spin = QSpinBox()
        self.rpm_spin.setRange(0, 100)
        self.rpm_spin.setValue(int(ship.rpm_cmd))
        self.rpm_spin.setSuffix("%")
        self.rpm_label = QLabel(f"{int(ship.rpm_cmd)}%")
        self.rpm_label.setMinimumWidth(50)
        self.rpm_label.setAlignment(Qt.AlignCenter)
        rpm_layout.addWidget(QLabel("0%"))
        rpm_layout.addWidget(self.rpm_slider, stretch=1)
        rpm_layout.addWidget(self.rpm_label)
        rpm_layout.addWidget(self.rpm_spin)
        rpm_layout.addWidget(QLabel("100%"))
        rpm_group.setLayout(rpm_layout)
        layout.addWidget(rpm_group)
        self.rudder_slider.valueChanged.connect(self.on_rudder_slider)
        self.rudder_spin.valueChanged.connect(self.on_rudder_spin)
        self.rpm_slider.valueChanged.connect(self.on_rpm_slider)
        self.rpm_spin.valueChanged.connect(self.on_rpm_spin)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)
        self.setLayout(layout)

    def on_llm_toggled(self, state):
        self.ship.toggle_llm_control(state == Qt.Checked)

    def on_rudder_slider(self, value):
        self.ship.rudder_cmd = float(value)
        self.rudder_spin.blockSignals(True)
        self.rudder_spin.setValue(value)
        self.rudder_spin.blockSignals(False)
        self.rudder_label.setText(f"{value}°")
        if value < 0:
            self.rudder_label.setStyleSheet("color: blue; font-weight: bold;")
        elif value > 0:
            self.rudder_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.rudder_label.setStyleSheet("color: black;")

    def on_rudder_spin(self, value):
        self.ship.rudder_cmd = float(value)
        self.rudder_slider.blockSignals(True)
        self.rudder_slider.setValue(value)
        self.rudder_slider.blockSignals(False)
        self.rudder_label.setText(f"{value}°")

    def on_rpm_slider(self, value):
        self.ship.rpm_cmd = float(value)
        self.rpm_spin.blockSignals(True)
        self.rpm_spin.setValue(value)
        self.rpm_spin.blockSignals(False)
        self.rpm_label.setText(f"{value}%")

    def on_rpm_spin(self, value):
        self.ship.rpm_cmd = float(value)
        self.rpm_slider.blockSignals(True)
        self.rpm_slider.setValue(value)
        self.rpm_slider.blockSignals(False)
        self.rpm_label.setText(f"{value}%")


class LLMSettingsDialog(QDialog):
    def __init__(self, current_provider, api_keys, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ LLM Settings")
        self.setFixedSize(500, 400)
        self.current_provider = current_provider
        self.api_keys = api_keys
        self.result_provider = current_provider
        self.result_api_keys = dict(api_keys)
        layout = QVBoxLayout()
        provider_group = QGroupBox("LLM Provider")
        provider_layout = QVBoxLayout()
        self.provider_combo = QComboBox()
        for key, config in LLMCoordinator.PROVIDERS.items():
            self.provider_combo.addItem(config['name'], key)
        provider_keys = list(LLMCoordinator.PROVIDERS.keys())
        if current_provider in provider_keys:
            self.provider_combo.setCurrentIndex(provider_keys.index(current_provider))
        self.provider_combo.currentIndexChanged.connect(self.on_provider_changed)
        provider_layout.addWidget(self.provider_combo)
        provider_group.setLayout(provider_layout)
        layout.addWidget(provider_group)
        self.key_group = QGroupBox("API Key")
        key_layout = QVBoxLayout()
        self.key_label = QLabel("Enter API key for the selected provider:")
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Paste API key...")
        self.key_input.setEchoMode(QLineEdit.Password)
        self.show_key_cb = QCheckBox("Show key")
        self.show_key_cb.toggled.connect(
            lambda checked: self.key_input.setEchoMode(
                QLineEdit.Normal if checked else QLineEdit.Password
            )
        )
        key_layout.addWidget(self.key_label)
        key_layout.addWidget(self.key_input)
        key_layout.addWidget(self.show_key_cb)
        self.key_group.setLayout(key_layout)
        layout.addWidget(self.key_group)
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: gray; font-size: 10px; padding: 5px;")
        layout.addWidget(self.info_label)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.on_provider_changed(self.provider_combo.currentIndex())

    def on_provider_changed(self, index):
        provider_keys = list(LLMCoordinator.PROVIDERS.keys())
        if index < 0 or index >= len(provider_keys):
            return
        provider = provider_keys[index]
        config = LLMCoordinator.PROVIDERS[provider]
        self.key_group.setVisible(config['needs_key'])
        self.key_input.setText(self.result_api_keys.get(provider, ''))
        info = f"<b>{config['name']}</b><br>"
        info += f"Default model: {config['default_model']}<br>"
        info += f"URL: {config['url']}<br>"
        if config['needs_key']:
            info += "API key required"
        else:
            info += "No API key required"
        self.info_label.setText(info)

    def get_results(self):
        provider_keys = list(LLMCoordinator.PROVIDERS.keys())
        idx = self.provider_combo.currentIndex()
        if 0 <= idx < len(provider_keys):
            self.result_provider = provider_keys[idx]
            if LLMCoordinator.PROVIDERS[self.result_provider]['needs_key']:
                self.result_api_keys[self.result_provider] = self.key_input.text().strip()
        return self.result_provider, self.result_api_keys


class ShipCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(15.6, 10), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.view_center_x = 0.0
        self.view_center_y = 0.0
        self.view_scale = 5000.0
        self.min_scale = 200.0
        self.max_scale = 20000.0
        self.zoom_factor = 1.2
        self.is_panning = False
        self.pan_start_pos = None
        self.pan_start_center = None
        self.vector_length_minutes = 12.0
        self.track_length_meters = 5000.0
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
        self.ax.set_xlim(-self.view_scale, self.view_scale)
        self.ax.set_ylim(-self.view_scale, self.view_scale)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.5)
        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('scroll_event', self.on_scroll)
        self.ships = []
        self.on_click_callback = None
        self.on_ship_click_callback = None
        self.on_mouse_move_callback = None

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self.is_panning = True
            self.pan_start_pos = (event.x, event.y)
            self.pan_start_center = (self.view_center_x, self.view_center_y)
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button == 3:
            x, y = event.xdata, event.ydata
            clicked_ship = None
            click_radius = max(60, self.view_scale * 0.02)
            for ship in self.ships:
                if ship.distance_to(x, y) < click_radius:
                    clicked_ship = ship
                    break
            if clicked_ship:
                if self.on_ship_click_callback:
                    self.on_ship_click_callback(clicked_ship)
            else:
                if self.on_click_callback:
                    self.on_click_callback(x, y)

    def on_motion(self, event):
        if self.is_panning and event.inaxes == self.ax:
            dx_pix = event.x - self.pan_start_pos[0]
            dy_pix = event.y - self.pan_start_pos[1]
            bbox = self.ax.get_window_extent()
            if bbox.width == 0 or bbox.height == 0:
                return
            data_width = 2 * self.view_scale
            pixels_per_data_x = bbox.width / data_width
            pixels_per_data_y = bbox.height / data_width
            dx_data = dx_pix / pixels_per_data_x
            dy_data = -dy_pix / pixels_per_data_y
            self.view_center_x = self.pan_start_center[0] - dx_data
            self.view_center_y = self.pan_start_center[1] + dy_data
            self.ax.set_xlim(self.view_center_x - self.view_scale,
                             self.view_center_x + self.view_scale)
            self.ax.set_ylim(self.view_center_y - self.view_scale,
                             self.view_center_y + self.view_scale)
            self.draw()
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            if self.on_mouse_move_callback:
                self.on_mouse_move_callback(event.xdata, event.ydata)

    def on_release(self, event):
        if event.button == 1 and self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.OpenHandCursor)

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 'up':
            self.view_scale /= self.zoom_factor
        elif event.button == 'down':
            self.view_scale *= self.zoom_factor
        self.view_scale = np.clip(self.view_scale, self.min_scale, self.max_scale)
        self.ax.set_xlim(self.view_center_x - self.view_scale,
                         self.view_center_x + self.view_scale)
        self.ax.set_ylim(self.view_center_y - self.view_scale,
                         self.view_center_y + self.view_scale)
        self.draw()

    def draw_velocity_vector(self, ship):
        if ship.u < 0.1:
            return
        vector_length = ship.u * self.vector_length_minutes * 60
        vector_end_x = ship.x + vector_length * np.sin(ship.psi)
        vector_end_y = ship.y + vector_length * np.cos(ship.psi)
        self.ax.annotate('', xy=(vector_end_x, vector_end_y), xytext=(ship.x, ship.y),
                         arrowprops=dict(arrowstyle='->', color='blue',
                                         lw=2.0, mutation_scale=15), zorder=6)

    def update_plot(self, ships, running, simulation_time):
        self.ships = ships
        self.ax.clear()
        self.ax.set_xlim(self.view_center_x - self.view_scale,
                         self.view_center_x + self.view_scale)
        self.ax.set_ylim(self.view_center_y - self.view_scale,
                         self.view_center_y + self.view_scale)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.5)
        for ship in ships:
            track_x = []
            track_y = []
            for hx, hy in zip(ship.history_x, ship.history_y):
                dist = np.sqrt((hx - ship.x) ** 2 + (hy - ship.y) ** 2)
                if dist <= self.track_length_meters:
                    track_x.append(hx)
                    track_y.append(hy)
            if len(track_x) > 1:
                self.ax.plot(track_x, track_y, color=ship.color,
                             alpha=0.4, linewidth=1.5, linestyle='--', zorder=2)
            vertices = ship.get_pentagon_vertices()
            pentagon = Polygon(vertices, closed=True, facecolor='black',
                               edgecolor='black', linewidth=1.5, alpha=0.95, zorder=5)
            self.ax.add_patch(pentagon)
            if ship.llm_controlled:
                rect = Rectangle((ship.x - 60, ship.y - 60), 120, 120,
                                 fill=False, edgecolor='gold', linewidth=3, zorder=4)
                self.ax.add_patch(rect)
            self.draw_velocity_vector(ship)
            font_size = max(7, min(10, 1000 / self.view_scale * 10))
            llm_text = ship.get_llm_status_text()
            self.ax.text(ship.x + 80, ship.y + 80,
                         f"{ship.name}\n{ship.get_heading_deg():.0f}°\n"
                         f"{ship.u:.1f} m/s\n"
                         f"R:{ship.rudder_cmd:.0f}° RPM:{ship.rpm_cmd:.0f}%{llm_text}",
                         fontsize=font_size, ha='left', va='bottom',
                         bbox=dict(boxstyle='round', facecolor='white',
                                   alpha=0.8, edgecolor='gray'))
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MASS Sim')
        self.resize(1400, 900)
        self.ships = []
        self.running = False
        self.dt = 1.0
        self.simulation_time = 0.0
        self.analysis_window = None
        self.llm_coordinator = None
        self.current_provider = 'ollama'
        self.api_keys = {}
        self.llm_status_text = "Not tested"
        self.llm_update_interval = 3.0
        self.last_llm_update = 0.0
        self.recording = False
        self.rec_session_path = None
        self.log_files = {}
        self.tasks_dir = os.path.join(os.getcwd(), "Tasks")
        self.llm_worker = None
        self.llm_pending = False
        
        # === НОВОЕ: Режим перемещения судна ===
        self.move_mode = False
        self.move_ship = None
        
        self.init_menu()
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)

    def init_menu(self):
        menu_bar = self.menuBar()
        view_menu = menu_bar.addMenu("View")
        self.action_vector_len = view_menu.addAction("Vector length (minutes)")
        self.action_vector_len.triggered.connect(self.set_vector_length)
        self.action_track_len = view_menu.addAction("Track length (meters)")
        self.action_track_len.triggered.connect(self.set_track_length)
        view_menu.addSeparator()
        self.action_sim_speed = view_menu.addAction("Simulation speed...")
        self.action_sim_speed.triggered.connect(self.set_simulation_speed)
        view_menu.addSeparator()
        self.action_reset_view = view_menu.addAction("Reset view settings")
        self.action_reset_view.triggered.connect(self.reset_view_settings)
        llm_menu = menu_bar.addMenu("LLM")
        self.action_llm_settings = llm_menu.addAction("️ LLM Settings...")
        self.action_llm_settings.triggered.connect(self.open_llm_settings)
        self.action_llm_test = llm_menu.addAction("🔍 Test connection")
        self.action_llm_test.triggered.connect(self.test_llm_connection)
        llm_menu.addSeparator()
        self.action_llm_interval = llm_menu.addAction(
            f"⏱ Query interval: {self.llm_update_interval:.1f}s")
        self.action_llm_interval.triggered.connect(self.set_llm_interval)
        llm_menu.addSeparator()
        self.action_llm_status = llm_menu.addAction(f"📊 Status: {self.llm_status_text}")
        self.action_llm_status.setEnabled(False)
        llm_menu.addSeparator()
        self.action_llm_toggle = llm_menu.addAction("🤖 Enable LLM for all vessels")
        self.action_llm_toggle.triggered.connect(self.toggle_llm_for_all)
        tasks_menu = menu_bar.addMenu("Tasks")
        self.action_save_task = tasks_menu.addAction("💾 Save Task")
        self.action_save_task.triggered.connect(self.save_task)
        self.action_load_task = tasks_menu.addAction("📂 Load Task")
        self.action_load_task.triggered.connect(self.load_task)
        help_menu = menu_bar.addMenu("Help")
        self.action_help = help_menu.addAction("📘 User guide")
        self.action_help.triggered.connect(self.show_help)
        self.action_api_help = help_menu.addAction("🔑 Where to get API keys?")
        self.action_api_help.triggered.connect(self.show_api_help)
        help_menu.addSeparator()
        self.action_about = help_menu.addAction("️ About")
        self.action_about.triggered.connect(self.show_about)

    def set_llm_interval(self):
        val, ok = QInputDialog.getDouble(
            self, "LLM Query Interval",
            "Interval between LLM queries (seconds):",
            self.llm_update_interval, 1.0, 60.0, 1)
        if ok:
            self.llm_update_interval = val
            self.action_llm_interval.setText(f"⏱ Query interval: {val:.1f}s")
            self.statusBar().showMessage(f"LLM query interval set to {val:.1f}s")

    def set_simulation_speed(self):
        current = self.dt
        val, ok = QInputDialog.getDouble(
            self, "Simulation Speed",
            "Simulation speed multiplier (dt per step, seconds):",
            current, 0.1, 10.0, 1)
        if ok:
            self.dt = val
            self.statusBar().showMessage(f"Simulation speed set: dt={val:.2f}s")

    def open_llm_settings(self):
        dialog = LLMSettingsDialog(self.current_provider, self.api_keys, self)
        if dialog.exec_() == QDialog.Accepted:
            new_provider, new_keys = dialog.get_results()
            self.current_provider = new_provider
            self.api_keys = new_keys
            self.llm_coordinator = None
            self.llm_status_text = "Settings changed"
            self.update_llm_status_display()
            config = LLMCoordinator.PROVIDERS[self.current_provider]
            self.statusBar().showMessage(f"LLM settings updated: {config['name']}")

    def test_llm_connection(self):
        coordinator = self.get_or_create_coordinator()
        self.statusBar().showMessage(
            f"Testing connection to {coordinator.PROVIDERS[self.current_provider]['name']}...")
        QApplication.processEvents()
        success, message = coordinator.test_connection()
        if success:
            self.llm_status_text = f"OK: {coordinator.PROVIDERS[self.current_provider]['name']}"
            QMessageBox.information(
                self, "Connection successful",
                f"OK: {message}\n"
                f"Provider: {coordinator.PROVIDERS[self.current_provider]['name']}\n"
                f"Model: {coordinator.model}")
        else:
            self.llm_status_text = f"Error: {message[:50]}"
            QMessageBox.warning(
                self, "Connection error",
                f"Error: {message}\n"
                f"Provider: {coordinator.PROVIDERS[self.current_provider]['name']}")
        self.update_llm_status_display()
        self.statusBar().showMessage(message)

    def update_llm_status_display(self):
        self.action_llm_status.setText(f"📊 Status: {self.llm_status_text}")

    def get_or_create_coordinator(self):
        api_key = self.api_keys.get(self.current_provider)
        if (self.llm_coordinator is None or
                self.llm_coordinator.provider != self.current_provider):
            self.llm_coordinator = LLMCoordinator(
                provider=self.current_provider, api_key=api_key)
        else:
            self.llm_coordinator.api_key = api_key
        return self.llm_coordinator

    def set_vector_length(self):
        current_val = self.canvas.vector_length_minutes
        val, ok = QInputDialog.getDouble(
            self, "Vector length",
            "Enter vector length (minutes):",
            current_val, 0.1, 120.0, 1)
        if ok:
            self.canvas.vector_length_minutes = val
            self.statusBar().showMessage(f"Vector length set: {val} min")

    def set_track_length(self):
        current_val = self.canvas.track_length_meters
        val, ok = QInputDialog.getInt(
            self, "Track length",
            "Enter track length (meters):",
            int(current_val), 100, 50000, 100)
        if ok:
            self.canvas.track_length_meters = float(val)
            self.statusBar().showMessage(f"Track length set: {val} m")

    def reset_view_settings(self):
        self.canvas.vector_length_minutes = 12.0
        self.canvas.track_length_meters = 5000.0
        self.canvas.view_center_x = 0.0
        self.canvas.view_center_y = 0.0
        self.canvas.view_scale = 5000.0
        self.canvas.ax.set_xlim(-5000, 5000)
        self.canvas.ax.set_ylim(-5000, 5000)
        self.canvas.draw()
        self.statusBar().showMessage("View settings reset")

    def show_api_help(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Where to get API keys?")
        dialog.setFixedSize(650, 520)
        layout = QVBoxLayout()
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setHtml("""
            <h3>Getting API keys for LLM</h3>
            <p><b>1. Ollama (Local, Free):</b><br>
            No key needed! Download from <a href="https://ollama.com">ollama.com</a> and run it.</p>
            <p><b>2. Groq (Free, Very fast):</b><br>
            Go to <a href="https://console.groq.com">console.groq.com</a>, sign in via GitHub/Google,
            create an API Key in the "API Keys" section. Key starts with <code>gsk_</code>.</p>
            <p><b>3. OpenAI (Paid, ~$0.01/request):</b><br>
            Keys at <a href="https://platform.openai.com/api-keys">platform.openai.com</a>.</p>
            <p><b>4. DeepSeek (Very cheap):</b><br>
            Keys at <a href="https://platform.deepseek.com">platform.deepseek.com</a>.</p>
            <p><b>5. Anthropic Claude (Paid):</b><br>
            Keys at <a href="https://console.anthropic.com">console.anthropic.com</a>.</p>
            <p><i>Copy the key and paste it via menu LLM → LLM Settings.</i></p>
        """)
        text_browser.setStyleSheet("font-size: 12px;")
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.close)
        layout.addWidget(text_browser)
        layout.addWidget(btn_close)
        dialog.setLayout(layout)
        dialog.exec_()

    def show_help(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("User guide")
        msg.setIcon(QMessageBox.Information)
        msg.setTextFormat(Qt.RichText)
        text = """
        <h2>MASS Sim - Autonomous Vessel Collision Avoidance Simulator</h2>
        <h3>Scene control:</h3>
        <ul>
        <li><b>Pan:</b> Hold <b>Left Mouse Button (LMB)</b> and drag.</li>
        <li><b>Zoom:</b> Use the <b>Mouse Wheel</b>.</li>
        </ul>
        <h3>Vessel operations:</h3>
        <ul>
        <li><b>Add vessel:</b> <b>Right-click (RMB)</b> on empty space. Set course and speed.</li>
        <li><b>Control vessel:</b> <b>RMB on vessel icon</b>. Opens context menu with Delete/Move/Options.</li>
        <li><b>Move vessel:</b> RMB on vessel → Move → Click LMB on new position → Set course and speed.</li>
        </ul>
        <h3>Tasks:</h3>
        <ul>
        <li>Menu <b>Tasks → Save Task</b> — save current vessel configuration to a file.</li>
        <li>Menu <b>Tasks → Load Task</b> — load vessel configuration from a file.</li>
        <li>Task files are stored in the <code>Tasks</code> folder.</li>
        </ul>
        <h3>LLM (AI Coordinator) setup:</h3>
        <ul>
        <li>Menu <b>LLM → LLM Settings...</b> — choose provider and API key.</li>
        <li>Menu <b>LLM → Test connection</b> — test link to provider.</li>
        <li>Menu <b>LLM → Query interval</b> — set frequency of LLM requests.</li>
        <li>Menu <b>LLM → Enable LLM for all vessels</b> — quick AI control toggle.</li>
        </ul>
        <h3>Running simulation:</h3>
        <ul>
        <li>Add at least 2 vessels or load a task.</li>
        <li>Enable LLM (via menu or RMB on vessel → Options).</li>
        <li>Press <b>▶ Start</b>. AI will analyze COLREGs and issue commands.</li>
        <li>Menu <b>View → Simulation speed...</b> — change simulation speed.</li>
        </ul>
        <h3>Data recording:</h3>
        <ul>
        <li>Press <b>Start Rec</b> to begin recording. A session folder is created under <code>Logs/yy_mm_dd_hh_mm_ss/</code>.</li>
        <li>Each vessel's state is logged to a CSV file named after the vessel.</li>
        <li>Press <b>Stop Rec</b> to stop recording and close files.</li>
        </ul>
        <h3>COLREG Analysis:</h3>
        <ul>
        <li>Button <b>ColReg analysis</b> opens a table with CPA, TCPA, risk index and rules for all pairs.</li>
        </ul>
        <p><i>Gold frame around a vessel = under AI control.</i></p>
        """
        msg.setText(text)
        msg.exec_()

    def show_about(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("About")
        msg.setIcon(QMessageBox.Information)
        msg.setTextFormat(Qt.RichText)
        text = """
        <h2>MASS Sim</h2>
        <p>Version 1.0</p>
        <p>Autonomous vessel collision avoidance simulator with AI coordinator
        compliant with COLREGs (International Regulations for Preventing Collisions at Sea).</p>
        <p><b>Features:</b></p>
        <ul>
        <li>Nomoto vessel dynamics model</li>
        <li>CPA/TCPA and risk index calculation</li>
        <li>Automatic COLREGs rule detection (13, 14, 15, 17)</li>
        <li>LLM coordinator for maneuver coordination</li>
        <li>Support for local (Ollama) and cloud (OpenAI, Groq, Anthropic, DeepSeek) models</li>
        <li>Simulation data logging to CSV</li>
        <li>Task save/load functionality</li>
        </ul>
        <p><i>Based on architecture from the paper
        "Large Language Model-based Decision-making for COLREGs-compliant autonomous surface vehicles"</i></p>
        """
        msg.setText(text)
        msg.exec_()

    def save_task(self):
        if len(self.ships) == 0:
            QMessageBox.warning(self, "No vessels", "No vessels to save.")
            return
        filename, ok = QInputDialog.getText(
            self, "Save Task", "Enter task filename (without extension):")
        if not ok or not filename.strip():
            return
        filename = filename.strip()
        if not filename.endswith('.json'):
            filename += '.json'
        os.makedirs(self.tasks_dir, exist_ok=True)
        filepath = os.path.join(self.tasks_dir, filename)
        task_data = {'vessels': []}
        for ship in self.ships:
            vessel_data = {
                'name': ship.name,
                'x': ship.x,
                'y': ship.y,
                'course_deg': ship.get_heading_deg(),
                'speed_ms': ship.u,
                'rudder_deg': ship.rudder_cmd,
                'rpm_percent': ship.rpm_cmd
            }
            task_data['vessels'].append(vessel_data)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(task_data, f, indent=2, ensure_ascii=False)
            self.statusBar().showMessage(f"Task saved: {filepath}")
            QMessageBox.information(self, "Task Saved", f"Task saved successfully:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save task:\n{e}")

    def load_task(self):
        if not os.path.exists(self.tasks_dir):
            QMessageBox.warning(self, "No Tasks", "No Tasks folder found.")
            return
        task_files = [f for f in os.listdir(self.tasks_dir) if f.endswith('.json')]
        if not task_files:
            QMessageBox.warning(self, "No Tasks", "No task files found in Tasks folder.")
            return
        filename, ok = QInputDialog.getItem(
            self, "Load Task", "Select task file to load:",
            task_files, 0, False)
        if not ok or not filename:
            return
        filepath = os.path.join(self.tasks_dir, filename)
        if self.running:
            self.stop_simulation()
        self.clear_ships()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            for vessel_data in task_data.get('vessels', []):
                ship = Ship(
                    x=vessel_data['x'],
                    y=vessel_data['y'],
                    psi_deg=vessel_data['course_deg'],
                    speed_ms=vessel_data['speed_ms'],
                    name=vessel_data.get('name')
                )
                ship.rudder_cmd = vessel_data.get('rudder_deg', 0.0)
                ship.rpm_cmd = vessel_data.get('rpm_percent', 50.0)
                self.ships.append(ship)
            self.statusBar().showMessage(f"Task loaded: {filepath}")
            QMessageBox.information(
                self, "Task Loaded",
                f"Task loaded successfully:\n{filepath}\n\nVessels: {len(self.ships)}")
            self.canvas.update_plot(self.ships, self.running, self.simulation_time)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load task:\n{e}")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.canvas = ShipCanvas()
        self.canvas.on_click_callback = self.on_empty_field_click
        self.canvas.on_ship_click_callback = self.on_ship_click
        self.canvas.on_mouse_move_callback = self.on_mouse_move
        main_layout.addWidget(self.canvas, stretch=1)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        self.btn_start = QPushButton("▶ Start")
        self.btn_start.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        self.btn_start.setFixedHeight(40)
        self.btn_start.clicked.connect(self.start_simulation)
        self.btn_stop = QPushButton(" Stop")
        self.btn_stop.setStyleSheet("background-color: lightyellow; font-weight: bold;")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.clicked.connect(self.stop_simulation)
        self.btn_clear = QPushButton("🗑 Clear All")
        self.btn_clear.setStyleSheet("background-color: lightcoral; font-weight: bold;")
        self.btn_clear.setFixedHeight(40)
        self.btn_clear.clicked.connect(self.clear_ships)
        self.btn_rec = QPushButton("🔴 Start Rec")
        self.btn_rec.setStyleSheet("background-color: lightgray; font-weight: bold;")
        self.btn_rec.setFixedHeight(40)
        self.btn_rec.setEnabled(False)
        self.btn_rec.clicked.connect(self.toggle_recording)
        self.btn_analysis = QPushButton("📊 ColReg analysis")
        self.btn_analysis.setStyleSheet("background-color: lightblue; font-weight: bold;")
        self.btn_analysis.setFixedHeight(40)
        self.btn_analysis.clicked.connect(self.open_collision_analysis)
        right_layout.addWidget(self.btn_start)
        right_layout.addWidget(self.btn_stop)
        right_layout.addWidget(self.btn_clear)
        right_layout.addWidget(self.btn_rec)
        right_layout.addWidget(self.btn_analysis)
        right_layout.addStretch()
        main_layout.addWidget(right_panel)
        self.statusBar().showMessage("Ready. Use the menus at the top.")

    def on_mouse_move(self, x, y):
        # Если в режиме перемещения - показать подсказку
        if self.move_mode:
            self.statusBar().showMessage(
                f"📍 MOVE MODE: Click to place {self.move_ship.name} at X={x:.0f} m, Y={y:.0f} m (or press Esc to cancel)")
        else:
            self.statusBar().showMessage(f"Cursor: X={x:.0f} m, Y={y:.0f} m")

    def update_rec_button_state(self):
        can_record = len(self.ships) > 0 or self.running
        self.btn_rec.setEnabled(can_record)

    def toggle_recording(self):
        if not self.recording:
            if len(self.ships) == 0 and not self.running:
                QMessageBox.warning(
                    self, "Recording unavailable",
                    "Cannot start recording: no vessels present and simulation is not running.")
                return
            now = datetime.now()
            session_name = now.strftime("%y_%m_%d_%H_%M_%S")
            logs_dir = os.path.join(os.getcwd(), "Logs")
            os.makedirs(logs_dir, exist_ok=True)
            self.rec_session_path = os.path.join(logs_dir, session_name)
            os.makedirs(self.rec_session_path, exist_ok=True)
            for ship in self.ships:
                self.open_log_file(ship)
            self.recording = True
            self.btn_rec.setText("⏹ Stop Rec")
            self.btn_rec.setStyleSheet("background-color: #ff9999; font-weight: bold;")
            self.statusBar().showMessage(f"Recording started: {self.rec_session_path}")
        else:
            self.close_all_logs()
            self.recording = False
            self.rec_session_path = None
            self.btn_rec.setText("🔴 Start Rec")
            self.btn_rec.setStyleSheet("background-color: lightgray; font-weight: bold;")
            self.statusBar().showMessage("Recording stopped")

    def open_log_file(self, ship):
        if ship.name in self.log_files:
            return
        if self.rec_session_path is None:
            return
        filename = os.path.join(self.rec_session_path, f"{ship.name}.csv")
        try:
            f = open(filename, 'w', newline='')
            writer = csv.writer(f)
            writer.writerow(['time_s', 'x_m', 'y_m',
                             'course_deg', 'speed_ms', 'rudder_deg', 'rpm_percent'])
            self.log_files[ship.name] = (f, writer)
        except Exception as e:
            print(f"Failed to open log file for {ship.name}: {e}")

    def close_log_file(self, ship_name):
        """Закрыть лог-файл для конкретного судна"""
        if ship_name in self.log_files:
            f, _ = self.log_files[ship_name]
            try:
                f.close()
            except Exception:
                pass
            del self.log_files[ship_name]

    def log_ship_state(self, ship):
        if not self.recording:
            return
        if ship.name not in self.log_files:
            self.open_log_file(ship)
        if ship.name in self.log_files:
            f, writer = self.log_files[ship.name]
            try:
                writer.writerow(ship.get_log_row(self.simulation_time))
                f.flush()
            except Exception as e:
                print(f"Log write error for {ship.name}: {e}")

    def close_all_logs(self):
        for name, (f, _) in self.log_files.items():
            try:
                f.close()
            except Exception:
                pass
        self.log_files.clear()

    def on_empty_field_click(self, x, y):
        """Обработчик клика ЛКМ по пустому месту"""
        # === НОВОЕ: Если в режиме перемещения - переместить судно ===
        if self.move_mode and self.move_ship:
            self.complete_move_ship(x, y)
            return
        
        # Обычное добавление нового судна
        dialog = AddShipDialog(x, y, self)
        if dialog.exec_() == QDialog.Accepted:
            name, course, speed = dialog.get_values()
            if course is not None and speed is not None:
                ship = Ship(x, y, course, speed, name)
                self.ships.append(ship)
                if self.recording:
                    self.open_log_file(ship)
                self.statusBar().showMessage(
                    f"Added: {ship.name} | ({x:.0f}, {y:.0f}) | "
                    f"Course: {course}° | Speed: {speed} m/s")
                self.update_rec_button_state()
                self.canvas.update_plot(self.ships, self.running, self.simulation_time)
            else:
                QMessageBox.warning(self, "Error", "Invalid course or speed values")

    def on_ship_click(self, ship):
        """Показать контекстное меню при ПКМ на судно"""
        # Если в режиме перемещения - отменить его
        if self.move_mode:
            self.cancel_move_ship()
        
        menu = QMenu(self)
        
        action_delete = QAction("🗑 Delete", self)
        action_delete.triggered.connect(lambda: self.delete_ship(ship))
        menu.addAction(action_delete)
        
        action_move = QAction("📍 Move", self)
        action_move.triggered.connect(lambda: self.start_move_ship(ship))
        menu.addAction(action_move)
        
        action_options = QAction("⚙️ Options", self)
        action_options.triggered.connect(lambda: self.open_ship_options(ship))
        menu.addAction(action_options)
        
        menu.exec_(QCursor.pos())

    def start_move_ship(self, ship):
        """Начать режим перемещения судна"""
        self.move_mode = True
        self.move_ship = ship
        self.statusBar().showMessage(
            f"📍 MOVE MODE: Click LMB on the scene to place {ship.name} (or press Esc to cancel)")
        # Изменить курсор для визуальной индикации
        self.canvas.setCursor(Qt.CrossCursor)

    def complete_move_ship(self, x, y):
        """Завершить перемещение судна на новые координаты"""
        if not self.move_mode or not self.move_ship:
            return
        
        ship = self.move_ship
        current_course = ship.get_heading_deg()
        current_speed = ship.u
        current_name = ship.name
        
        # Закрыть лог-файл старого судна
        self.close_log_file(ship.name)
        
        # Удалить старое судно из списка
        if ship in self.ships:
            self.ships.remove(ship)
        
        # Сбросить режим перемещения
        self.move_mode = False
        self.move_ship = None
        self.canvas.setCursor(Qt.OpenHandCursor)
        
        # Открыть диалог с новыми координатами
        dialog = AddShipDialog(
            x, y, self,
            default_course=current_course,
            default_speed=current_speed,
            default_name=current_name
        )
        
        if dialog.exec_() == QDialog.Accepted:
            name, course, speed = dialog.get_values()
            if course is not None and speed is not None:
                new_ship = Ship(x, y, course, speed, name)
                self.ships.append(new_ship)
                if self.recording:
                    self.open_log_file(new_ship)
                self.statusBar().showMessage(
                    f"Moved: {new_ship.name} | ({x:.0f}, {y:.0f}) | "
                    f"Course: {course}° | Speed: {speed} m/s")
        else:
            # Если отменили - восстановить судно на старом месте
            self.ships.append(ship)
            self.statusBar().showMessage(f"Move cancelled: {ship.name} remains at original position")
        
        self.update_rec_button_state()
        self.canvas.update_plot(self.ships, self.running, self.simulation_time)

    def cancel_move_ship(self):
        """Отменить режим перемещения"""
        if self.move_mode:
            self.move_mode = False
            self.move_ship = None
            self.canvas.setCursor(Qt.OpenHandCursor)
            self.statusBar().showMessage("Move cancelled")

    def delete_ship(self, ship):
        """Удалить судно"""
        # Если в режиме перемещения - отменить его
        if self.move_mode:
            self.cancel_move_ship()
        
        # Закрыть лог-файл если запись активна
        self.close_log_file(ship.name)
        
        # Удалить из списка
        if ship in self.ships:
            self.ships.remove(ship)
        
        self.statusBar().showMessage(f"Deleted: {ship.name}")
        self.update_rec_button_state()
        self.canvas.update_plot(self.ships, self.running, self.simulation_time)

    def open_ship_options(self, ship):
        """Открыть окно управления судном"""
        # Если в режиме перемещения - отменить его
        if self.move_mode:
            self.cancel_move_ship()
        
        dialog = ControlDialog(ship, self)
        dialog.exec_()
        llm_status = "LLM enabled" if ship.llm_controlled else "Manual control"
        self.statusBar().showMessage(
            f"{ship.name}: {llm_status} | Rudder={ship.rudder_cmd}°, RPM={ship.rpm_cmd}%")
        self.canvas.update_plot(self.ships, self.running, self.simulation_time)

    def toggle_llm_for_all(self):
        llm_enabled = not any(s.llm_controlled for s in self.ships)
        if llm_enabled:
            coordinator = self.get_or_create_coordinator()
            success, message = coordinator.test_connection()
            if not success:
                QMessageBox.warning(
                    self, "LLM connection error",
                    f"Failed to connect to {coordinator.PROVIDERS[self.current_provider]['name']}:\n"
                    f"{message}\n\nCheck settings in menu LLM → LLM Settings.")
                return
            self.llm_status_text = f"OK: {coordinator.PROVIDERS[self.current_provider]['name']}"
        else:
            self.llm_status_text = "LLM disabled"
        for ship in self.ships:
            ship.toggle_llm_control(llm_enabled)
        if llm_enabled:
            self.statusBar().showMessage(
                f"LLM ({coordinator.PROVIDERS[self.current_provider]['name']}) "
                f"enabled for all vessels")
            self.action_llm_toggle.setText("⏹ Disable LLM for all vessels")
        else:
            self.statusBar().showMessage("LLM control disabled")
            self.action_llm_toggle.setText("🤖 Enable LLM for all vessels")
        self.update_llm_status_display()
        self.canvas.update_plot(self.ships, self.running, self.simulation_time)

    def open_collision_analysis(self):
        if self.analysis_window is None:
            self.analysis_window = launch_collision_analysis(self.ships)
        else:
            self.analysis_window.show()
            self.analysis_window.raise_()
            self.analysis_window.activateWindow()

    def start_simulation(self):
        if not self.running:
            self.running = True
            self.timer.start(50)
            self.statusBar().showMessage("Simulation started")
            self.update_rec_button_state()

    def stop_simulation(self):
        self.running = False
        self.timer.stop()
        if self.llm_worker and self.llm_worker.isRunning():
            self.llm_worker.stop()
            self.llm_worker.wait(2000)
            self.llm_pending = False
        if self.recording:
            self.toggle_recording()
        self.statusBar().showMessage("Simulation stopped")
        self.update_rec_button_state()

    def clear_ships(self):
        if self.recording:
            self.toggle_recording()
        if self.move_mode:
            self.cancel_move_ship()
        self.ships.clear()
        Ship._counter = 0
        self.simulation_time = 0.0
        self.last_llm_update = 0.0
        self.statusBar().showMessage("All vessels cleared")
        self.update_rec_button_state()
        self.canvas.update_plot(self.ships, self.running, self.simulation_time)

    def collect_collision_data(self):
        analyzer = CollisionAnalyzer()
        collision_data = []
        for i in range(len(self.ships)):
            for j in range(i + 1, len(self.ships)):
                ship1 = self.ships[i]
                ship2 = self.ships[j]
                cpa_data = analyzer.calculate_cpa_tcpa(ship1, ship2)
                risk_index = analyzer.calculate_risk_index(
                    cpa_data['DCPA'], cpa_data['TCPA'], cpa_data['dist'])
                if risk_index > 0.3:
                    rule_num, role1, role2 = analyzer.determine_colreg_rule(ship1, ship2)
                    action1 = analyzer.get_required_action(rule_num, role1)
                    action2 = analyzer.get_required_action(rule_num, role2)
                    collision_data.append({
                        'pair': f"{ship1.name} ↔ {ship2.name}",
                        'dist': cpa_data['dist'],
                        'cpa': cpa_data['DCPA'],
                        'tcpa': cpa_data['TCPA'],
                        'risk': risk_index,
                        'rule': rule_num,
                        'actions': f"{ship1.name}: {action1}, {ship2.name}: {action2}"
                    })
        return collision_data

    def on_llm_result(self, commands):
        self.llm_pending = False
        if self.llm_coordinator:
            self.llm_coordinator.apply_commands(self.ships, commands)
        self.statusBar().showMessage(f"LLM commands applied at t={self.simulation_time:.1f}s")

    def on_llm_error(self, error_msg):
        self.llm_pending = False
        print(f"LLM Worker Error: {error_msg}")
        self.statusBar().showMessage(f"LLM Error: {error_msg[:50]}")

    def start_llm_worker(self, collision_data):
        if self.llm_pending:
            return
        coordinator = self.get_or_create_coordinator()
        self.llm_worker = LLMWorker(coordinator, self.ships, collision_data)
        self.llm_worker.result_ready.connect(self.on_llm_result)
        self.llm_worker.error_occurred.connect(self.on_llm_error)
        self.llm_pending = True
        self.llm_worker.start()
        self.statusBar().showMessage(f"LLM query sent at t={self.simulation_time:.1f}s (async)")

    def simulation_step(self):
        if self.running:
            llm_ships = [s for s in self.ships if s.llm_controlled]
            if (llm_ships and 
                    (self.simulation_time - self.last_llm_update >= self.llm_update_interval) and
                    not self.llm_pending):
                collision_data = self.collect_collision_data()
                if collision_data:
                    self.start_llm_worker(collision_data)
                self.last_llm_update = self.simulation_time
            for ship in self.ships:
                ship.update(self.dt)
                self.log_ship_state(ship)
            self.simulation_time += self.dt
            self.canvas.update_plot(self.ships, self.running, self.simulation_time)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())