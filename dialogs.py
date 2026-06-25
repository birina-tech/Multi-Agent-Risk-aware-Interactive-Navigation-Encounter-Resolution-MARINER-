from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QFormLayout, QDialogButtonBox, QGroupBox,
                             QCheckBox, QSlider, QSpinBox, QComboBox, QPushButton)
from PyQt5.QtCore import Qt
from llm_controller import LLMCoordinator

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
        llm_status = "🤖 LLM" if ship.llm_controlled else "👤 Manual"
        info_label = QLabel(
            f"<b>{ship.name}</b> [{llm_status}]<br>"
            f"Course: {ship.get_heading_deg():.1f}° | Speed: {ship.u:.2f} m/s<br>"
            f"Position: ({ship.x:.0f}, {ship.y:.0f})"
        )
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        llm_group = QGroupBox(" LLM Control")
        llm_layout = QHBoxLayout()
        self.llm_checkbox = QCheckBox("Enable LLM")
        self.llm_checkbox.setChecked(ship.llm_controlled)
        self.llm_checkbox.stateChanged.connect(self.on_llm_toggled)
        llm_layout.addWidget(self.llm_checkbox)
        llm_group.setLayout(llm_layout)
        layout.addWidget(llm_group)
        rudder_group = QGroupBox(" Rudder")
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
        rpm_group = QGroupBox("⚙️ Engine RPM")
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