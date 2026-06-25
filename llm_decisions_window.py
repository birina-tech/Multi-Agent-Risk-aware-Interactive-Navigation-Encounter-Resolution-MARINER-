"""
llm_decisions_window.py
Окно отображения решений LLM по управлению судами
"""
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTableWidget, QTableWidgetItem,
                             QHeaderView, QScrollArea, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont  # ДОБАВЛЕН QFont


class LLMDecisionsWindow(QMainWindow):
    """Окно отображения решений LLM"""
    
    def __init__(self, ships_ref, llm_coordinator_ref=None):
        super().__init__()
        self.ships_ref = ships_ref
        self.llm_coordinator_ref = llm_coordinator_ref
        
        self.setWindowTitle("LLM Decisions Monitor")
        self.resize(900, 600)
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_table)
        self.timer.start(1000)  # Обновление каждую секунду
        
        self.update_table()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Заголовок
        title_label = QLabel("LLM Control Decisions & Reasoning")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; background-color: #e3f2fd;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Создаём scroll area для таблицы
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Таблица
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Vessel",
            "Status",
            "Rudder (°)",
            "RPM (%)",
            "LLM Reasoning"
        ])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        
        scroll.setWidget(self.table)
        layout.addWidget(scroll)
        
        # Кнопка закрытия
        btn_close = QPushButton("Close window")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)
    
    def update_table(self):
        """Обновить таблицу решений LLM"""
        ships = self.ships_ref() if callable(self.ships_ref) else self.ships_ref
        
        if len(ships) == 0:
            self.table.setRowCount(0)
            return
        
        self.table.setRowCount(0)
        
        row = 0
        for ship in ships:
            if not ship.llm_controlled:
                continue
            
            self.table.insertRow(row)
            
            # Имя судна
            name_item = QTableWidgetItem(ship.name)
            name_item.setFont(QFont("Arial", 10, QFont.Bold))
            self.table.setItem(row, 0, name_item)
            
            # Статус
            if ship.in_maneuver:
                status = "MANEUVERING"
                status_color = QColor(255, 200, 0)
            else:
                status = "ON COURSE"
                status_color = QColor(144, 238, 144)
            
            status_item = QTableWidgetItem(status)
            status_item.setBackground(status_color)
            self.table.setItem(row, 1, status_item)
            
            # Руль
            rudder_item = QTableWidgetItem(f"{ship.rudder_cmd:.1f}°")
            if abs(ship.rudder_cmd) > 15:
                rudder_item.setBackground(QColor(255, 150, 150))
            self.table.setItem(row, 2, rudder_item)
            
            # RPM
            rpm_item = QTableWidgetItem(f"{ship.rpm_cmd:.0f}%")
            if ship.rpm_cmd < 40 or ship.rpm_cmd > 60:
                rpm_item.setBackground(QColor(255, 200, 150))
            self.table.setItem(row, 3, rpm_item)
            
            # Обоснование LLM - ИСПРАВЛЕНО
            reasoning = ""
            if hasattr(ship, 'llm_decision') and ship.llm_decision:
                reasoning = ship.llm_decision.get('reasoning', '')
            
            if not reasoning and hasattr(ship, 'llm_reasoning'):
                reasoning = ship.llm_reasoning
            
            if not reasoning:
                # Генерируем обоснование на основе статуса
                if ship.in_maneuver:
                    if ship.rudder_cmd > 0:
                        reasoning = f"Turning starboard {ship.rudder_cmd:.0f}° to avoid collision"
                    elif ship.rudder_cmd < 0:
                        reasoning = f"Turning port {ship.rudder_cmd:.0f}° to avoid collision"
                    else:
                        reasoning = "Reducing speed for safety"
                else:
                    reasoning = "Maintaining course and speed - no collision risk"
            
            reasoning_item = QTableWidgetItem(reasoning)
            reasoning_item.setFlags(reasoning_item.flags() | Qt.ItemIsEditable)
            self.table.setItem(row, 4, reasoning_item)
            
            row += 1
        
        if row == 0:
            self.table.setRowCount(1)
            no_data_item = QTableWidgetItem("No vessels under LLM control")
            no_data_item.setForeground(QColor(128, 128, 128))
            self.table.setItem(0, 0, no_data_item)
            self.table.setSpan(0, 0, 1, 5)


def launch_llm_decisions_window(ships_ref, llm_coordinator_ref=None):
    """Запустить окно решений LLM"""
    window = LLMDecisionsWindow(ships_ref, llm_coordinator_ref)
    window.show()
    return window