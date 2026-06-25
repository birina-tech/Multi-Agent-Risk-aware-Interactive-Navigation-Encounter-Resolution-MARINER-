"""
collision_analyzer.py
Анализ столкновений и определение правил МППСС
"""
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTableWidget, QTableWidgetItem,
                             QHeaderView, QGroupBox, QScrollArea)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor 
from colreg_rules import determine_colreg_situation


class CollisionAnalyzer:
    """Анализатор столкновений"""
    
    def calculate_cpa_tcpa(self, ship1, ship2):
        """
        Рассчитать CPA (Distance at Closest Point of Approach) и TCPA (Time to CPA)
        """
        dx = ship2.x - ship1.x
        dy = ship2.y - ship1.y
        dist = np.sqrt(dx**2 + dy**2)
        
        v1_x = ship1.u * np.sin(ship1.psi)
        v1_y = ship1.u * np.cos(ship1.psi)
        v2_x = ship2.u * np.sin(ship2.psi)
        v2_y = ship2.u * np.cos(ship2.psi)
        
        v_rel_x = v2_x - v1_x
        v_rel_y = v2_y - v1_y
        v_rel = np.sqrt(v_rel_x**2 + v_rel_y**2)
        
        if v_rel < 0.1:
            return {
                'dist': dist,
                'DCPA': dist,
                'TCPA': float('inf'),
                'v_rel': v_rel
            }
        
        v_rel_norm_x = v_rel_x / v_rel
        v_rel_norm_y = v_rel_y / v_rel
        
        proj = dx * v_rel_norm_x + dy * v_rel_norm_y
        tcpa = -proj / v_rel
        
        if tcpa < 0:
            tcpa = 0
        
        cpa_x = dx + v_rel_x * tcpa
        cpa_y = dy + v_rel_y * tcpa
        dcpa = np.sqrt(cpa_x**2 + cpa_y**2)
        
        return {
            'dist': dist,
            'DCPA': dcpa,
            'TCPA': tcpa,
            'v_rel': v_rel
        }
    
    def calculate_risk_index(self, dcpa, tcpa, dist):
        """
        Рассчитать индекс риска (0-1)
        """
        dcpa_risk = max(0, min(1, (1000 - dcpa) / 1000))
        tcpa_risk = max(0, min(1, (600 - tcpa) / 600))
        dist_risk = max(0, min(1, (5000 - dist) / 5000))
        
        risk_index = (dcpa_risk + tcpa_risk + dist_risk) / 3
        
        return risk_index
    
    def calculate_course_crossing(self, ship1, ship2):
        """
        Определить, какое судно пересекает курс другого по носу.
        
        Логика:
        1. Найти точку пересечения ЛИНИЙ КУРСОВ (геометрических прямых)
        2. Вычислить, какое судно быстрее достигнет этой точки
        3. То судно, которое придёт раньше, и пересекает курс другого по носу
        
        Returns:
            dict с результатами анализа
        """
        result = {
            'crossing_point': None,
            'crosses_1_by_2': False,
            'crosses_2_by_1': False,
            'time_1_to_cross': None,
            'time_2_to_cross': None,
            'time_gap': None,
            'crossing_type': None,
        }
        
        # Направления движения
        sin1, cos1 = np.sin(ship1.psi), np.cos(ship1.psi)
        sin2, cos2 = np.sin(ship2.psi), np.cos(ship2.psi)
        
        dx = ship2.x - ship1.x
        dy = ship2.y - ship1.y
        
        # Определитель системы
        D = sin2 * cos1 - cos2 * sin1  # = sin(ψ2 - ψ1)
        
        if abs(D) < 1e-6:
            # Курсы параллельны — пересечения нет
            return result
        
        # Параметры вдоль линий курсов (в метрах)
        s = (dy * sin2 - dx * cos2) / D   # для судна 1
        t = (dy * sin1 - dx * cos1) / D   # для судна 2
        
        # Точка пересечения (через судно 1)
        cross_x = ship1.x + s * sin1
        cross_y = ship1.y + s * cos1
        result['crossing_point'] = (cross_x, cross_y)
        
        # Проверяем, впереди ли точка для обоих судов
        point_ahead_of_ship1 = s > 0
        point_ahead_of_ship2 = t > 0
        
        if not (point_ahead_of_ship1 and point_ahead_of_ship2):
            # Точка позади хотя бы одного судна — пересечения по носу нет
            return result
        
        # Обе точки впереди — считаем времена
        if ship1.u < 0.1 or ship2.u < 0.1:
            return result
        
        time_1 = s / ship1.u  # секунды
        time_2 = t / ship2.u  # секунды
        
        result['time_1_to_cross'] = time_1
        result['time_2_to_cross'] = time_2
        result['time_gap'] = abs(time_1 - time_2)
        
        # УБРАНО условие проверки разницы во времени > 60 секунд
        # Теперь пересечение фиксируется всегда, если точка впереди обоих судов
        
        # Определяем, кто пересёк курс
        if time_1 < time_2:
            # Судно 1 придёт раньше → оно пересечёт курс судна 2 по носу
            result['crosses_2_by_1'] = True
            result['crossing_type'] = f'{ship1.name} crosses {ship2.name} ahead'
        else:
            # Судно 2 придёт раньше → оно пересечёт курс судна 1 по носу
            result['crosses_1_by_2'] = True
            result['crossing_type'] = f'{ship2.name} crosses {ship1.name} ahead'
        
        return result


class CollisionAnalysisWindow(QMainWindow):
    """Окно анализа столкновений"""
    
    def __init__(self, ships_ref):
        super().__init__()
        self.ships_ref = ships_ref
        self.analyzer = CollisionAnalyzer()
        
        self.setWindowTitle("ColRegs Analysis")
        self.resize(1400, 700)
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_table)
        self.timer.start(1000)
        
        self.update_table()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        title_label = QLabel("Interaction of each pair of vessels in the context of the COLREGs")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.table = QTableWidget()
        self.table.setColumnCount(12)  # 12 колонок (добавлена Crosses ahead)
        self.table.setHorizontalHeaderLabels([
            "Pair",
            "Distance (m)",
            "CPA (m)",
            "TCPA (s)",
            "Risk",
            "Rule №",
            "Situation",
            "Bearing 1→2",
            "Bearing 2→1",
            "Crosses ahead",  # НОВАЯ КОЛОНКА
            f"Action {self.ships_ref[0].name if len(self.ships_ref) > 0 else 'Ship1'}",
            f"Action {self.ships_ref[1].name if len(self.ships_ref) > 1 else 'Ship2'}"
        ])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 80)
        self.table.setColumnWidth(3, 80)
        self.table.setColumnWidth(4, 60)
        self.table.setColumnWidth(5, 70)
        self.table.setColumnWidth(6, 150)
        self.table.setColumnWidth(7, 90)
        self.table.setColumnWidth(8, 90)
        self.table.setColumnWidth(9, 200)  # Crosses ahead
        
        scroll.setWidget(self.table)
        layout.addWidget(scroll)
        
        btn_close = QPushButton("Close window")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)
    
    def update_table(self):
        """Обновить таблицу анализа"""
        ships = self.ships_ref() if callable(self.ships_ref) else self.ships_ref
        
        if len(ships) < 2:
            self.table.setRowCount(0)
            return
        
        self.table.setRowCount(0)
        
        row = 0
        for i in range(len(ships)):
            for j in range(i + 1, len(ships)):
                ship1 = ships[i]
                ship2 = ships[j]
                
                cpa_data = self.analyzer.calculate_cpa_tcpa(ship1, ship2)
                risk_index = self.analyzer.calculate_risk_index(
                    cpa_data['DCPA'], cpa_data['TCPA'], cpa_data['dist']
                )
                
                colreg_data = determine_colreg_situation(
                    ship1, ship2,
                    cpa_data['dist'], cpa_data['DCPA'], cpa_data['TCPA']
                )
                
                # Расчет пересечения курсов
                crossing_data = self.analyzer.calculate_course_crossing(ship1, ship2)
                
                self.table.insertRow(row)
                
                pair_item = QTableWidgetItem(f"{ship1.name} ↔ {ship2.name}")
                self.table.setItem(row, 0, pair_item)
                
                dist_item = QTableWidgetItem(f"{cpa_data['dist']:.0f}")
                self.table.setItem(row, 1, dist_item)
                
                cpa_item = QTableWidgetItem(f"{cpa_data['DCPA']:.0f}")
                self.table.setItem(row, 2, cpa_item)
                
                tcpa_val = cpa_data['TCPA']
                tcpa_str = f"{tcpa_val:.0f}" if tcpa_val != float('inf') else ""
                tcpa_item = QTableWidgetItem(tcpa_str)
                self.table.setItem(row, 3, tcpa_item)
                
                risk_item = QTableWidgetItem(f"{risk_index:.2f}")
                self.table.setItem(row, 4, risk_item)
                
                if risk_index > 0.7:
                    color = Qt.red
                elif risk_index > 0.4:
                    color = Qt.yellow
                else:
                    color = Qt.green
                
                for col in range(5):
                    self.table.item(row, col).setBackground(color)
                
                rule_item = QTableWidgetItem(f"Rule {colreg_data['rule']}")
                self.table.setItem(row, 5, rule_item)
                
                situation_item = QTableWidgetItem(colreg_data['situation'])
                self.table.setItem(row, 6, situation_item)
                
                details = colreg_data.get('details', {})
                bearing_1_to_2 = details.get('bearing_1_to_2', None)
                bearing_2_to_1 = details.get('bearing_2_to_1', None)
                
                if bearing_1_to_2 is not None:
                    bearing_1_item = QTableWidgetItem(f"{bearing_1_to_2:.0f}°")
                    self.table.setItem(row, 7, bearing_1_item)
                else:
                    self.table.setItem(row, 7, QTableWidgetItem("-"))
                
                if bearing_2_to_1 is not None:
                    bearing_2_item = QTableWidgetItem(f"{bearing_2_to_1:.0f}°")
                    self.table.setItem(row, 8, bearing_2_item)
                else:
                    self.table.setItem(row, 8, QTableWidgetItem("-"))
                
                # НОВАЯ КОЛОНКА 9: Crosses ahead
                if crossing_data['crossing_type']:
                    cross_text = f"{crossing_data['crossing_type']}\n(t={min(crossing_data['time_1_to_cross'], crossing_data['time_2_to_cross']):.0f}s)"
                    cross_item = QTableWidgetItem(cross_text)
                    cross_item.setBackground(QColor(255, 200, 200))
                    self.table.setItem(row, 9, cross_item)
                else:
                    self.table.setItem(row, 9, QTableWidgetItem("No crossing"))
                
                action1_item = QTableWidgetItem(colreg_data['ship1_action'])
                self.table.setItem(row, 10, action1_item)
                
                action2_item = QTableWidgetItem(colreg_data['ship2_action'])
                self.table.setItem(row, 11, action2_item)
                
                row += 1


def launch_collision_analysis(ships_ref):
    """Запустить окно анализа столкновений"""
    window = CollisionAnalysisWindow(ships_ref)
    window.show()
    return window