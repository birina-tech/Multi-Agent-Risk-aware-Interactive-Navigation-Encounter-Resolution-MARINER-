"""
collision_analyzer.py
Анализ столкновений и правил МППСС для симуляции судов.
Основан на алгоритмах из статьи "Large Language Model-based Decision-making for COLREGs..."
"""
import sys
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
                             QStatusBar, QScrollArea, QAbstractItemView, QApplication)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor
import warnings
warnings.filterwarnings("ignore")


class CollisionAnalyzer:
    """Класс для расчета CPA, TCPA, Risk Index и правил МППСС"""
    
    def __init__(self):
        self.max_distance_nm = 12.0  # Максимальная дистанция анализа (морские мили)
        self.max_distance_m = self.max_distance_nm * 1852.0  # Перевод в метры
    
    def calculate_cpa_tcpa(self, ship1, ship2):
        """
        Расчет CPA и TCPA по формулам (13) и (14) из статьи.
        DCPA(t) = R(t) * sin(alpha(t))
        TCPA(t) = R(t) * cos(alpha(t)) / Vrel(t)
        """
        dx = ship2.x - ship1.x
        dy = ship2.y - ship1.y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Скорости в глобальной системе (X=Восток, Y=Север)
        # В ship_simulation.py: psi=0 это Север (+Y)
        v1_x = ship1.u * np.sin(ship1.psi)
        v1_y = ship1.u * np.cos(ship1.psi)
        v2_x = ship2.u * np.sin(ship2.psi)
        v2_y = ship2.u * np.cos(ship2.psi)
        
        # Относительная скорость
        v_rel_x = v2_x - v1_x
        v_rel_y = v2_y - v1_y
        v_rel = np.sqrt(v_rel_x**2 + v_rel_y**2)
        
        if v_rel > 0.01:
            # Углы как в cpa_calculations.py
            psi_LOS = np.arctan2(dy, dx)
            psi_v_rel = np.arctan2(-v_rel_y, -v_rel_x)
            alpha = psi_LOS - psi_v_rel
            
            DCPA = abs(dist * np.sin(alpha))
            TCPA = (dist * np.cos(alpha)) / v_rel
            TCPA = max(0, TCPA)  # TCPA не может быть отрицательным
        else:
            DCPA = dist
            TCPA = float('inf')
        
        return {
            'dist': dist,
            'DCPA': DCPA,
            'TCPA': TCPA,
            'v_rel': v_rel
        }
    
    def calculate_risk_index(self, dcpa, tcpa, dist):
        """
        Расчет индекса риска по формуле (15) из статьи:
        Risk(t) = (f(DCPA) + f(TCPA) + f(R(t))) / 3
        где f - Z-shaped Fuzzy Membership функция.
        Пороги из Eq. (3): DCPA < 250м, TCPA < 60с, R < 1000м.
        """
        f_DCPA = max(0, min(1, (1000 - dcpa) / (1000 - 250))) if dcpa < 1000 else 0
        f_TCPA = max(0, min(1, (300 - tcpa) / (300 - 60))) if tcpa < 300 else 0
        f_Range = max(0, min(1, (2000 - dist) / (2000 - 1000))) if dist < 2000 else 0
        
        return (f_DCPA + f_TCPA + f_Range) / 3
    
    def determine_colreg_rule(self, ship1, ship2):
        """
        Определение правила МППСС по классификации из Eq. (2) и decision_making.py.
        relative_bearing = psi - los_ob
        """
        dx = ship2.x - ship1.x
        dy = ship2.y - ship1.y
        los_ob = np.arctan2(dy, dx)
        
        # Относительный пеленг (нормализованный в [-pi, pi])
        rel_bearing = ship1.psi - los_ob
        rel_bearing = np.arctan2(np.sin(rel_bearing), np.cos(rel_bearing))
        rel_bearing_deg = np.degrees(rel_bearing)
        
        # Классификация ситуаций
        if abs(rel_bearing_deg) <= 6:
            # Head-on (Rule 14)
            return 14, 'both-give-way', 'both-give-way'
        
        elif 6 < rel_bearing_deg <= 112:
            # Crossing - Give way (Rule 15)
            return 15, 'give-way', 'stand-on'
        
        elif -118 <= rel_bearing_deg < -6:
            # Crossing - Stand on (Rule 17)
            return 17, 'stand-on', 'give-way'
        
        elif rel_bearing_deg > 112 or rel_bearing_deg < -118:
            # Overtaking (Rule 13)
            # Определяем, кто обгоняет (кто быстрее)
            if ship1.u > ship2.u:
                return 13, 'give-way', 'stand-on'
            else:
                return 13, 'stand-on', 'give-way'
        
        return 0, 'none', 'none'
    
    def get_required_action(self, rule_num, role):
        """Определение требуемого действия"""
        if rule_num == 0 or role == 'none':
            return "Нет действий"
        
        actions = {
            (14, 'both-give-way'): "Изменить курс ВПРАВО",
            (13, 'give-way'): "Уступить дорогу (не пересекать курс)",
            (13, 'stand-on'): "Сохранять курс и скорость",
            (15, 'give-way'): "Изменить курс ВПРАВО / Снизить скорость",
            (15, 'stand-on'): "Сохранять курс и скорость",
            (17, 'stand-on'): "Сохранять курс и скорость",
            (17, 'give-way'): "Изменить курс ВПРАВО / Снизить скорость"
        }
        
        return actions.get((rule_num, role), "Анализ ситуации")


class CollisionAnalysisWindow(QMainWindow):
    """Окно отображения таблицы анализа столкновений"""
    
    def __init__(self, ships_ref):
        super().__init__()
        self.ships_ref = ships_ref
        self.analyzer = CollisionAnalyzer()
        
        self.setWindowTitle(" Анализ столкновений и МППСС")
        self.resize(1400, 700)
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_analysis)
        self.timer.start(1000)  # Обновление каждую секунду
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        title_label = QLabel("📊 Таблица анализа столкновений (CPA/TCPA/МППСС)")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        info_label = QLabel(
            "Анализ пар судов на дистанции ≤ 12 морских миль (22 224 м). "
            "Цветовая индикация: 🔴 критично | 🟠 высоко | 🟡 средне | 🟢 низкий риск | ⚪ нет угрозы"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        main_layout.addWidget(info_label)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Пара судов",
            "Дистанция (м)",
            "CPA (м)",
            "TCPA (с)",
            "Risk Index",
            "Правило МППСС",
            "Требуемые действия"
        ])
        
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setStyleSheet("""
            QTableWidget {
                font-size: 11px;
                gridline-color: #cccccc;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #4a90d9;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
        """)
        
        scroll_area.setWidget(self.table)
        main_layout.addWidget(scroll_area)
        
        self.statusBar().showMessage("Ожидание данных...")
    
    def update_analysis(self):
        """Обновление таблицы анализа"""
        if not self.ships_ref or len(self.ships_ref) < 2:
            self.table.setRowCount(0)
            self.statusBar().showMessage("Недостаточно судов для анализа (нужно ≥ 2)")
            return
        
        ships = self.ships_ref
        n = len(ships)
        row = 0
        
        # Количество уникальных пар
        num_pairs = n * (n - 1) // 2
        self.table.setRowCount(num_pairs)
        
        for i in range(n):
            for j in range(i + 1, n):
                ship1 = ships[i]
                ship2 = ships[j]
                
                # 1. Расчет CPA/TCPA
                cpa_data = self.analyzer.calculate_cpa_tcpa(ship1, ship2)
                
                # 2. Расчет индекса риска
                risk_index = self.analyzer.calculate_risk_index(
                    cpa_data['DCPA'], cpa_data['TCPA'], cpa_data['dist']
                )
                
                # 3. Определение правила МППСС (только если дистанция < 12 миль)
                if cpa_data['dist'] < self.analyzer.max_distance_m:
                    rule_num, role1, role2 = self.analyzer.determine_colreg_rule(ship1, ship2)
                    action1 = self.analyzer.get_required_action(rule_num, role1)
                    action2 = self.analyzer.get_required_action(rule_num, role2)
                    
                    if rule_num > 0:
                        rule_text = f"Правило {rule_num}"
                        actions_text = f"{ship1.name}: {action1}\n{ship2.name}: {action2}"
                    else:
                        rule_text = "—"
                        actions_text = "Нет угрозы"
                    
                    risk_color = self.get_risk_color(cpa_data['DCPA'], cpa_data['TCPA'], cpa_data['dist'])
                else:
                    rule_text = "> 12 миль"
                    actions_text = "—"
                    risk_color = QColor(240, 240, 240)
                
                # 4. Заполнение строки таблицы
                self.table.setItem(row, 0, QTableWidgetItem(f"{ship1.name} ↔ {ship2.name}"))
                self.table.setItem(row, 1, QTableWidgetItem(f"{cpa_data['dist']:.0f}"))
                self.table.setItem(row, 2, QTableWidgetItem(f"{cpa_data['DCPA']:.0f}"))
                
                tcpa_text = f"{cpa_data['TCPA']:.0f}" if cpa_data['TCPA'] != float('inf') else "∞"
                self.table.setItem(row, 3, QTableWidgetItem(tcpa_text))
                self.table.setItem(row, 4, QTableWidgetItem(f"{risk_index:.2f}"))
                self.table.setItem(row, 5, QTableWidgetItem(rule_text))
                self.table.setItem(row, 6, QTableWidgetItem(actions_text))
                
                # Применяем цветовую индикацию ко всей строке
                for col in range(7):
                    item = self.table.item(row, col)
                    if item:
                        item.setBackground(risk_color)
                
                row += 1
        
        # Обновление статусной строки
        active_pairs = sum(
            1 for i in range(n) 
            for j in range(i + 1, n) 
            if self.analyzer.calculate_cpa_tcpa(ships[i], ships[j])['dist'] < self.analyzer.max_distance_m
        )
        
        self.statusBar().showMessage(
            f"Всего пар: {num_pairs} | "
            f"Активных (≤ 12 миль): {active_pairs} | "
            f"Судов в симуляции: {n}"
        )
    
    def get_risk_color(self, dcpa, tcpa, dist):
        """Цветовая индикация риска по порогам из статьи (Eq. 3)"""
        if dcpa < 250 and tcpa < 60:
            return QColor(255, 100, 100)  # Красный - критический
        elif dcpa < 500 and tcpa < 120:
            return QColor(255, 180, 100)  # Оранжевый - высокий
        elif dcpa < 1000 and tcpa < 300:
            return QColor(255, 255, 150)  # Желтый - средний
        elif dist < self.analyzer.max_distance_m * 0.5:
            return QColor(200, 255, 200)  # Зеленый - низкий
        else:
            return QColor(240, 240, 240)  # Серый - нет угрозы


def launch_collision_analysis(ships_ref):
    """Функция для запуска окна анализа из основной симуляции"""
    window = CollisionAnalysisWindow(ships_ref)
    window.show()
    return window


if __name__ == "__main__":
    # Для тестирования создаем фиктивные суда
    class TestShip:
        def __init__(self, x, y, psi_deg, u, name):
            self.x = x
            self.y = y
            self.psi = np.deg2rad(psi_deg)
            self.u = u
            self.name = name
    
    test_ships = [
        TestShip(0, 0, 0, 5, "Ship_1"),
        TestShip(2000, 0, 180, 5, "Ship_2"),
        TestShip(1000, 2000, 270, 4, "Ship_3")
    ]
    
    app = QApplication([])
    window = launch_collision_analysis(test_ships)
    app.exec_()