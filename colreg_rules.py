"""
colreg_rules.py
Определение правил МППСС на основе относительных пеленгов
"""
import numpy as np


def calculate_relative_bearing(ship1, ship2):
    """
    Рассчитать ОТНОСИТЕЛЬНЫЙ пеленг с ship1 на ship2
    Отсчитывается от курса ship1 (носа судна) по часовой стрелке 0-360°
    """
    # Истинный пеленг на ship2 (от севера)
    dx = ship2.x - ship1.x
    dy = ship2.y - ship1.y
    true_bearing_rad = np.arctan2(dx, dy)
    true_bearing_deg = np.degrees(true_bearing_rad)
    if true_bearing_deg < 0:
        true_bearing_deg += 360
    
    # Курс ship1 в градусах (0-360)
    ship1_course_deg = np.degrees(ship1.psi)
    if ship1_course_deg < 0:
        ship1_course_deg += 360
    
    # Относительный пеленг = истинный пеленг - курс судна
    relative_bearing = true_bearing_deg - ship1_course_deg
    
    # Нормализовать к 0-360
    if relative_bearing < 0:
        relative_bearing += 360
    elif relative_bearing >= 360:
        relative_bearing -= 360
    
    return relative_bearing


def check_rule_14(ship1, ship2):
    """
    Правило 14 - Встречная ситуация
    """
    bearing_1_to_2 = calculate_relative_bearing(ship1, ship2)
    bearing_2_to_1 = calculate_relative_bearing(ship2, ship1)
    
    def is_in_range_350_10(bearing):
        return bearing >= 350 or bearing <= 10
    
    if is_in_range_350_10(bearing_1_to_2) and is_in_range_350_10(bearing_2_to_1):
        return True, bearing_1_to_2, bearing_2_to_1
    
    return False, bearing_1_to_2, bearing_2_to_1


def check_rule_13(ship1, ship2):
    """
    Правило 13 - Обгон
    """
    bearing_1_to_2 = calculate_relative_bearing(ship1, ship2)
    bearing_2_to_1 = calculate_relative_bearing(ship2, ship1)
    
    def is_in_range_110_260(bearing):
        return 110 <= bearing <= 260
    
    def is_in_range_270_90(bearing):
        return bearing >= 270 or bearing <= 90
    
    v1_x = ship1.u * np.sin(ship1.psi)
    v1_y = ship1.u * np.cos(ship1.psi)
    v2_x = ship2.u * np.sin(ship2.psi)
    v2_y = ship2.u * np.cos(ship2.psi)
    
    v_rel_x = v2_x - v1_x
    v_rel_y = v2_y - v1_y
    
    dx = ship2.x - ship1.x
    dy = ship2.y - ship1.y
    
    approaching = (v_rel_x * dx + v_rel_y * dy) < 0
    
    if (is_in_range_110_260(bearing_1_to_2) and 
        is_in_range_270_90(bearing_2_to_1) and 
        approaching):
        return True, bearing_1_to_2, bearing_2_to_1
    
    return False, bearing_1_to_2, bearing_2_to_1


def check_rule_15(ship1, ship2):
    """
    Правило 15 - Пересечение курсов
    """
    is_rule_14, _, _ = check_rule_14(ship1, ship2)
    is_rule_13, _, _ = check_rule_13(ship1, ship2)
    
    if not is_rule_14 and not is_rule_13:
        bearing_1_to_2 = calculate_relative_bearing(ship1, ship2)
        bearing_2_to_1 = calculate_relative_bearing(ship2, ship1)
        return True, bearing_1_to_2, bearing_2_to_1
    
    return False, None, None


def check_rule_17_2(dist_m, cpa_m, tcpa_s):
    """
    Правило 17.2 - Критическое сближение
    """
    dist_nm = dist_m / 1852.0
    cpa_nm = cpa_m / 1852.0
    tcpa_min = tcpa_s / 60.0
    
    if dist_nm < 2.0 and cpa_nm < 1.0 and tcpa_min < 30.0:
        return True
    
    return False


def check_normal_conditions(dist_m, cpa_m, tcpa_s):
    """
    Проверка нормальных условий для применения правил 13, 14, 15
    """
    dist_nm = dist_m / 1852.0
    cpa_nm = cpa_m / 1852.0
    tcpa_min = tcpa_s / 60.0
    
    if 2.0 <= dist_nm <= 12.0 and cpa_nm < 2.0 and tcpa_min < 30.0:
        return True
    
    return False


def determine_colreg_situation(ship1, ship2, dist_m, cpa_m, tcpa_s):
    """
    Определить ситуацию МППСС для пары судов
    """
    # Пеленги вычисляются ВСЕГДА
    bearing_1_to_2 = calculate_relative_bearing(ship1, ship2)
    bearing_2_to_1 = calculate_relative_bearing(ship2, ship1)
    
    # Проверка критического сближения (правило 17.2)
    if check_rule_17_2(dist_m, cpa_m, tcpa_s):
        return {
            'rule': '17.2',
            'situation': 'Critical convergence',
            'ship1_action': 'Change course/speed',
            'ship2_action': 'Change course/speed',
            'details': {
                'bearing_1_to_2': bearing_1_to_2,
                'bearing_2_to_1': bearing_2_to_1,
                'dist_nm': dist_m / 1852.0,
                'cpa_nm': cpa_m / 1852.0,
                'tcpa_min': tcpa_s / 60.0
            }
        }
    
    # Проверка нормальных условий
    if not check_normal_conditions(dist_m, cpa_m, tcpa_s):
        return {
            'rule': 'None',
            'situation': 'Convergence',
            'ship1_action': 'Stand on',
            'ship2_action': 'Stand on',
            'details': {
                'bearing_1_to_2': bearing_1_to_2,
                'bearing_2_to_1': bearing_2_to_1,
            }
        }
    
    # Проверка правила 14 (встречная)
    is_rule_14, b1, b2 = check_rule_14(ship1, ship2)
    if is_rule_14:
        return {
            'rule': '14',
            'situation': 'Head-on',
            'ship1_action': 'Alter to Stb',
            'ship2_action': 'Alter to Stb',
            'details': {
                'bearing_1_to_2': b1,
                'bearing_2_to_1': b2,
                'dist_nm': dist_m / 1852.0,
                'cpa_nm': cpa_m / 1852.0,
                'tcpa_min': tcpa_s / 60.0
            }
        }
    
    # Проверка правила 13 (обгон)
    is_rule_13, b1, b2 = check_rule_13(ship1, ship2)
    if is_rule_13:
        return {
            'rule': '13',
            'situation': 'Overtaking',
            'ship1_action': 'Stand on (Rule 17.1)',
            'ship2_action': 'Give-way (Rule 16)',
            'details': {
                'bearing_1_to_2': b1,
                'bearing_2_to_1': b2,
                'dist_nm': dist_m / 1852.0,
                'cpa_nm': cpa_m / 1852.0,
                'tcpa_min': tcpa_s / 60.0,
                'overtaking_ship': ship2.name
            }
        }
    
    # Проверка правила 15 (пересечение)
    is_rule_15, b1, b2 = check_rule_15(ship1, ship2)
    if is_rule_15:
        def is_stand_on(bearing):
            return 10 <= bearing <= 110
        
        ship1_stand_on = is_stand_on(b2)
        
        if ship1_stand_on:
            ship1_action = 'Stand on (Rule 17.1)'
            ship2_action = 'Give-way (Rule 16)'
        else:
            ship1_action = 'Give-way (Rule 16)'
            ship2_action = 'Stand on (Rule 17.1)'
        
        return {
            'rule': '15',
            'situation': 'Crossing',
            'ship1_action': ship1_action,
            'ship2_action': ship2_action,
            'details': {
                'bearing_1_to_2': b1,
                'bearing_2_to_1': b2
            }
        }
    
    return {
        'rule': 'Unknown',
        'situation': 'Uncertain',
        'ship1_action': 'Stand on',
        'ship2_action': 'Stand on',
        'details': {
            'bearing_1_to_2': bearing_1_to_2,
            'bearing_2_to_1': bearing_2_to_1
        }
    }