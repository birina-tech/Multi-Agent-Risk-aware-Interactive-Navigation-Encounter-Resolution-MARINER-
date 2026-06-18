"""
llm_controller.py
Универсальный LLM-координатор с поддержкой локальных и онлайн моделей.
"""
import requests
import json
import os
import numpy as np


class LLMCoordinator:
    # Конфигурация провайдеров
    PROVIDERS = {
        'ollama': {
            'name': 'Ollama (локально)',
            'url': 'http://localhost:11434/api/chat',
            'models': ['llama3', 'llama3.1:70b', 'qwen2.5:72b', 'mistral'],
            'default_model': 'llama3',
            'needs_key': False
        },
        'openai': {
            'name': 'OpenAI GPT-4o',
            'url': 'https://api.openai.com/v1/chat/completions',
            'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
            'default_model': 'gpt-4o',
            'needs_key': True,
            'key_env': 'OPENAI_API_KEY'
        },
        'anthropic': {
            'name': 'Anthropic Claude',
            'url': 'https://api.anthropic.com/v1/messages',
            'models': ['claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022'],
            'default_model': 'claude-sonnet-4-20250514',
            'needs_key': True,
            'key_env': 'ANTHROPIC_API_KEY'
        },
        'groq': {
            'name': 'Groq (быстро, бесплатно)',
            'url': 'https://api.groq.com/openai/v1/chat/completions',
            'models': ['llama-3.3-70b-versatile', 'llama-3.1-70b-versatile', 'mixtral-8x7b-32768'],
            'default_model': 'llama-3.3-70b-versatile',
            'needs_key': True,
            'key_env': 'GROQ_API_KEY'
        },
        'deepseek': {
            'name': 'DeepSeek (дёшево)',
            'url': 'https://api.deepseek.com/v1/chat/completions',
            'models': ['deepseek-chat', 'deepseek-reasoner'],
            'default_model': 'deepseek-chat',
            'needs_key': True,
            'key_env': 'DEEPSEEK_API_KEY'
        },
        'openrouter': {
            'name': 'OpenRouter (много моделей)',
            'url': 'https://openrouter.ai/api/v1/chat/completions',
            'models': ['meta-llama/llama-3.1-70b-instruct', 'anthropic/claude-3.5-sonnet'],
            'default_model': 'meta-llama/llama-3.1-70b-instruct',
            'needs_key': True,
            'key_env': 'OPENROUTER_API_KEY'
        }
    }
    
    def __init__(self, provider='ollama', model=None, api_key=None):
        self.provider = provider
        config = self.PROVIDERS.get(provider, self.PROVIDERS['ollama'])
        
        self.url = config['url']
        self.model = model or config['default_model']
        self.needs_key = config['needs_key']
        
        # API ключ: явно переданный > переменная окружения
        if api_key:
            self.api_key = api_key
        elif config.get('key_env'):
            self.api_key = os.environ.get(config['key_env'], '')
        else:
            self.api_key = None
        
        self.system_prompt = """You are an AI Vessel Traffic Coordinator. 
Your task: Given a complete collision analysis table for ALL vessel pairs, 
generate coordinated rudder and RPM commands for EACH vessel to avoid collisions 
while strictly complying with COLREGs.

INPUT: You receive a table with columns:
- Pair: vessel names
- Distance (m), CPA (m), TCPA (s), Risk Index
- COLREG Rule (13/14/15/17)
- Required actions for each vessel

OUTPUT: JSON with commands for EACH vessel:
{
  "vessel_commands": {
    "Ship_1": {"rudder_deg": 15, "rpm_percent": 50, "reasoning": "..."},
    "Ship_2": {"rudder_deg": 0, "rpm_percent": 50, "reasoning": "..."}
  }
}

COORDINATION RULES:
1. If a vessel appears in MULTIPLE pairs, prioritize the MOST CRITICAL situation:
   - Priority order: Head-on (Rule 14) > Crossing give-way (Rule 15) > Overtaking (Rule 13) > Crossing stand-on (Rule 17)
2. For Head-on (Rule 14): BOTH vessels must turn STARBOARD (positive rudder)
3. For Crossing (Rule 15): Give-way vessel turns STARBOARD, stand-on vessel maintains
4. For Overtaking (Rule 13): Overtaking vessel keeps clear, overtaken maintains
5. Avoid contradictory commands: if two vessels are in head-on, BOTH must turn right
6. ECO-MODE: Prefer rudder changes over RPM changes. Keep RPM at 50% unless emergency.
7. Smooth maneuvers: Avoid sharp rudder changes (>15° at once).

CRITICAL THRESHOLDS:
- Risk > 0.75 or TCPA < 60s or CPA < 250m: URGENT action required
- Risk 0.5-0.75: Begin maneuver
- Risk < 0.5: Maintain course if stand-on, minor adjustments if give-way

Respond STRICTLY with valid JSON. No extra text."""
        
        # Статус подключения
        self.last_status = None
        self.last_error = None

    def test_connection(self):
        """Проверка подключения к провайдеру. Возвращает (success, message)"""
        try:
            if self.provider == 'ollama':
                # Проверяем, запущена ли Ollama
                r = requests.get('http://localhost:11434/api/tags', timeout=5)
                if r.status_code == 200:
                    # Проверяем, есть ли нужная модель
                    models = r.json().get('models', [])
                    model_names = [m.get('name', '') for m in models]
                    if any(self.model in m for m in model_names):
                        return True, f"Ollama запущена, модель {self.model} доступна"
                    else:
                        return False, f"Модель {self.model} не найдена. Доступны: {', '.join(model_names[:5])}"
                else:
                    return False, "Ollama вернула ошибку"
            
            elif self.provider == 'anthropic':
                if not self.api_key:
                    return False, "API ключ не установлен"
                headers = {
                    'x-api-key': self.api_key,
                    'anthropic-version': '2023-06-01'
                }
                # Простой запрос для проверки
                payload = {
                    'model': self.model,
                    'max_tokens': 10,
                    'messages': [{'role': 'user', 'content': 'Hi'}]
                }
                r = requests.post(self.url, json=payload, headers=headers, timeout=10)
                if r.status_code == 200:
                    return True, "Подключение к Anthropic успешно"
                else:
                    return False, f"Ошибка {r.status_code}: {r.text[:100]}"
            
            else:  # OpenAI-совместимые API
                if not self.api_key:
                    return False, "API ключ не установлен"
                headers = {'Authorization': f'Bearer {self.api_key}'}
                payload = {
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': 'Hi'}],
                    'max_tokens': 10
                }
                r = requests.post(self.url, json=payload, headers=headers, timeout=10)
                if r.status_code == 200:
                    return True, f"Подключение к {self.provider} успешно"
                else:
                    return False, f"Ошибка {r.status_code}: {r.text[:100]}"
        
        except requests.exceptions.ConnectionError:
            return False, f"Нет соединения с {self.provider}. Проверьте интернет/запуск сервиса."
        except Exception as e:
            return False, f"Ошибка: {str(e)[:100]}"

    def format_analysis_table(self, ships, collision_data):
        table_text = "CURRENT VESSEL STATES:\n"
        for ship in ships:
            table_text += (f"{ship.name}: pos=({ship.x:.0f},{ship.y:.0f}), "
                           f"heading={ship.get_heading_deg():.0f}°, "
                           f"speed={ship.u:.1f} m/s, "
                           f"rudder={ship.rudder_cmd:.0f}°, rpm={ship.rpm_cmd:.0f}%\n")
        
        table_text += "\nCOLLISION ANALYSIS TABLE:\n"
        table_text += "Pair | Distance(m) | CPA(m) | TCPA(s) | Risk | Rule | Actions\n"
        table_text += "-" * 80 + "\n"
        
        for pair_data in collision_data:
            tcpa_str = f"{pair_data['tcpa']:.0f}" if pair_data['tcpa'] != float('inf') else "inf"
            table_text += (f"{pair_data['pair']} | {pair_data['dist']:.0f} | "
                           f"{pair_data['cpa']:.0f} | {tcpa_str} | "
                           f"{pair_data['risk']:.2f} | Rule {pair_data['rule']} | "
                           f"{pair_data['actions']}\n")
        return table_text

    def _call_openai_compatible(self, user_message):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            'temperature': 0.2,
            'response_format': {'type': 'json_object'}
        }
        response = requests.post(self.url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def _call_anthropic(self, user_message):
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        payload = {
            'model': self.model,
            'max_tokens': 1024,
            'temperature': 0.2,
            'system': self.system_prompt,
            'messages': [{'role': 'user', 'content': user_message}]
        }
        response = requests.post(self.url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()['content'][0]['text']

    def _call_ollama(self, user_message):
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            'stream': False,
            'format': 'json',
            'options': {'temperature': 0.2}
        }
        response = requests.post(self.url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['message']['content']

    def get_coordinated_commands(self, ships, collision_data):
        if not collision_data:
            return {ship.name: {"rudder_deg": 0, "rpm_percent": 50, "reasoning": "No threats"} 
                    for ship in ships}
        
        table_text = self.format_analysis_table(ships, collision_data)
        user_message = (f"Coordinate maneuvers for all vessels based on this analysis:\n\n"
                        f"{table_text}\n\nGenerate commands for ALL vessels listed above.")
        
        try:
            if self.provider == 'ollama':
                content = self._call_ollama(user_message)
            elif self.provider == 'anthropic':
                content = self._call_anthropic(user_message)
            else:
                content = self._call_openai_compatible(user_message)
            
            content = content.replace("```json", "").replace("```", "").strip()
            commands = json.loads(content)
            self.last_status = 'ok'
            return commands.get('vessel_commands', {})
            
        except Exception as e:
            error_msg = str(e)
            self.last_status = 'error'
            self.last_error = error_msg
            print(f"LLM Coordinator Error ({self.provider}): {error_msg}")
            return {ship.name: {"rudder_deg": 0, "rpm_percent": 50, "reasoning": f"Error: {error_msg[:50]}"} 
                    for ship in ships}

    def apply_commands(self, ships, commands):
        for ship in ships:
            if ship.name in commands:
                cmd = commands[ship.name]
                target_rudder = cmd.get('rudder_deg', 0)
                target_rpm = cmd.get('rpm_percent', 50)
                
                rudder_diff = target_rudder - ship.rudder_cmd
                if abs(rudder_diff) > 5:
                    ship.rudder_cmd += np.sign(rudder_diff) * 5
                else:
                    ship.rudder_cmd = target_rudder
                
                rpm_diff = target_rpm - ship.rpm_cmd
                if abs(rpm_diff) > 10:
                    ship.rpm_cmd += np.sign(rpm_diff) * 10
                else:
                    ship.rpm_cmd = target_rpm
                
                ship.llm_decision = cmd