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

        self.system_prompt = """You are an AI Vessel Traffic Controller.
You receive ship position data. Your first priority is to maintain ship safety. 
Your second priority is to maintain the course set for the vessel using base_heading_deg (return to it is nessesary) if ship is in safe situation. 
To achive that your ONLY task is to define and output rudder and RPM values.
You do NOT determine COLREG rules. Rules are already determined by the maritime code.


INPUT FORMAT (JSON):
{
  "ships": [
    {
      "name": "Ship_1",
      "current_heading_deg": 45,
      "base_heading_deg": 45,
      "heading_diff_deg": 15,
      "speed_ms": 5.0,
      "current_rudder": 0,
      "current_rpm": 50,
      "status": "MUST_YIELD" | "HOLD_COURSE" | "RETURN_TO_COURSE",
      "no_left_turn": true | false,
      "in_maneuver": true | false,
      "pairs": [
        {
          "other_ship": "Ship_2",
          "rule": "14" | "15" | "13" | "17.2",
          "role": "GIVE_WAY" | "STAND_ON" | "BOTH_ALTER",
          "cpa_m": 500,
          "tcpa_s": 120,
          "crosses_ahead": "Ship_2 crosses Ship_1 ahead" | null
        }
      ]
    }
  ]
}

OUTPUT FORMAT (strict JSON only, no extra text):
{
  "vessel_commands": {
    "Ship_1": {
      "rudder_deg": 15,
      "rpm_percent": 50,
      "reasoning": "Brief explanation: Rule 15 give-way vessel, turning starboard to pass astern"
    }
  }
}

RULES (follow strictly):

1. STATUS PRIORITY:
   - If status == "MUST_YIELD" -> you MUST maneuver (change rudder or RPM).
   - If status == "HOLD_COURSE" -> keep rudder=0, rpm=50 (maintain course and speed).
   - If status == "RETURN_TO_COURSE" -> gradually steer toward base_heading_deg set for this ship.
   - If a ship is MUST_YIELD for one pair but HOLD_COURSE for another -> choose MUST_YIELD.

2. MANEUVER DIRECTION (HARD RULE) used when there are other ships nearby:
   - ALWAYS prefer STARBOARD turn (positive rudder).
   - CRITICAL CONVERGENCE (Rule 17.2 / Emergency / CPA < 1000 meters): YOU MUST TURN STARBOARD. Port turn (negative rudder) is STRICTLY FORBIDDEN in emergencies.
   - If no_left_turn == true -> rudder_deg MUST be >= 0. Negative values are FORBIDDEN.
   - If status == "HOLD_COURSE" -> rudder_deg MUST be 0, rpm_percent MUST be 50. NO MANEUVERS ALLOWED for stand-on vessels.
   - If status == "MUST_YIELD" -> rudder_deg MUST be >= 0 (STARBOARD turn ONLY). NEGATIVE RUDDER IS STRICTLY FORBIDDEN.
   - If status == "RETURN_TO_COURSE" -> small rudder toward base_heading, max +/-10 deg.

3. MANEUVER MAGNITUDE:
   - For Rule 14 (head-on): rudder should be from 15 to 25 deg starboard.
   - For Rule 15 (crossing, give-way): rudder should be from 15 to 25 deg starboard.
   - For Rule 13 (overtaking): rudder should be from 10 to 20 deg away from overtaken vessel.
   - For Rule 17.2 (critical convergence / emergency): rudder should be from 20 to 35 deg STARBOARD. Reduce RPM to 30-40% if CPA < 500 meters.
   - In other situations apply smooth changes: max 15 deg rudder change per step.

4. RETURN TO BASE COURSE:
   - If status == "RETURN_TO_COURSE":
     * Calculate rudder needed to base_heading_deg.
     * Use small rudder (from -10 to 10 deg) toward base course. Negative rudder is allowed if it leades to faster return to base_heading_deg.
     * If you defined that heading_diff_deg < abs(3) deg -> output rudder equal to 0 (course restored).
     * Maintain RPM at 50% during the return to base course.

5. ECO-MODE:
   - Prefer rudder changes over RPM changes.
   - Keep RPM at 50% unless CPA < 1000m or emergency.
   - If reducing RPM: min 30%, never 0%.

6. NO MANEUVER NEEDED:
   - If status == "HOLD_COURSE" AND not returning -> output rudder=0, rpm=50.

Respond with valid JSON only. No markdown, no explanation outside JSON."""

        # Статус подключения
        self.last_status = None
        self.last_error = None

    def test_connection(self):
        """Provider connection check. Return (success, message)"""
        try:
            if self.provider == 'ollama':
                r = requests.get('http://localhost:11434/api/tags', timeout=5)
                if r.status_code == 200:
                    models = r.json().get('models', [])
                    model_names = [m.get('name', '') for m in models]
                    if any(self.model in m for m in model_names):
                        return True, f"ollama is launched, model {self.model} is available"
                    else:
                        return False, f"Model {self.model} is not accessible. Available: {', '.join(model_names[:5])}"
                else:
                    return False, "ollama returned an error"

            elif self.provider == 'anthropic':
                if not self.api_key:
                    return False, "API key is not set"
                headers = {
                    'x-api-key': self.api_key,
                    'anthropic-version': '2023-06-01'
                }
                payload = {
                    'model': self.model,
                    'max_tokens': 10,
                    'messages': [{'role': 'user', 'content': 'Hi'}]
                }
                r = requests.post(self.url, json=payload, headers=headers, timeout=10)
                if r.status_code == 200:
                    return True, "Connection to Anthropic is sucessful"
                else:
                    return False, f"Error {r.status_code}: {r.text[:100]}"

            else:  # OpenAI-совместимые API
                if not self.api_key:
                    return False, "API key is not set"
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
        """Форматирование данных для LLM.
        collision_data — disctionary {'ships': [...]} от collect_collision_data().
        Use JSON format — LLM resive only sctructured data.
        """
        import json
        return json.dumps(collision_data, indent=2, ensure_ascii=False)

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


            # Remove possible markdown-блоки
            content = content.replace("```json", "").replace("```", "").strip()
            commands = json.loads(content)
            self.last_status = 'ok'

            # Note: возвращаем ПОЛНЫЙ ответ, а не извлечённый vessel_commands
            # чтобы apply_commands мог найти ключ "vessel_commands"
            return commands

        except Exception as e:
            error_msg = str(e)
            self.last_status = 'error'
            self.last_error = error_msg
            print(f"LLM Coordinator Error ({self.provider}): {error_msg}")
            print(f"Raw content: {content[:300] if 'content' in dir() else 'N/A'}")
            return {ship.name: {"rudder_deg": 0, "rpm_percent": 50,
                                "reasoning": f"Error: {error_msg[:50]}"}
                    for ship in ships}

    def apply_commands(self, ships, commands):
        """
        Применить команды от LLM к судам.
        
        commands может быть в одном из двух форматов:
        1. {"vessel_commands": {"Ship_1": {...}}} — с обёрткой
        2. {"Ship_1": {...}, "Ship_2": {...}} — без обёртки (уже извлечено)
        """
        if not commands:
            print("No commands received from LLM")
            return

        # Определяем формат
        vessel_commands = None

        if isinstance(commands, dict):
            if "vessel_commands" in commands:
                # Формат 1: с обёрткой
                vessel_commands = commands["vessel_commands"]
            else:
                # Формат 2: без обёртки — имена судов как ключи
                ship_names = [s.name for s in ships]
                if any(name in commands for name in ship_names):
                    vessel_commands = commands
                else:
                    print(f"Unexpected format. Keys: {list(commands.keys())}")
                    return

        if not vessel_commands:
            print(f"No vessel commands found. Type: {type(commands)}")
            return

        #print(f"Applying commands to {len(vessel_commands)} vessels")

        for ship in ships:
            if ship.name in vessel_commands:
                cmd = vessel_commands[ship.name]

                rudder = cmd.get("rudder_deg", 0)
                rpm = cmd.get("rpm_percent", 50)
                reasoning = cmd.get("reasoning", "No reasoning provided")

                ship.apply_llm_command(rudder, rpm)
                ship.llm_decision = cmd
                ship.llm_reasoning = reasoning

                #print(f"OK {ship.name}: rudder={rudder} deg, rpm={rpm}%, reasoning={reasoning[:60]}")
            else:
                if ship.llm_controlled:
                    print(f"Warning: {ship.name} not found in LLM commands")