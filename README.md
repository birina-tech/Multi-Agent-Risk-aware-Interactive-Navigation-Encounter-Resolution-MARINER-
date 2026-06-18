# MARINER: Multi-Agent Risk-aware Interactive Navigation & Encounter Resolution

The **MARINER Project** is an interactive simulation framework that leverages *Large Language Models* (LLMs) to coordinate multi-agent autonomous surface vessels. This system acts as an AI-driven Vessel Traffic Service (VTS), analyzing the entire collision matrix and generating simultaneous, COLREGs-compliant maneuver commands (Rudder and RPM) for an entire swarm of ships.


## Features

- Interactive GUI Simulation built with PyQt5 and embedded Matplotlib 

- Global swarm coordination modeled as LLM-guided coordination 

- Universal LLM integration including support for DeepSeek, OpenAI, Anthropic, Groq, OpenRouter, and local models via Ollama

- Real-time COLREGs analysis including live calculation of DCPA, TCPA, Fuzzy Risk Index, and automatic situation classification (Rules 13, 14, 15, and 17)

- Task management system allowing to save and load multi-ship encounter scenarios to JSON files 

- Telemetry Logging (x, y, course, speed, rudder, rpm) to CSV files for post-simulation analysis.

## Methodology

### Vessel Dynamics Simulation (Nomoto Model)

The movement and steering physics of the vessels are modeled using the first-order **Nomoto** model. This widely accepted mathematical model accurately represents the yaw dynamics of marine vessels. Control inputs are human-readable parameters:

- **Rudder Angle**: Commanded in degrees (limited to a realistic turn rate of 5° per step).

- **Engine RPM**: Commanded as a percentage of maximum thrust.

### Risk Assessment & COLREGs

Collision risks are calculated using a Z-shaped fuzzy logic membership function combining three critical parameters: *DCPA* (Distance at Closest Point of Approach), *TCPA* (Time to Closest Point of Approach), and raw *Distance*. Depending on the relative bearings and risk index, the $collision_analyzer.py$ module explicitly classifies the encounter into specific COLREGs rules to guide the LLM's decision.

## Project Structure

MARINER/

├── ship_simulation.py      # Main entry point: PyQt5 GUI and simulation loop

├── llm_controller.py       # API integrations and VTS Swarm prompt logic

├── collision_analyzer.py   # Math engine for CPA, TCPA, Risk, and COLREGs

├── Tasks/                  # Directory for saved JSON scenario configurations

└── Logs/                   # Directory for CSV telemetry recordings

```
MARINER/
├── Tasks/                  # Directory for saved JSON scenario configurations
├── Logs/                   # Directory for simulation log exports
├── ship_simulation.py      # Main entry point: PyQt5 GUI and simulation loop
├── llm_controller.py       # API integrations and VTS Swarm prompt logic
└── collision_analyzer.py   # Math engine for CPA, TCPA, Risk, and COLREGs
```


## Installation & Dependencies
### Prerequisites

Python 3.8 or higher

Windows, macOS, or Linux

### Required Libraries

The project relies on standard scientific and GUI libraries. Install them via pip:

Bash
pip install PyQt5 numpy matplotlib requests


### Running the Application
No terminal flags are required. Simply launch the main script:

Bash
python ship_simulation.py


## Usability & Functionality
### Scene Control

**Pan View**: Hold the Left Mouse Button (LMB) on the map and drag.

**Zoom**: Use the Mouse Wheel to zoom in and out.

**Add Vessel**: Right-click (RMB) on an empty space on the map to spawn a new ship (prompts for Course and Speed).

**Manual Control**: Right-click (RMB) on an existing vessel to open its control panel. Here you can manually override the Rudder and RPM or toggle LLM control for that specific ship.

## LLM Setup & Safety
API keys are handled securely within the application's memory and are never written to disk.

- Navigate to **Menu → LLM → ⚙️ LLM Settings...**

- Select your provider (e.g., DeepSeek, OpenAI, Ollama).

- Paste your API key (if required).

- Click **Test connection** to verify before running.

## Task Management (Scenarios)
The application includes a *Task Management* system located in the Tasks menu for reproducible scenario testing:

- Save Task: Validates that vessels exist on the map (warns if empty).

-- Prompts the user for a filename via QInputDialog.getText().

-- Automatically creates a Tasks/ directory if it does not exist.

-- Serializes all current vessel parameters (name, coordinates, course, speed, rudder, RPM) into a structured JSON file.

-- Displays a success confirmation message.

- Load Task: Verifies the existence of the Tasks/ directory and .json files.

-- Presents a dropdown list of available tasks via QInputDialog.getItem().

-- Safely halts any running simulation and clears the canvas.

-- Reconstructs the vessels with their exact saved states and parameters.

-- Displays a summary of the loaded vessels.


## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Developers

PI **Yaroslav Burylin** 

Navigation Department

Admiral Ushakov Maritime State University (AUMSU)

Novorossiysk, Russian Federation

Co-I **Irina Benedyk** 

Civil, Structural, and Environmental Department

State University of New York (SUNY) at Buffalo

Buffalo, USA
