from PyQt5.QtCore import QThread, pyqtSignal

class LLMWorker(QThread):
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