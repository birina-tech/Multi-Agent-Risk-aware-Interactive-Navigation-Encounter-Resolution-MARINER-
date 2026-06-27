from PyQt5.QtCore import QThread, pyqtSignal

       
        
class LLMWorker(QThread):
    # Emit the ship's name along with its command dictionary
    result_ready = pyqtSignal(str, dict)
    error_occurred = pyqtSignal(str, str) # ship_name, error_msg
    
    def __init__(self, coordinator, ego_ship_name, collision_data):
        super().__init__()
        self.coordinator = coordinator
        self.ego_ship_name = ego_ship_name
        self.collision_data = collision_data
        self.is_running = True
    
    def run(self):
        try:
            command = self.coordinator.get_ego_command(self.collision_data)
            if self.is_running:
                self.result_ready.emit(self.ego_ship_name, command)
        except Exception as e:
            if self.is_running:
                self.error_occurred.emit(self.ego_ship_name, str(e))
    
    def stop(self):
        self.is_running = False