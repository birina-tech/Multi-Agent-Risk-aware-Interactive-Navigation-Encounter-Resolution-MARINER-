import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

class ShipCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.view_center_x = 0.0
        self.view_center_y = 0.0
        self.view_scale = 5000.0
        self.min_scale = 200.0
        self.max_scale = 20000.0
        self.zoom_factor = 1.2
        self.is_panning = False
        self.pan_start_pos = None
        self.pan_start_center = None
        self.vector_length_minutes = 12.0
        self.track_length_meters = 5000.0
        
        # Убираем все отступы - график занимает всю область окна
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        self.ax.set_xlim(-self.view_scale, self.view_scale)
        self.ax.set_ylim(-self.view_scale, self.view_scale)
        
        # Метки ВНУТРИ области построения
        self.ax.tick_params(axis='both', which='both', direction='in', 
                            top=True, right=True, left=True, bottom=True)
        self.ax.tick_params(axis='x', which='both', pad=-15)
        self.ax.tick_params(axis='y', which='both', pad=-40)  # было -40, стало -60
        
        self.ax.grid(True, alpha=0.5)
        
        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('scroll_event', self.on_scroll)
        
        self.ships = []
        self.on_click_callback = None
        self.on_ship_click_callback = None
        self.on_mouse_move_callback = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = event.size().width()
        h = event.size().height()
        if w > 0 and h > 0:
            self.fig.set_size_inches(w / self.fig.dpi, h / self.fig.dpi)
            self.update_plot(self.ships, False, 0.0)

    def _get_view_limits(self):
        bbox = self.ax.get_window_extent()
        if bbox.width == 0 or bbox.height == 0:
            return (-self.view_scale, self.view_scale, 
                    -self.view_scale, self.view_scale)
        
        window_aspect = bbox.width / bbox.height
        base_range = self.view_scale
        
        if window_aspect > 1.0:
            x_range = base_range * window_aspect
            y_range = base_range
        else:
            x_range = base_range
            y_range = base_range / window_aspect
        
        return (self.view_center_x - x_range, self.view_center_x + x_range,
                self.view_center_y - y_range, self.view_center_y + y_range)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self.is_panning = True
            self.pan_start_pos = (event.x, event.y)
            self.pan_start_center = (self.view_center_x, self.view_center_y)
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button == 3:
            x, y = event.xdata, event.ydata
            clicked_ship = None
            click_radius = max(60, self.view_scale * 0.02)
            for ship in self.ships:
                if ship.distance_to(x, y) < click_radius:
                    clicked_ship = ship
                    break
            if clicked_ship:
                if self.on_ship_click_callback:
                    self.on_ship_click_callback(clicked_ship)
            else:
                if self.on_click_callback:
                    self.on_click_callback(x, y)

    def on_motion(self, event):
        if self.is_panning and event.inaxes == self.ax:
            dx_pix = event.x - self.pan_start_pos[0]
            dy_pix = event.y - self.pan_start_pos[1]
            bbox = self.ax.get_window_extent()
            if bbox.width == 0 or bbox.height == 0:
                return
            
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            data_width = xlim[1] - xlim[0]
            data_height = ylim[1] - ylim[0]
            
            pixels_per_data_x = bbox.width / data_width
            pixels_per_data_y = bbox.height / data_height
            
            dx_data = dx_pix / pixels_per_data_x
            dy_data = -dy_pix / pixels_per_data_y
            
            self.view_center_x = self.pan_start_center[0] - dx_data
            self.view_center_y = self.pan_start_center[1] + dy_data
            
            x_min, x_max, y_min, y_max = self._get_view_limits()
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.draw()
        
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            if self.on_mouse_move_callback:
                self.on_mouse_move_callback(event.xdata, event.ydata)

    def on_release(self, event):
        if event.button == 1 and self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.OpenHandCursor)

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 'up':
            self.view_scale /= self.zoom_factor
        elif event.button == 'down':
            self.view_scale *= self.zoom_factor
        
        self.view_scale = np.clip(self.view_scale, self.min_scale, self.max_scale)
        
        x_min, x_max, y_min, y_max = self._get_view_limits()
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.draw()

    def draw_velocity_vector(self, ship):
        if ship.u < 0.1:
            return
        vector_length = ship.u * self.vector_length_minutes * 60
        vector_end_x = ship.x + vector_length * np.sin(ship.psi)
        vector_end_y = ship.y + vector_length * np.cos(ship.psi)
        self.ax.annotate('', xy=(vector_end_x, vector_end_y), xytext=(ship.x, ship.y),
                         arrowprops=dict(arrowstyle='->', color='blue',
                                         lw=0.5, mutation_scale=15), zorder=6)

    def _apply_tick_style(self):
        """Применяет стиль меток: сдвиг внутрь + белая подложка для читаемости"""
        self.ax.tick_params(axis='both', which='both', direction='in', 
                            top=True, right=True, left=True, bottom=True)
        self.ax.tick_params(axis='x', which='both', pad=-15)
        self.ax.tick_params(axis='y', which='both', pad=-40)  # глубокий сдвиг вправо
        
        # Белая подложка под цифры, чтобы они читались поверх сетки и линий
        for label in self.ax.get_yticklabels() + self.ax.get_xticklabels():
            label.set_bbox(dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.85, edgecolor='none'))
            label.set_color('black')
            label.set_fontsize(8)

    def update_plot(self, ships, running, simulation_time):
        self.ships = ships
        self.ax.clear()
        
        x_min, x_max, y_min, y_max = self._get_view_limits()
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        self.ax.set_aspect('equal')
        
        # Применяем стиль меток после clear()
        self._apply_tick_style()
        
        self.ax.grid(True, alpha=0.5)
        
        for ship in ships:
            track_x = []
            track_y = []
            for hx, hy in zip(ship.history_x, ship.history_y):
                dist = np.sqrt((hx - ship.x) ** 2 + (hy - ship.y) ** 2)
                if dist <= self.track_length_meters:
                    track_x.append(hx)
                    track_y.append(hy)
            if len(track_x) > 1:
                self.ax.plot(track_x, track_y, color=ship.color,
                             alpha=0.4, linewidth=1.5, linestyle='--', zorder=2)
            
            vertices = ship.get_pentagon_vertices()
            pentagon = Polygon(vertices, closed=True, facecolor='black',
                               edgecolor='black', linewidth=1.5, alpha=0.95, zorder=5)
            self.ax.add_patch(pentagon)
            
            if ship.llm_controlled:
                rect = Rectangle((ship.x - 60, ship.y - 60), 120, 120,
                                 fill=False, edgecolor='gold', linewidth=3, zorder=4)
                self.ax.add_patch(rect)
            
            self.draw_velocity_vector(ship)
            
            font_size = max(7, min(10, 1000 / self.view_scale * 10))
            llm_text = ship.get_llm_status_text()
            self.ax.text(ship.x + 80, ship.y + 80,
                         f"{ship.name}\n{ship.get_heading_deg():.0f}°\n"
                         f"{ship.u:.1f} m/s\n"
                         f"R:{ship.rudder_cmd:.0f}° RPM:{ship.rpm_cmd:.0f}%{llm_text}",
                         fontsize=font_size, ha='left', va='bottom',
                         bbox=dict(boxstyle='round', facecolor='white',
                                   alpha=0.8, edgecolor='gray'))
        
        self.draw()