import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from collections import deque

from detection import DebrisDetector
from sensors import SensorFusion
from navigation import Navigator
from quantum import QuantumNavigator

VIDEO_W = 640
VIDEO_H = 480

class QNavGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Q-NAV: Quantum Navigation System")
        self.root.configure(bg='#0a0a0f')
        self.root.state('zoomed')

        self.detector    = None
        self.sensors     = None
        self.navigator   = None
        self.quantum_nav = None

        self.running         = False
        self.camera          = None
        self.target_position = np.array([50.0, 0.0, 0.0])
        self.sensor_health   = 1.0
        self.auto_degrade    = False
        self._frame_count    = 0
        self._flash_job      = None
        self._flash_count    = 0
        self._flash_mode     = None

        self.time_history   = deque(maxlen=100)
        self.health_history = deque(maxlen=100)
        self.start_time     = time.time()

        self.create_widgets()

    def create_widgets(self):

        header = tk.Frame(self.root, bg='#12121a', height=55)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="Q-NAV", font=('Courier New', 22, 'bold'),
                 bg='#12121a', fg='#00ff88').pack(side=tk.LEFT, padx=20, pady=8)
        tk.Label(header, text="QUANTUM NAVIGATION SYSTEM", font=('Courier New', 10),
                 bg='#12121a', fg='#445566').pack(side=tk.LEFT, pady=8)
        self.mode_label = tk.Label(header, text="● CLASSICAL MODE",
                                   font=('Courier New', 13, 'bold'),
                                   bg='#12121a', fg='#00ff00')
        self.mode_label.pack(side=tk.RIGHT, padx=20)

        self.alert_banner = tk.Frame(self.root, bg='#1a0a2e', height=0)
        self.alert_banner.pack(fill=tk.X)
        self.alert_banner.pack_propagate(False)
        self.alert_label = tk.Label(self.alert_banner, text="",
                                    font=('Courier New', 13, 'bold'),
                                    bg='#1a0a2e', fg='#ff00ff')
        self.alert_label.place(relx=0.5, rely=0.5, anchor='center')

        content = tk.Frame(self.root, bg='#0a0a0f')
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # LEFT — fixed width, fixed video size, nothing expands
        left = tk.Frame(content, bg='#0a0a0f', width=VIDEO_W + 20)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        self.video_border = tk.Frame(left, bg='#00ff88', bd=2,
                                     width=VIDEO_W + 4, height=VIDEO_H + 4)
        self.video_border.pack(pady=(10, 4))
        self.video_border.pack_propagate(False)

        self.video_label = tk.Label(self.video_border, bg='#000000',
                                    width=VIDEO_W, height=VIDEO_H)
        self.video_label.place(x=0, y=0, width=VIDEO_W, height=VIDEO_H)

        self.video_mode_badge = tk.Label(left, text="● CLASSICAL MODE",
                                         font=('Courier New', 11, 'bold'),
                                         bg='#0a0a0f', fg='#00ff88')
        self.video_mode_badge.pack(pady=(0, 4))

        # RIGHT — fixed width, all controls + stats
        right = tk.Frame(content, bg='#12121a')
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # TOP ROW: buttons on left | activation log on right
        top_row = tk.Frame(right, bg='#12121a')
        top_row.pack(fill=tk.BOTH, expand=True, padx=6, pady=(8,4))

        # Buttons column
        btn_col = tk.Frame(top_row, bg='#12121a')
        btn_col.pack(side=tk.LEFT, fill=tk.Y, padx=(8,4))

        tk.Label(btn_col, text="CONTROLS", font=('Courier New', 11, 'bold'),
                 bg='#12121a', fg='#00ff88').pack(anchor=tk.W, pady=(4,6))

        btn = dict(font=('Courier New', 10, 'bold'), relief=tk.FLAT,
                   bd=0, padx=12, pady=7, cursor='hand2', width=18)

        self.start_btn = tk.Button(btn_col, text="▶  START NAVIGATION",
                                   command=self.start_navigation,
                                   bg='#00ff88', fg='#000000', **btn)
        self.start_btn.pack(pady=3, anchor=tk.W)

        self.stop_btn = tk.Button(btn_col, text="■  STOP",
                                  command=self.stop_navigation,
                                  bg='#ff4444', fg='#ffffff',
                                  state=tk.DISABLED, **btn)
        self.stop_btn.pack(pady=3, anchor=tk.W)

        self.quantum_btn = tk.Button(btn_col, text="⚛  ACTIVATE QUANTUM",
                                     command=self.activate_quantum,
                                     bg='#6600cc', fg='#ffffff', **btn)
        self.quantum_btn.pack(pady=3, anchor=tk.W)

        self.restore_btn = tk.Button(btn_col, text="✓  RESTORE SENSORS",
                                     command=self.restore_sensors,
                                     bg='#004499', fg='#ffffff', **btn)
        self.restore_btn.pack(pady=3, anchor=tk.W)

        self.auto_var = tk.BooleanVar()
        tk.Checkbutton(btn_col, text="Auto Degradation Mode",
                       variable=self.auto_var, command=self.toggle_auto,
                       bg='#12121a', fg='#888888', selectcolor='#000000',
                       activebackground='#12121a',
                       font=('Courier New', 10)).pack(anchor=tk.W, pady=(6,2))

        # Vertical divider
        tk.Frame(top_row, bg='#333355', width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Activation log column
        log_col = tk.Frame(top_row, bg='#12121a')
        log_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4,8))

        tk.Label(log_col, text="ACTIVATION LOG", font=('Courier New', 11, 'bold'),
                 bg='#12121a', fg='#00ff88').pack(anchor=tk.W, pady=(4,6))
        self.log_box = tk.Text(log_col, bg='#0a0a0f', fg='#00ff88',
                               font=('Courier New', 9), relief=tk.FLAT,
                               state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True)

        # BOTTOM: system status + graph side by side
        tk.Frame(right, bg='#333355', height=1).pack(fill=tk.X, padx=14, pady=4)

        bottom_row = tk.Frame(right, bg='#12121a')
        bottom_row.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,6))

        # Stats column
        stats_col = tk.Frame(bottom_row, bg='#12121a')
        stats_col.pack(side=tk.LEFT, fill=tk.Y, padx=(8,4))

        tk.Label(stats_col, text="SYSTEM STATUS", font=('Courier New', 11, 'bold'),
                 bg='#12121a', fg='#00ff88').pack(anchor=tk.W, pady=(4,6))

        hrow = tk.Frame(stats_col, bg='#12121a')
        hrow.pack(anchor=tk.W, pady=3)
        tk.Label(hrow, text="SENSOR HEALTH", font=('Courier New', 9),
                 bg='#12121a', fg='#445566', width=14, anchor=tk.W).pack(side=tk.LEFT)
        self.health_bar = tk.Canvas(hrow, width=120, height=18, bg='#1a1a2e',
                                    highlightthickness=1, highlightbackground='#333355')
        self.health_bar.pack(side=tk.LEFT, padx=4)
        self.health_text = tk.Label(hrow, text="100%", font=('Courier New', 9, 'bold'),
                                    bg='#12121a', fg='#00ff00', width=5)
        self.health_text.pack(side=tk.LEFT)

        self.position_label  = self._stat(stats_col, "POSITION",       "0.0  0.0  0.0")
        self.velocity_label  = self._stat(stats_col, "VELOCITY",       "0.00 m/s")
        self.distance_label  = self._stat(stats_col, "TARGET DIST",    "50.0 m")
        self.risk_label      = self._stat(stats_col, "COLLISION RISK", "SAFE")
        self.obstacles_label = self._stat(stats_col, "OBSTACLES",      "0 detected")

        # Vertical divider
        tk.Frame(bottom_row, bg='#333355', width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Graph column
        graph_col = tk.Frame(bottom_row, bg='#12121a')
        graph_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4,8))

        tk.Label(graph_col, text="SENSOR HEALTH OVER TIME", font=('Courier New', 9),
                 bg='#12121a', fg='#445566').pack(anchor=tk.W, pady=(4,0))
        self.fig = Figure(facecolor='#0a0a0f')
        self.ax  = self.fig.add_subplot(111, facecolor='#12121a')
        self._style_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, graph_col)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=4)

        bar = tk.Frame(self.root, bg='#12121a', height=24)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)
        self.status_label = tk.Label(bar,
            text="● READY  —  Click  ▶ START NAVIGATION  to begin",
            font=('Courier New', 9), bg='#12121a', fg='#00ff88', anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=12)

    def _stat(self, parent, title, value):
        row = tk.Frame(parent, bg='#12121a')
        row.pack(anchor=tk.W, pady=2)
        tk.Label(row, text=title, font=('Courier New', 9), bg='#12121a',
                 fg='#445566', width=14, anchor=tk.W).pack(side=tk.LEFT)
        lbl = tk.Label(row, text=value, font=('Courier New', 9, 'bold'),
                       bg='#12121a', fg='#00ff88')
        lbl.pack(side=tk.LEFT, padx=4)
        return lbl

    def _style_axes(self):
        self.ax.set_xlabel('Time (s)', color='#445566', fontsize=7)
        self.ax.set_ylabel('Health %', color='#445566', fontsize=7)
        self.ax.tick_params(colors='#445566', labelsize=6)
        for sp in self.ax.spines.values():
            sp.set_color('#333355')
        self.fig.tight_layout(pad=0.8)

    def start_navigation(self):
        if not self.running:
            self.start_btn.config(state=tk.DISABLED)
            self.status_label.config(
                text="● LOADING...  YOLO model initialising, please wait",
                fg='#ffff00')
            self.root.update()
            threading.Thread(target=self._init_and_run, daemon=True).start()

    def _init_and_run(self):
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.root.after(0, lambda: self.status_label.config(
                    text="● ERROR: Camera not found", fg='#ff4444'))
                self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
                return

            self.running = True
            self.root.after(0, lambda: self.status_label.config(
                text="● LOADING YOLO... camera ready", fg='#ffff00'))
            threading.Thread(target=self._raw_camera_preview, daemon=True).start()

            self.sensors     = SensorFusion()
            self.navigator   = Navigator()
            self.quantum_nav = QuantumNavigator()
            self.detector    = DebrisDetector()
            self.navigator.set_target(self.target_position)

            self.running = False
            time.sleep(0.1)
            self.running = True
            self.root.after(0, lambda: self.stop_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_label.config(
                text="● RUNNING", fg='#00ff88'))
            self.root.after(0, lambda: self._log("System started — classical mode"))
            self.update_loop()

        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(
                text=f"● ERROR: {str(e)[:50]}", fg='#ff4444'))
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

    def _raw_camera_preview(self):
        while self.running and self.detector is None:
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (VIDEO_W, VIDEO_H))
            img   = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.root.after(0, self._set_video_frame, imgtk)
            time.sleep(0.033)

    def _set_video_frame(self, imgtk):
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def stop_navigation(self):
        self.running = False
        if self.camera:
            self.camera.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="● STOPPED", fg='#ff4444')
        self._log("System stopped")

    def activate_quantum(self):
        self.sensor_health = 0.1
        if self.quantum_nav:
            self.quantum_nav.check_sensor_health(self.sensor_health)
        self._show_quantum_activated()
        self._log("⚛  QUANTUM BACKUP ACTIVATED — sensor failure simulated")
        self.status_label.config(text="● QUANTUM MODE ACTIVE", fg='#ff00ff')

    def restore_sensors(self):
        self.sensor_health = 1.0
        if self.quantum_nav:
            self.quantum_nav.check_sensor_health(self.sensor_health)
        self._show_sensors_restored()
        self._log("✓  Classical sensors restored")
        self.status_label.config(text="● CLASSICAL MODE", fg='#00ff88')

    def toggle_auto(self):
        self.auto_degrade = self.auto_var.get()
        if self.auto_degrade:
            self._log("Auto degradation ON — quantum activates in ~15s")
            self.status_label.config(text="● AUTO DEGRADE ON", fg='#ffff00')
        else:
            self._log("Auto degradation disabled")

    def _show_quantum_activated(self):
        self._flash_mode  = 'quantum'
        self._flash_count = 0
        self.alert_banner.config(bg='#2a0050', height=46)
        self.alert_label.config(
            text="⚛   QUANTUM BACKUP ACTIVATED  —  CLASSICAL SENSORS OFFLINE   ⚛",
            bg='#2a0050', fg='#ff00ff')
        self.mode_label.config(text="⚛ QUANTUM MODE", fg='#ff00ff')
        self.video_mode_badge.config(text="⚛  QUANTUM MODE ACTIVE", fg='#ff00ff')
        self._flash()

    def _show_sensors_restored(self):
        self._flash_mode  = 'classical'
        self._flash_count = 0
        self.alert_banner.config(bg='#002a10', height=46)
        self.alert_label.config(
            text="✓   CLASSICAL SENSORS RESTORED  —  QUANTUM BACKUP DEACTIVATED   ✓",
            bg='#002a10', fg='#00ff88')
        self.mode_label.config(text="● CLASSICAL MODE", fg='#00ff00')
        self.video_mode_badge.config(text="● CLASSICAL MODE", fg='#00ff88')
        self._flash()

    def _flash(self):
        if self._flash_job:
            self.root.after_cancel(self._flash_job)
        if self._flash_count >= 8:
            return
        bright = (self._flash_count % 2 == 0)
        if self._flash_mode == 'quantum':
            bg = '#3d007a' if bright else '#1a0040'
            fg = '#ff88ff' if bright else '#aa00cc'
        else:
            bg = '#004d1f' if bright else '#001a0a'
            fg = '#00ff88' if bright else '#007744'
        self.alert_banner.config(bg=bg)
        self.alert_label.config(bg=bg, fg=fg)
        self._flash_count += 1
        self._flash_job = self.root.after(180, self._flash)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, f"[{ts}]  {msg}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def update_loop(self):
        last_time = time.time()
        while self.running:
            now = time.time()
            dt  = now - last_time
            last_time = now

            ret, frame = self.camera.read()
            if not ret:
                continue

            detections, annotated = self.detector.detect(frame)
            risk_level, _         = self.detector.calculate_risk(detections, frame.shape)
            pos, vel, _           = self.sensors.update(dt, [])

            if self.auto_degrade:
                self.sensor_health = max(0.0, self.sensor_health - 0.0333 * dt)

            if self.quantum_nav:
                was  = self.quantum_nav.quantum_active
                self.quantum_nav.check_sensor_health(self.sensor_health)
                now_q = self.quantum_nav.quantum_active
                if self.auto_degrade and not was and now_q:
                    self.root.after(0, self._show_quantum_activated)
                    self.root.after(0, lambda: self._log(
                        "⚛  AUTO: quantum activated — health < 50%"))
                elif self.auto_degrade and was and not now_q:
                    self.root.after(0, self._show_sensors_restored)

            if self.navigator and self.quantum_nav:
                if self.quantum_nav.quantum_active:
                    pos, vel, _ = self.quantum_nav.navigate(dt, self.target_position, [])
                else:
                    self.navigator.update_state(pos, vel)
                    pos, vel, _ = self.navigator.navigate_step([], dt)

            elapsed = time.time() - self.start_time
            self.time_history.append(elapsed)
            self.health_history.append(self.sensor_health * 100)

            self.root.after(0, self.update_ui, annotated, pos, vel,
                            risk_level, len(detections))
            time.sleep(0.033)

    def update_ui(self, frame, pos, vel, risk_level, obstacle_count):
        frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (VIDEO_W, VIDEO_H))
        img   = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        is_quantum = self.quantum_nav and self.quantum_nav.quantum_active
        self.video_border.config(bg='#ff00ff' if is_quantum else '#00ff88')

        self.health_bar.delete("all")
        hcol = '#00ff00' if self.sensor_health > 0.5 else '#ff2200'
        self.health_bar.create_rectangle(
            0, 0, int(180 * self.sensor_health), 18, fill=hcol, outline='')
        self.health_text.config(text=f"{int(self.sensor_health*100)}%", fg=hcol)

        self.position_label.config(
            text=f"X:{pos[0]:.1f}  Y:{pos[1]:.1f}  Z:{pos[2]:.1f}")
        self.velocity_label.config(text=f"{np.linalg.norm(vel):.2f} m/s")
        self.distance_label.config(
            text=f"{np.linalg.norm(self.target_position - pos):.1f} m")
        rc = {'SAFE': '#00ff00', 'LOW': '#ffff00',
              'MEDIUM': '#ff8800', 'HIGH': '#ff2200'}
        self.risk_label.config(text=risk_level, fg=rc.get(risk_level, '#fff'))
        self.obstacles_label.config(text=f"{obstacle_count} detected")

        self._frame_count += 1
        if self._frame_count % 10 == 0 and len(self.time_history) > 1:
            self.ax.clear()
            times  = list(self.time_history)
            health = list(self.health_history)
            lc = '#ff2200' if self.sensor_health < 0.5 else '#00ff88'
            self.ax.plot(times, health, color=lc, linewidth=1.6)
            self.ax.axhline(y=50, color='#ff4444', linestyle='--',
                            alpha=0.5, linewidth=1)
            self.ax.fill_between(times, health, 50,
                                 where=[h < 50 for h in health],
                                 color='#ff000033', interpolate=True)
            self.ax.set_ylim(0, 105)
            self._style_axes()
            self.canvas.draw()

def main():
    root = tk.Tk()
    app  = QNavGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()