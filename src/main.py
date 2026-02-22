"""
Q-Nav Main System Integration
Combines all modules into complete autonomous navigation system
"""

import cv2
import numpy as np
import time
from pathlib import Path

# Import all Q-Nav modules
from detection import DebrisDetector
from sensors import SensorFusion
from navigation import Navigator, CollisionPredictor
from quantum import QuantumNavigator, quantum_collision_prediction


class QNavSystem:
    """
    Complete Q-Nav Spacecraft Navigation System
    Integrates: Detection + Sensors + Navigation + Quantum Backup
    """
    
    def __init__(self):
        print("\n" + "="*60)
        print("🚀 INITIALIZING Q-NAV SYSTEM")
        print("="*60 + "\n")
        
        # Initialize all subsystems
        print("Loading subsystems...")
        self.detector = DebrisDetector()
        self.sensors = SensorFusion()
        self.navigator = Navigator(safe_distance=2.0)
        self.quantum_nav = QuantumNavigator()
        self.collision_predictor = CollisionPredictor(time_horizon=5.0)
        
        # System state
        self.target_position = np.array([50.0, 0.0, 0.0])
        self.detected_obstacles = []
        
        # Performance metrics
        self.start_time = time.time()
        self.frames_processed = 0
        self.collisions_avoided = 0
        self.quantum_activations = 0
        
        # Mode change tracking for visual feedback
        self.last_mode = "classical"
        self.mode_change_frame = 0
        
        # Set navigator target
        self.navigator.set_target(self.target_position)
        
        print("\n" + "="*60)
        print("✓ Q-NAV SYSTEM READY")
        print("="*60 + "\n")
    
    def process_camera(self, frame):
        """
        Process camera frame for obstacle detection
        
        Args:
            frame: Camera image
            
        Returns:
            detections, annotated_frame, risk_level, risk_score
        """
        # Detect obstacles
        detections, annotated = self.detector.detect(frame)
        
        # Calculate collision risk
        risk_level, risk_score = self.detector.calculate_risk(
            detections, frame.shape
        )
        
        # Convert 2D detections to 3D obstacle positions (simplified)
        self.detected_obstacles = []
        height, width = frame.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Estimate 3D position
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            size = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Rough distance (larger = closer)
            distance = max(50.0 / (size + 1), 5.0)
            
            # 3D position
            obstacle_pos = np.array([
                distance,
                (center_x - width/2) * 0.02,
                (center_y - height/2) * 0.02
            ])
            
            self.detected_obstacles.append(obstacle_pos)
        
        return detections, annotated, risk_level, risk_score
    
    def update_navigation(self, dt):
        """
        Update navigation system
        
        Args:
            dt: Time step (seconds)
            
        Returns:
            position, velocity, status, collision_warning
        """
        # Update sensor fusion
        sensor_pos, sensor_vel, sensor_orient = self.sensors.update(
            dt, self.detected_obstacles
        )
        
        # Calculate sensor health (simulate degradation)
        sensor_health = np.random.uniform(0.7, 1.0)
        
        # Check if quantum backup needed
        quantum_active = self.quantum_nav.check_sensor_health(sensor_health)
        
        if quantum_active:
            self.quantum_activations += 1
            
            # Use quantum navigation
            pos, vel, q_status = self.quantum_nav.navigate(
                dt, self.target_position, self.detected_obstacles
            )
            status = q_status
        else:
            # Use classical navigation
            self.navigator.update_state(sensor_pos, sensor_vel)
            pos, vel, nav_status = self.navigator.navigate_step(
                self.detected_obstacles, dt
            )
            status = nav_status
        
        # Predict collisions
        will_collide, time_to, collision_point = quantum_collision_prediction(
            pos, vel,
            self.detected_obstacles,
            time_horizon=5.0
        )
        
        if will_collide:
            self.collisions_avoided += 1
            collision_warning = f"⚠️  COLLISION in {time_to:.1f}s"
        else:
            collision_warning = "✓ Safe"
        
        return pos, vel, status, collision_warning
    
    def create_display(self, frame, annotated, pos, vel, status, 
                       collision_warning, risk_level, risk_score):
        """
        Create complete system visualization
        
        Args:
            frame: Original frame
            annotated: Detection annotated frame
            pos: Current position
            vel: Current velocity
            status: Navigation status
            collision_warning: Collision warning text
            risk_level: Risk level
            risk_score: Risk score
            
        Returns:
            Display frame
        """
        # ADD BIG MODE INDICATOR ON CAMERA FEED
        current_mode = "quantum" if self.quantum_nav.quantum_active else "classical"
        
        # Detect mode change
        if current_mode != self.last_mode:
            self.mode_change_frame = self.frames_processed
            self.last_mode = current_mode
        
        # Flash effect for 30 frames after mode change
        is_flashing = (self.frames_processed - self.mode_change_frame) < 30
        flash = (self.frames_processed % 10 < 5) if is_flashing else False
        
        if self.quantum_nav.quantum_active:
            # QUANTUM MODE - Purple banner at top
            color = (255, 255, 255) if flash else (128, 0, 128)
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 80), 
                         color if flash else (128, 0, 128), -1)
            cv2.putText(annotated, "QUANTUM MODE ACTIVE", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 
                       (128, 0, 128) if flash else (255, 255, 255), 4)
        else:
            # CLASSICAL MODE - Green banner at top
            color = (255, 255, 255) if flash else (0, 150, 0)
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 70), 
                         color if flash else (0, 150, 0), -1)
            cv2.putText(annotated, "CLASSICAL MODE", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
                       (0, 150, 0) if flash else (255, 255, 255), 3)
        
        # Create info panel
        height, width = annotated.shape[:2]
        panel = np.zeros((height, 400, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)  # Dark background
        
        y = 30
        line_h = 30
        
        # Title
        cv2.putText(panel, "Q-NAV SYSTEM", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += line_h * 1.5
        
        # Collision Risk
        risk_colors = {
            'SAFE': (0, 255, 0),
            'LOW': (0, 255, 255),
            'MEDIUM': (0, 165, 255),
            'HIGH': (0, 0, 255)
        }
        color = risk_colors.get(risk_level, (255, 255, 255))
        
        cv2.putText(panel, "COLLISION RISK:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        cv2.putText(panel, f"{risk_level} ({risk_score}/100)", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += line_h * 1.2
        
        # Position
        cv2.putText(panel, "POSITION:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        cv2.putText(panel, f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y += line_h * 1.2
        
        # Velocity
        speed = np.linalg.norm(vel)
        cv2.putText(panel, "VELOCITY:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        cv2.putText(panel, f"{speed:.2f} m/s", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y += line_h * 1.2
        
        # Target Distance
        distance = np.linalg.norm(self.target_position - pos)
        cv2.putText(panel, "TARGET DISTANCE:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        cv2.putText(panel, f"{distance:.1f} m", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y += line_h * 1.2
        
        # Navigation Status
        cv2.putText(panel, "STATUS:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        
        # Color code quantum mode
        if "QUANTUM" in status:
            status_color = (255, 0, 255)  # Magenta for quantum
        else:
            status_color = (100, 255, 100)  # Green for classical
        
        # Wrap status text if too long
        if len(status) > 25:
            status = status[:25] + "..."
        
        cv2.putText(panel, status, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y += line_h * 1.2
        
        # Collision Warning
        cv2.putText(panel, "COLLISION:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        
        warning_color = (0, 0, 255) if "⚠️" in collision_warning else (0, 255, 0)
        cv2.putText(panel, collision_warning, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, warning_color, 1)
        y += line_h * 1.5
        
        # Divider
        cv2.line(panel, (10, y), (390, y), (100, 100, 100), 1)
        y += line_h
        
        # Quantum Mode Indicator
        if self.quantum_nav.quantum_active:
            cv2.putText(panel, "⚛️  QUANTUM MODE", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            cv2.putText(panel, "CLASSICAL MODE", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 1)
        y += line_h * 1.5
        
        # Statistics
        cv2.putText(panel, "STATISTICS:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        
        stats = [
            f"Frames: {self.frames_processed}",
            f"Obstacles: {len(self.detected_obstacles)}",
            f"Collisions Avoided: {self.collisions_avoided}",
            f"Quantum Activations: {self.quantum_activations}"
        ]
        
        for stat in stats:
            cv2.putText(panel, stat, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            y += 22
        
        # Combine display
        display = np.hstack([annotated, panel])
        
        return display
    
    def run_live(self, camera_id=0):
        """
        Run Q-Nav with live camera
        
        Args:
            camera_id: Camera device ID
        """
        print("🎥 Starting Q-Nav live operation...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Simulate sensor failure")
        print("  'r' - Restore sensors")
        print("  't' - Change target\n")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Cannot access camera!")
            return
        
        last_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate time step
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                # Process frame
                detections, annotated, risk_level, risk_score = \
                    self.process_camera(frame)
                
                # Update navigation
                pos, vel, status, collision_warning = \
                    self.update_navigation(dt)
                
                # Create display
                display = self.create_display(
                    frame, annotated, pos, vel, status,
                    collision_warning, risk_level, risk_score
                )
                
                # Show
                cv2.imshow('Q-Nav Complete System', display)
                
                self.frames_processed += 1
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting Q-Nav...")
                    break
                elif key == ord('s'):
                    print("\n" + "="*50)
                    print("⚠️  SIMULATING SENSOR FAILURE")
                    print("="*50)
                    self.quantum_nav.check_sensor_health(0.1)
                    print("Watch the display - should show QUANTUM MODE!\n")
                elif key == ord('r'):
                    print("\n" + "="*50)
                    print("✓ RESTORING SENSORS")
                    print("="*50)
                    self.quantum_nav.check_sensor_health(1.0)
                    print("Watch the display - should show CLASSICAL MODE!\n")
                elif key == ord('t'):
                    # Random new target
                    old_target = self.target_position.copy()
                    self.target_position = np.random.uniform(-50, 50, 3)
                    self.navigator.set_target(self.target_position)
                    print("\n" + "="*50)
                    print("🎯 TARGET CHANGED")
                    print("="*50)
                    print(f"Old target: {old_target}")
                    print(f"New target: {self.target_position}")
                    print("Watch TARGET DISTANCE change on display!\n")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_final_stats()
    
    def print_final_stats(self):
        """Print final mission statistics"""
        runtime = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("Q-NAV MISSION COMPLETE")
        print("="*60)
        print(f"Runtime: {runtime:.2f}s")
        print(f"Frames processed: {self.frames_processed}")
        print(f"Processing rate: {self.frames_processed/runtime:.2f} fps")
        print(f"Obstacles detected: {len(self.detected_obstacles)}")
        print(f"Collisions avoided: {self.collisions_avoided}")
        print(f"Quantum activations: {self.quantum_activations}")
        print(f"Final position: {self.navigator.position}")
        print(f"Distance to target: {self.navigator.calculate_distance_to_target():.2f}m")
        print("="*60 + "\n")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("    🚀 Q-NAV: QUANTUM NAVIGATION SYSTEM 🚀")
    print("         Autonomous Spacecraft Navigation")
    print("="*60)
    
    # Initialize complete system
    qnav = QNavSystem()
    
    # Run live
    qnav.run_live(camera_id=0)


if __name__ == "__main__":
    main()