"""
Q-Nav Quantum Navigation Module
Quantum-inspired backup navigation system
"""

import numpy as np
import time
from typing import List, Tuple


class QuantumGyroscope:
    """
    Simulates quantum gyroscope
    Uses quantum mechanics for ultra-precise rotation measurement
    """
    
    def __init__(self, precision_factor=100):
        """
        Initialize quantum gyroscope
        
        Args:
            precision_factor: How much better than classical (100x default)
        """
        self.precision_factor = precision_factor
        
        # Quantum sensors have MUCH lower noise/drift
        self.noise_level = 1e-5  # 100x better than classical
        self.drift_rate = 1e-6   # Minimal drift
        
        # Quantum state
        self.quantum_phase = np.zeros(3)  # Roll, pitch, yaw
        self.coherence_time = 1000.0  # How long quantum state stays stable
        self.last_measurement = time.time()
        
        print(f"⚛️  Quantum Gyroscope initialized ({precision_factor}x precision)")
    
    def measure_rotation(self):
        """
        Measure angular velocity using quantum interference
        
        Returns:
            3D rotation vector (rad/s)
        """
        current_time = time.time()
        dt = current_time - self.last_measurement
        
        # Quantum measurement (ultra-low noise)
        base_rotation = np.array([0., 0., 0.])
        quantum_noise = np.random.normal(0, self.noise_level, 3)
        drift = np.random.normal(0, self.drift_rate, 3) * dt
        
        rotation = base_rotation + quantum_noise + drift
        
        # Update quantum phase
        self.quantum_phase += rotation * dt
        
        self.last_measurement = current_time
        
        return rotation
    
    def get_orientation(self):
        """
        Get accumulated orientation
        
        Returns:
            3D orientation (roll, pitch, yaw) in radians
        """
        return self.quantum_phase.copy()
    
    def recalibrate(self):
        """Quantum recalibration"""
        self.quantum_phase = np.zeros(3)
        print("⚛️  Quantum gyroscope recalibrated")


class QuantumNavigator:
    """
    Main quantum navigation system
    Backup navigation when classical sensors fail
    """
    
    def __init__(self):
        self.q_gyro = QuantumGyroscope(precision_factor=100)
        
        # Navigation state
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)
        
        # System status
        self.quantum_active = False
        self.classical_active = True
        
        # Performance tracking
        self.activation_count = 0
        self.trajectory_history = []
        
        print("⚛️  Quantum Navigation System initialized")
    
    def activate_quantum_mode(self):
        """Activate quantum backup"""
        if not self.quantum_active:
            self.quantum_active = True
            self.classical_active = False
            self.activation_count += 1
            
            print("\n" + "="*50)
            print("⚛️  QUANTUM BACKUP ACTIVATED")
            print("    Classical sensors offline")
            print("    Switching to quantum navigation")
            print("="*50 + "\n")
    
    def deactivate_quantum_mode(self):
        """Return to classical navigation"""
        if self.quantum_active:
            self.quantum_active = False
            self.classical_active = True
            
            print("\n" + "="*50)
            print("✓ CLASSICAL SENSORS RESTORED")
            print("    Quantum backup deactivated")
            print("="*50 + "\n")
    
    def check_sensor_health(self, sensor_quality):
        """
        Monitor sensor health and activate quantum if needed
        
        Args:
            sensor_quality: 0-1 (0=failed, 1=perfect)
            
        Returns:
            True if quantum is active
        """
        threshold = 0.5
        
        if sensor_quality < threshold and not self.quantum_active:
            self.activate_quantum_mode()
            return True
        elif sensor_quality >= threshold and self.quantum_active:
            self.deactivate_quantum_mode()
            return False
        
        return self.quantum_active
    
    def quantum_position_estimate(self, dt):
        """
        Estimate position using quantum sensors
        
        Args:
            dt: Time step (seconds)
            
        Returns:
            Estimated position
        """
        # Get ultra-precise rotation
        rotation = self.q_gyro.measure_rotation()
        
        # Update orientation
        self.orientation += rotation * dt
        
        # Dead reckoning with quantum precision
        self.position += self.velocity * dt
        
        # Quantum uncertainty (very small!)
        uncertainty = np.sqrt(dt) * 1e-9
        position_noise = np.random.normal(0, uncertainty, 3)
        
        estimated_pos = self.position + position_noise
        
        return estimated_pos
    
    def quantum_path_optimization(self, current_pos, target_pos, obstacles):
        """
        Use quantum-inspired optimization for path planning
        
        simulates quantum annealing/superposition
        
        Args:
            current_pos: Current position
            target_pos: Target position  
            obstacles: List of obstacle positions
            
        Returns:
            Optimal direction vector
        """
        # Quantum superposition: evaluate multiple paths simultaneously
        num_quantum_states = 100
        
        best_direction = None
        best_score = -np.inf
        
        for _ in range(num_quantum_states):
            # Generate random direction (quantum superposition)
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            # Evaluate this quantum state
            score = self._evaluate_path_quality(
                current_pos, direction, target_pos, obstacles
            )
            
            if score > best_score:
                best_score = score
                best_direction = direction
        
        return best_direction
    
    def _evaluate_path_quality(self, pos, direction, target, obstacles):
        """
        Evaluate path quality score
        
        Returns:
            Quality score (higher = better)
        """
        # Predict next position
        next_pos = pos + direction * 1.0
        
        # Distance to target (want to minimize)
        dist_to_target = np.linalg.norm(target - next_pos)
        target_score = -dist_to_target
        
        # Distance to obstacles (want to maximize)
        obstacle_score = 0
        for obs in obstacles:
            obs_pos = np.array(obs)
            dist_to_obs = np.linalg.norm(obs_pos - next_pos)
            
            if dist_to_obs < 2.0:
                # Too close - heavy penalty
                obstacle_score -= 100 / (dist_to_obs + 0.1)
            else:
                # Good - reward
                obstacle_score += dist_to_obs
        
        return target_score + obstacle_score
    
    def quantum_error_correction(self, state):
        """
        Apply quantum error correction to navigation state
        
        Uses trajectory history to correct drift
        
        Args:
            state: Current state estimate
            
        Returns:
            Corrected state
        """
        # Store in history
        self.trajectory_history.append(state.copy())
        
        # Keep limited history
        if len(self.trajectory_history) > 100:
            self.trajectory_history.pop(0)
        
        # Apply quantum error correction
        if len(self.trajectory_history) > 10:
            recent = np.array(self.trajectory_history[-10:])
            
            # Weighted average (more recent = higher weight)
            weights = np.exp(np.linspace(-1, 0, 10))
            weights /= weights.sum()
            
            corrected = np.average(recent, axis=0, weights=weights)
            
            # Small correction
            correction_factor = 0.1
            state = (1 - correction_factor) * state + correction_factor * corrected
        
        return state
    
    def navigate(self, dt, target_position, obstacles=None):
        """
        Main navigation function
        
        Args:
            dt: Time step (seconds)
            target_position: Target to reach
            obstacles: List of obstacle positions
            
        Returns:
            (position, velocity, status)
        """
        if obstacles is None:
            obstacles = []
        
        status = ""
        
        if self.quantum_active:
            # QUANTUM NAVIGATION MODE
            
            # Ultra-precise position estimate
            self.position = self.quantum_position_estimate(dt)
            
            # Quantum path optimization
            optimal_direction = self.quantum_path_optimization(
                self.position, target_position, obstacles
            )
            
            # Update velocity
            desired_vel = optimal_direction * 1.0
            self.velocity = 0.9 * self.velocity + 0.1 * desired_vel
            
            # Quantum error correction
            state = np.concatenate([self.position, self.velocity])
            corrected_state = self.quantum_error_correction(state)
            
            self.position = corrected_state[:3]
            self.velocity = corrected_state[3:6]
            
            status = "⚛️  QUANTUM MODE"
        
        else:
            # CLASSICAL NAVIGATION MODE
            self.position += self.velocity * dt
            status = "CLASSICAL MODE"
        
        return self.position.copy(), self.velocity.copy(), status


def quantum_collision_prediction(position, velocity, obstacles, obstacle_velocities=None, time_horizon=5.0):
    """
    Quantum-enhanced collision prediction
    
    Uses quantum superposition to check multiple future trajectories
    
    Args:
        position: Current position
        velocity: Current velocity
        obstacles: List of obstacle positions
        obstacle_velocities: List of obstacle velocities (optional)
        time_horizon: Prediction time (seconds)
        
    Returns:
        (will_collide, time_to_collision, collision_point)
    """
    if not obstacles:
        return False, float('inf'), None
    
    if obstacle_velocities is None:
        obstacle_velocities = [np.zeros(3) for _ in obstacles]
    
    dt = 0.01
    steps = int(time_horizon / dt)
    
    for step in range(steps):
        t = step * dt
        
        # Future position
        future_pos = position + velocity * t
        
        # Check each obstacle
        for obs_pos, obs_vel in zip(obstacles, obstacle_velocities):
            future_obs = obs_pos + obs_vel * t
            
            distance = np.linalg.norm(future_pos - future_obs)
            
            # Quantum uncertainty in prediction
            uncertainty = np.sqrt(t) * 0.1
            
            if distance < (2.0 + uncertainty):
                return True, t, future_obs
    
    return False, float('inf'), None


# ============= DEMO =============
if __name__ == "__main__":
    print("\n" + "="*50)
    print("⚛️  Q-NAV QUANTUM NAVIGATION DEMO")
    print("="*50 + "\n")
    
    # Initialize quantum system
    qnav = QuantumNavigator()
    
    # Set target
    target = np.array([50, 0, 0])
    print(f"Target: {target}\n")
    
    # Create obstacles
    obstacles = [
        np.array([20, 5, 0]),
        np.array([30, -3, 0]),
        np.array([40, 2, 0])
    ]
    print(f"Obstacles: {len(obstacles)}\n")
    
    # Simulate normal navigation
    print("Phase 1: Normal navigation...")
    for i in range(10):
        pos, vel, status = qnav.navigate(
            dt=0.1,
            target_position=target,
            obstacles=obstacles
        )
        if i % 3 == 0:
            print(f"  Step {i}: {status} | Pos={pos}")
        time.sleep(0.05)
    
    # Simulate sensor failure
    print("\n⚠️  SENSOR FAILURE DETECTED!")
    print("Activating quantum backup...\n")
    qnav.check_sensor_health(sensor_quality=0.2)
    
    # Navigate with quantum backup
    print("Phase 2: Quantum navigation...")
    for i in range(15):
        pos, vel, status = qnav.navigate(
            dt=0.1,
            target_position=target,
            obstacles=obstacles
        )
        if i % 3 == 0:
            print(f"  Step {i}: {status} | Pos={pos}")
        time.sleep(0.05)
    
    # Test collision prediction
    print("\nTesting quantum collision prediction...")
    test_pos = np.array([15, 0, 0])
    test_vel = np.array([1, 0, 0])
    
    will_collide, time_to, point = quantum_collision_prediction(
        test_pos, test_vel, obstacles
    )
    
    if will_collide:
        print(f"  ⚠️  Collision predicted in {time_to:.2f}s")
        print(f"      Impact point: {point}")
    else:
        print(f"  ✓ No collision detected")
    
    # Final stats
    print("\n" + "="*50)
    print("QUANTUM SYSTEM STATS")
    print("="*50)
    print(f"Quantum activations: {qnav.activation_count}")
    print(f"Final position: {qnav.position}")
    print(f"Trajectory points: {len(qnav.trajectory_history)}")
    print("="*50 + "\n")
    
    print("✓ Quantum navigation demo complete! ⚛️\n")