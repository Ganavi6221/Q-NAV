"""
Q-Nav Sensor Simulation Module
Simulates IMU, LiDAR, and Camera sensors
"""

import numpy as np
import time
from dataclasses import dataclass


@dataclass
class IMUData:
    """IMU sensor reading"""
    timestamp: float
    acceleration: np.ndarray   
    gyroscope: np.ndarray       
    temperature: float         


class IMUSensor:
    """
    Inertial Measurement Unit
    Measures: acceleration, rotation, orientation
    """
    
    def __init__(self, noise_level=0.01):
        """
        Initialize IMU
        
        Args:
            noise_level: How much random noise (higher = less accurate)
        """
        self.noise_level = noise_level
        self.drift = np.zeros(3)  
        self.start_time = time.time()
        
       
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3) 
        
        print(f"✓ IMU initialized (noise: {noise_level})")
    
    def read(self):
        """
        Read current sensor data
        
        Returns:
            IMUData with acceleration, gyro, temp
        """
        current_time = time.time() - self.start_time
        
        
        gravity = np.array([0., 0., -9.81])
        noise = np.random.normal(0, self.noise_level, 3)
        self.drift += np.random.normal(0, 0.001, 3)
        
        acceleration = gravity + noise + self.drift
        
       
        gyroscope = np.random.normal(0, self.noise_level * 0.1, 3)
        
        
        temperature = 20.0 + np.random.normal(0, 0.5)
        
        return IMUData(
            timestamp=current_time,
            acceleration=acceleration,
            gyroscope=gyroscope,
            temperature=temperature
        )
    
    def update_position(self, dt):
        """
        Update position based on acceleration
        
        Args:
            dt: Time step (seconds)
            
        Returns:
            Current position
        """
        data = self.read()
        
        
        self.velocity += data.acceleration * dt
        self.position += self.velocity * dt
        self.orientation += data.gyroscope * dt
        
        return self.position.copy()
    
    def get_orientation(self):
        """Get current orientation (roll, pitch, yaw)"""
        return self.orientation.copy()
    
    def calibrate(self):
        """Reset sensor drift"""
        self.drift = np.zeros(3)
        print("IMU calibrated!")


class LiDARSensor:
    """
    LiDAR (Light Detection and Ranging)
    Shoots lasers, measures distance
    """
    
    def __init__(self, max_range=100.0, resolution=5.0):
        """
        Initialize LiDAR
        
        Args:
            max_range: Maximum detection distance (meters)
            resolution: Angular resolution (degrees)
        """
        self.max_range = max_range
        self.resolution = resolution
        
        
        self.num_points = int(360 / resolution)
        
        print(f"✓ LiDAR initialized ({self.num_points} scan points)")
    
    def scan(self, obstacles=None):
        """
        Perform 360° scan
        
        Args:
            obstacles: List of (x, y, z) positions
            
        Returns:
            List of (angle, distance) measurements
        """
        measurements = []
        
        for i in range(self.num_points):
            angle = np.radians(i * self.resolution)
            
            if obstacles:
        
                distance = self._raycast(angle, obstacles)
            else:
                
                distance = np.random.uniform(10, self.max_range)
            
            
            distance += np.random.normal(0, 0.1)
            
            measurements.append({
                'angle': angle,
                'distance': distance
            })
        
        return measurements
    
    def _raycast(self, angle, obstacles):
        """
        Cast ray and find nearest obstacle
        
        Args:
            angle: Ray direction (radians)
            obstacles: List of obstacle positions
            
        Returns:
            Distance to nearest obstacle
        """
        
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        min_distance = self.max_range
        
        for obs_x, obs_y, obs_z in obstacles:
           
            t = obs_x * dx + obs_y * dy
            
            if t > 0:
                
                px = t * dx
                py = t * dy
                
               
                dist_to_ray = np.sqrt((px - obs_x)**2 + (py - obs_y)**2)
                
                if dist_to_ray < 1.0:  
                    distance = np.sqrt(px**2 + py**2)
                    min_distance = min(min_distance, distance)
        
        return min_distance
    
    def to_cartesian(self, measurements):
        """
        Convert polar (angle, distance) to cartesian (x, y)
        
        Args:
            measurements: List of {angle, distance}
            
        Returns:
            Nx2 array of (x, y) coordinates
        """
        points = []
        for m in measurements:
            x = m['distance'] * np.cos(m['angle'])
            y = m['distance'] * np.sin(m['angle'])
            points.append([x, y])
        
        return np.array(points)


class SensorFusion:
    """
    Combines IMU + LiDAR for accurate navigation
    """
    
    def __init__(self):
        self.imu = IMUSensor()
        self.lidar = LiDARSensor()
        
        
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)
        
        print("✓ Sensor Fusion initialized")
    
    def update(self, dt, obstacles=None):
        """
        Update fused sensor state
        
        Args:
            dt: Time step (seconds)
            obstacles: List of obstacle positions
            
        Returns:
            (position, velocity, orientation)
        """
        
        imu_data = self.imu.read()
        
        
        lidar_scan = self.lidar.scan(obstacles)
        
        
        self.velocity += imu_data.acceleration * dt
        self.position += self.velocity * dt
        self.orientation += imu_data.gyroscope * dt
        
        return self.position.copy(), self.velocity.copy(), self.orientation.copy()
    
    def get_nearby_obstacles(self, lidar_scan, threshold=10.0):
        """
        Find obstacles within threshold distance
        
        Args:
            lidar_scan: LiDAR measurements
            threshold: Distance threshold (meters)
            
        Returns:
            List of nearby obstacles
        """
        nearby = []
        for m in lidar_scan:
            if m['distance'] < threshold:
                nearby.append(m)
        
        return nearby



if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Q-NAV SENSOR SIMULATION")
    print("="*50 + "\n")
    
   
    print("Testing IMU sensor...")
    imu = IMUSensor(noise_level=0.01)
    
    for i in range(5):
        data = imu.read()
        print(f"  {i+1}. Accel: {data.acceleration}, Temp: {data.temperature:.1f}°C")
        time.sleep(0.2)
    
    print()
    
   
    print("Testing LiDAR sensor...")
    lidar = LiDARSensor(max_range=50, resolution=10)
    
    obstacles = [
        (10, 0, 0),
        (15, 5, 0),
        (-8, -8, 0)
    ]
    
    scan = lidar.scan(obstacles)
    print(f"  Scanned {len(scan)} points")
    
    
    for i in range(0, len(scan), 10):
        angle_deg = np.degrees(scan[i]['angle'])
        distance = scan[i]['distance']
        print(f"  Angle: {angle_deg:6.1f}° → Distance: {distance:.2f}m")
    
    print()
    
    
    print("Testing Sensor Fusion...")
    fusion = SensorFusion()
    
    print("  Running 10 simulation steps...")
    for i in range(10):
        pos, vel, orient = fusion.update(dt=0.1, obstacles=obstacles)
        
        if i % 3 == 0:
            print(f"  Step {i}: Pos={pos}, Vel={vel}")
        
        time.sleep(0.1)
    
    print("\n" + "="*50)
    print("✓ Sensor simulation complete!")
    print("="*50 + "\n")