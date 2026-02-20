"""
Q-Nav Navigation Module
Plans safe paths and avoids obstacles
"""

import numpy as np
from typing import List, Tuple


class Navigator:
    """
    Spacecraft navigation and path planning
    """
    
    def __init__(self, safe_distance=2.0):
        """
        Initialize navigator
        
        Args:
            safe_distance: Minimum safe distance from obstacles (meters)
        """
        self.safe_distance = safe_distance
        
        
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.target = np.zeros(3)
        
        
        self.path_history = []
        
        print(f"✓ Navigator initialized (safe distance: {safe_distance}m)")
    
    def set_target(self, target_position):
        """
        Set navigation target
        
        Args:
            target_position: (x, y, z) destination
        """
        self.target = np.array(target_position)
        print(f"Target set: {self.target}")
    
    def update_state(self, position, velocity):
        """
        Update current position and velocity
        
        Args:
            position: Current (x, y, z)
            velocity: Current velocity vector
        """
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.path_history.append(self.position.copy())
    
    def calculate_distance_to_target(self):
        """
        Calculate straight-line distance to target
        
        Returns:
            Distance in meters
        """
        return np.linalg.norm(self.target - self.position)
    
    def calculate_direction_to_target(self):
        """
        Calculate direction vector to target
        
        Returns:
            Normalized direction vector
        """
        direction = self.target - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.001:
            return np.zeros(3)
        
        return direction / distance
    
    def check_obstacles(self, obstacles):
        """
        Check if any obstacles are too close
        
        Args:
            obstacles: List of (x, y, z) obstacle positions
            
        Returns:
            (is_danger, closest_obstacle, distance)
        """
        if not obstacles:
            return False, None, float('inf')
        
        min_distance = float('inf')
        closest_obs = None
        
        for obs in obstacles:
            obs_pos = np.array(obs)
            distance = np.linalg.norm(obs_pos - self.position)
            
            if distance < min_distance:
                min_distance = distance
                closest_obs = obs_pos
        
        is_danger = min_distance < self.safe_distance
        
        return is_danger, closest_obs, min_distance
    
    def find_avoidance_direction(self, obstacles):
        """
        Find safe direction to avoid obstacles
        
        Args:
            obstacles: List of obstacle positions
            
        Returns:
            Safe direction vector
        """
        if not obstacles:
            return self.calculate_direction_to_target()
        
        
        target_dir = self.calculate_direction_to_target()
        
        
        is_danger, closest_obs, distance = self.check_obstacles(obstacles)
        
        if not is_danger:
            return target_dir
        
  
        to_obstacle = closest_obs - self.position
        to_obstacle_norm = to_obstacle / np.linalg.norm(to_obstacle)
        
        perp = np.cross(to_obstacle_norm, np.array([0, 0, 1]))
        if np.linalg.norm(perp) < 0.001:
            perp = np.cross(to_obstacle_norm, np.array([0, 1, 0]))
        
        perp = perp / np.linalg.norm(perp)
        
        
        avoidance_dir = 0.3 * target_dir + 0.7 * perp
        avoidance_dir = avoidance_dir / np.linalg.norm(avoidance_dir)
        
        return avoidance_dir
    
    def calculate_desired_velocity(self, obstacles, max_speed=1.0):
        """
        Calculate desired velocity vector
        
        Args:
            obstacles: List of obstacle positions
            max_speed: Maximum speed (m/s)
            
        Returns:
            Desired velocity vector
        """
        
        direction = self.find_avoidance_direction(obstacles)
        
       
        distance = self.calculate_distance_to_target()
        
        if distance < 1.0:
            
            speed = max_speed * (distance / 1.0)
        else:
            speed = max_speed
        
        return direction * speed
    
    def navigate_step(self, obstacles, dt=0.1):
        """
        Execute one navigation step
        
        Args:
            obstacles: List of obstacle positions
            dt: Time step (seconds)
            
        Returns:
            (new_position, new_velocity, status)
        """
        
        desired_vel = self.calculate_desired_velocity(obstacles)
         
        self.velocity = 0.8 * self.velocity + 0.2 * desired_vel
        
        
        self.position += self.velocity * dt
        
       
        distance = self.calculate_distance_to_target()
        is_danger, closest_obs, obs_dist = self.check_obstacles(obstacles)
        
        if distance < 0.5:
            status = "TARGET REACHED"
        elif is_danger:
            status = f"AVOIDING OBSTACLE ({obs_dist:.2f}m)"
        else:
            status = f"NAVIGATING ({distance:.2f}m to target)"
        
        return self.position.copy(), self.velocity.copy(), status


class CollisionPredictor:
    """
    Predicts potential collisions
    """
    
    def __init__(self, time_horizon=5.0):
        """
        Initialize predictor
        
        Args:
            time_horizon: How far ahead to predict (seconds)
        """
        self.time_horizon = time_horizon
        print(f"✓ Collision Predictor initialized ({time_horizon}s horizon)")
    
    def predict_collision(self, position, velocity, obstacles, obstacle_velocities=None):
        """
        Predict if collision will occur
        
        Args:
            position: Current position
            velocity: Current velocity
            obstacles: List of obstacle positions
            obstacle_velocities: List of obstacle velocities (optional)
            
        Returns:
            (will_collide, time_to_collision, collision_point)
        """
        if not obstacles:
            return False, float('inf'), None
        
        
        if obstacle_velocities is None:
            obstacle_velocities = [np.zeros(3) for _ in obstacles]
        
        dt = 0.1  
        steps = int(self.time_horizon / dt)
        
        for step in range(steps):
            t = step * dt
            
            
            future_pos = position + velocity * t
            
            
            for obs_pos, obs_vel in zip(obstacles, obstacle_velocities):
                future_obs_pos = obs_pos + obs_vel * t
                
                distance = np.linalg.norm(future_pos - future_obs_pos)
                
                if distance < 2.0: 
                    return True, t, future_obs_pos
        
        return False, float('inf'), None


class PathPlanner:
    """
    Plans optimal paths from start to goal
    """
    
    def __init__(self):
        print("✓ Path Planner initialized")
    
    def plan_path(self, start, goal, obstacles, num_waypoints=5):
        """
        Plan path with waypoints
        
        Args:
            start: Start position
            goal: Goal position
            obstacles: List of obstacle positions
            num_waypoints: Number of intermediate points
            
        Returns:
            List of waypoint positions
        """
        
        waypoints = [start]
        
        
        for i in range(1, num_waypoints + 1):
            t = i / (num_waypoints + 1)
            point = start + t * (goal - start)
            
            
            point = self._adjust_for_obstacles(point, obstacles)
            waypoints.append(point)
        
        waypoints.append(goal)
        
        return waypoints
    
    def _adjust_for_obstacles(self, point, obstacles, safe_margin=3.0):
        """
        Move point away from obstacles
        
        Args:
            point: Point to adjust
            obstacles: List of obstacles
            safe_margin: Safe distance (meters)
            
        Returns:
            Adjusted point
        """
        adjusted = point.copy()
        
        for obs in obstacles:
            obs_pos = np.array(obs)
            distance = np.linalg.norm(adjusted - obs_pos)
            
            if distance < safe_margin:
              
                direction = adjusted - obs_pos
                direction = direction / np.linalg.norm(direction)
                adjusted = obs_pos + direction * safe_margin
        
        return adjusted



if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Q-NAV NAVIGATION DEMO")
    print("="*50 + "\n")
    
   
    navigator = Navigator(safe_distance=2.0)
    
   
    navigator.set_target([50, 0, 0])
    
    
    obstacles = [
        np.array([10, 0, 0]),
        np.array([20, 5, 0]),
        np.array([30, -3, 0]),
        np.array([40, 2, 0])
    ]
    
    print(f"Starting at: {navigator.position}")
    print(f"Target: {navigator.target}")
    print(f"Obstacles: {len(obstacles)}\n")
    
    
    print("Simulating navigation...")
    for step in range(100):
        
        navigator.update_state(navigator.position, navigator.velocity)
        
        
        pos, vel, status = navigator.navigate_step(obstacles, dt=0.1)
        
        
        if step % 10 == 0:
            distance = navigator.calculate_distance_to_target()
            print(f"Step {step:3d}: Pos={pos} | {status}")
        
        
        if "REACHED" in status:
            print(f"\n✓ Target reached at step {step}!")
            break
    
    print()
    
    
    print("Testing collision predictor...")
    predictor = CollisionPredictor(time_horizon=5.0)
    
    test_pos = np.array([5, 0, 0])
    test_vel = np.array([1, 0, 0])
    
    will_collide, time_to, point = predictor.predict_collision(
        test_pos, test_vel, obstacles
    )
    
    if will_collide:
        print(f"  ⚠️  Collision predicted in {time_to:.2f}s at {point}")
    else:
        print(f"  ✓ No collision detected")
    
    print()
    
    
    print("Testing path planner...")
    planner = PathPlanner()
    
    waypoints = planner.plan_path(
        start=np.array([0, 0, 0]),
        goal=np.array([50, 0, 0]),
        obstacles=obstacles,
        num_waypoints=5
    )
    
    print(f"  Generated {len(waypoints)} waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"    {i+1}. {wp}")
    
    print("\n" + "="*50)
    print("✓ Navigation demo complete!")
    print("="*50 + "\n")