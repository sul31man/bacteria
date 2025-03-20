#this file is used to simulate the movement of a single bacterium in a flow field

import numpy as np



class Bacteria:
    def __init__(self, flow_field, bacteria_speed=0.0001, start_position=None):
        """
        Initialize a bacterium in the flow field
        
        Parameters:
            flow_field: FlowField object
            bacteria_speed: Speed of the bacterium's self-propulsion
            start_position: (x, y) starting position, or None for random
        """
        self.flow_field = flow_field
        self.bacteria_speed = bacteria_speed
        self.timestep = 0.001
        
        # Set starting position
        if start_position is None:
            # Place randomly in the domain, but not inside the obstacle
            while True:
                self.x = np.random.uniform(0, flow_field.L * 0.2)  # Start on left side
                self.y = np.random.uniform(0, flow_field.H)
                if not flow_field.is_inside_obstacle(self.x, self.y):
                    break
        else:
            self.x, self.y = start_position
        
        # Random initial orientation
        self.theta = np.random.uniform(0, 2*np.pi)
        self.director = np.array([np.cos(self.theta), np.sin(self.theta)])
        
        # For tracking motion
        self.traj_x = [self.x]
        self.traj_y = [self.y]
    
    def update(self, dt=None):
        """Update the bacterium's position and orientation"""
        if dt is None:
            dt = self.timestep
        
        # Get flow velocity at current position
        u_fluid = self.flow_field.get_velocity(self.x, self.y)
        
        # Get D and W tensors at current position
        D = self.flow_field.get_strain_rate_tensor(self.x, self.y)
        W = self.flow_field.get_vorticity_tensor(self.x, self.y)
        
        # Update director vector based on tensors (Jeffery's equation)
        director = np.array([np.cos(self.theta), np.sin(self.theta)])
        
        # Jeffery's equation for director evolution
        d_director = np.matmul(W, director) + 0.5 * (np.matmul(D, director) - 
                   np.matmul(director, D) * np.dot(director, director))
        
        # Update angle with random noise (rotational diffusion)
        angle_change = np.arctan2(d_director[1], d_director[0])
        self.theta = self.theta + angle_change * dt + 0.1 * np.random.normal(0, np.sqrt(dt))
        
        # Calculate potential new position (fluid flow + self-propulsion)
        new_x = self.x + u_fluid[0] * dt + self.bacteria_speed * np.cos(self.theta) * dt
        new_y = self.y + u_fluid[1] * dt + self.bacteria_speed * np.sin(self.theta) * dt
        
        # Check if new position is inside obstacle
        if self.flow_field.is_inside_obstacle(new_x, new_y):
            # Get the normal vector at the closest point on the obstacle
            normal = self.flow_field.get_obstacle_normal(new_x, new_y)
            
            # Reflect velocity and direction
            # First, project velocity onto normal
            vel = np.array([u_fluid[0] + self.bacteria_speed * np.cos(self.theta),
                           u_fluid[1] + self.bacteria_speed * np.sin(self.theta)])
            
            vel_norm = np.dot(vel, normal)
            
            # Reflect velocity
            if vel_norm < 0:  # Only reflect if moving toward obstacle
                vel = vel - 2 * vel_norm * normal
                
                # Update direction based on reflected velocity
                self.theta = np.arctan2(vel[1], vel[0])
                
                # Place bacterium slightly away from obstacle
                dist_to_move = self.bacteria_speed * dt
                safe_x = self.x + normal[0] * dist_to_move
                safe_y = self.y + normal[1] * dist_to_move
                
                # Only move if safe position is not inside obstacle
                if not self.flow_field.is_inside_obstacle(safe_x, safe_y):
                    self.x = safe_x
                    self.y = safe_y
            else:
                # If not moving toward obstacle, just move along the boundary
                tangent = np.array([-normal[1], normal[0]])  # Tangent vector
                vel_tang = np.dot(vel, tangent)
                self.x = self.x + vel_tang * tangent[0] * dt
                self.y = self.y + vel_tang * tangent[1] * dt
        else:
            # Check if outside domain boundaries
            domain_x_max = self.flow_field.L
            domain_y_max = self.flow_field.H
            
            # Handle x boundaries
            if new_x < 0:
                new_x = 0
                self.theta = np.random.uniform(-np.pi/2, np.pi/2)  # Random rightward direction
            elif new_x > domain_x_max:
                new_x = domain_x_max
                self.theta = np.random.uniform(np.pi/2, 3*np.pi/2)  # Random leftward direction
            
            # Handle y boundaries
            if new_y < 0:
                new_y = 0
                self.theta = np.random.uniform(0, np.pi)  # Random upward direction
            elif new_y > domain_y_max:
                new_y = domain_y_max
                self.theta = np.random.uniform(np.pi, 2*np.pi)  # Random downward direction
            
            # Update position
            self.x = new_x
            self.y = new_y
        
        # Store trajectory
        self.traj_x.append(self.x)
        self.traj_y.append(self.y)
        
        return self.x, self.y