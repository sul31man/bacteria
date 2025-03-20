#this file is used to generate a synthetic field for the purpose of testing the model
#due to issues with the real data from the stokes equations solver, this is the field that will be used

import numpy as np
import matplotlib.pyplot as plt
import os


class SyntheticFlowField:
    def __init__(self, domain_size=(2.2, 0.41), obstacle_center=(0.2, 0.2), obstacle_radius=0.05):
        """
        Create a synthetic flow field with an obstacle using analytical functions
        
        Parameters:
            domain_size: (length, height) of the domain
            obstacle_center: (x, y) center of the cylindrical obstacle
            obstacle_radius: radius of the cylindrical obstacle
        """
        self.L, self.H = domain_size
        self.obstacle_x, self.obstacle_y = obstacle_center
        self.r = obstacle_radius
        
        print("Created synthetic flow field")
    
    def is_inside_obstacle(self, x, y):
        """Check if a point is inside the cylindrical obstacle"""
        dist_sq = (x - self.obstacle_x)**2 + (y - self.obstacle_y)**2
        return dist_sq <= self.r**2
    
    def get_obstacle_normal(self, x, y):
        """Get the normal vector pointing outward from the obstacle at point (x,y)"""
        dx = x - self.obstacle_x
        dy = y - self.obstacle_y
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 1e-10:  # Avoid division by zero
            return np.array([1.0, 0.0])  # Arbitrary direction if at center
        return np.array([dx / dist, dy / dist])
    
    def get_velocity(self, x, y):
        """Get the velocity at point (x, y) using analytical formula"""
        # Check if inside obstacle
        if self.is_inside_obstacle(x, y):
            return np.array([0.0, 0.0])
        
        # Calculate parabolic profile
        vel_x = 4.0 * 1.5 * y * (self.H - y) / (self.H * self.H)
        
        # Calculate distance to obstacle
        dist_sq = (x - self.obstacle_x)**2 + (y - self.obstacle_y)**2
        dist = np.sqrt(dist_sq)
        
        # Create shadow of the obstacle - flow velocity decreases near the obstacle
        obstacle_influence = 1.0 - np.exp(-5.0 * (dist - self.r) / self.r)
        obstacle_influence = np.clip(obstacle_influence, 0, 1)
        
        # Apply obstacle influence to x-component
        vel_x = vel_x * obstacle_influence
        
        # Add some y-component for more realistic flow around obstacle
        dx = x - self.obstacle_x
        dy = y - self.obstacle_y
        angle = np.arctan2(dy, dx)
        vel_y = 0.2 * vel_x * np.sin(angle) * np.exp(-dist_sq / (2 * self.r * self.r))
        
        return np.array([vel_x, vel_y])
    
    def get_strain_rate_tensor(self, x, y):
        """
        Calculate the rate-of-strain tensor D using finite differences
        
        Parameters:
            x, y: coordinates to evaluate at
            
        Returns:
            2x2 tensor D = 0.5*(grad(u) + grad(u)^T)
        """
        if self.is_inside_obstacle(x, y):
            return np.zeros((2, 2))
        
        # Small step for finite differences
        h = 1e-6
        
        # Get velocities at nearby points
        v_center = self.get_velocity(x, y)
        v_x_plus = self.get_velocity(x + h, y)
        v_x_minus = self.get_velocity(x - h, y)
        v_y_plus = self.get_velocity(x, y + h)
        v_y_minus = self.get_velocity(x, y - h)
        
        # Calculate velocity gradients
        dudx = (v_x_plus[0] - v_x_minus[0]) / (2 * h)
        dudy = (v_y_plus[0] - v_y_minus[0]) / (2 * h)
        dvdx = (v_x_plus[1] - v_x_minus[1]) / (2 * h)
        dvdy = (v_y_plus[1] - v_y_minus[1]) / (2 * h)
        
        # Construct and return the rate-of-strain tensor D
        return 0.5 * np.array([
            [2 * dudx, dudy + dvdx],
            [dudy + dvdx, 2 * dvdy]
        ])
    
    def get_vorticity_tensor(self, x, y):
        """
        Calculate the vorticity tensor W using finite differences
        
        Parameters:
            x, y: coordinates to evaluate at
            
        Returns:
            2x2 tensor W = 0.5*(grad(u) - grad(u)^T)
        """
        if self.is_inside_obstacle(x, y):
            return np.zeros((2, 2))
        
        # Small step for finite differences
        h = 1e-6
        
        # Get velocities at nearby points
        v_center = self.get_velocity(x, y)
        v_x_plus = self.get_velocity(x + h, y)
        v_x_minus = self.get_velocity(x - h, y)
        v_y_plus = self.get_velocity(x, y + h)
        v_y_minus = self.get_velocity(x, y - h)
        
        # Calculate velocity gradients
        dudx = (v_x_plus[0] - v_x_minus[0]) / (2 * h)
        dudy = (v_y_plus[0] - v_y_minus[0]) / (2 * h)
        dvdx = (v_x_plus[1] - v_x_minus[1]) / (2 * h)
        dvdy = (v_y_plus[1] - v_y_minus[1]) / (2 * h)
        
        # Construct and return the vorticity tensor W
        return 0.5 * np.array([
            [0, dudy - dvdx],
            [dvdx - dudy, 0]
        ])
    
    def visualize_velocity_field(self, output_dir="results", resolution=100):
        """Visualize the flow field"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a grid of points
        x = np.linspace(0, self.L, resolution)
        y = np.linspace(0, self.H, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Get velocity at each point
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(resolution):
            for j in range(resolution):
                vel = self.get_velocity(X[i, j], Y[i, j])
                U[i, j] = vel[0]
                V[i, j] = vel[1]
        
        # Calculate magnitude
        magnitude = np.sqrt(U**2 + V**2)
        
        # Plot velocity magnitude
        plt.figure(figsize=(12, 4))
        plt.contourf(X, Y, magnitude, cmap='viridis')
        plt.colorbar(label='Velocity magnitude')
        
        # Add obstacle
        circle = plt.Circle((self.obstacle_x, self.obstacle_y), self.r, color='white', edgecolor='black')
        plt.gca().add_patch(circle)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Velocity Magnitude')
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, "velocity_magnitude.png"), dpi=300)
        
        # Plot velocity vectors
        plt.figure(figsize=(12, 4))
        skip = 5  # Skip some points for clearer visualization
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  U[::skip, ::skip], V[::skip, ::skip],
                  magnitude[::skip, ::skip], cmap='viridis')
        
        # Add obstacle
        circle = plt.Circle((self.obstacle_x, self.obstacle_y), self.r, color='white', edgecolor='black')
        plt.gca().add_patch(circle)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Velocity Vectors')
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, "velocity_vectors.png"), dpi=300)
        
        print(f"Saved visualization to {output_dir}")


    