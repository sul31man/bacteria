import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.colors import LogNorm
from bacteria import Bacteria
from synthetic_field import SyntheticFlowField

#this file is used to simulate the movement of a single bacterium in a flow field

class Simulation:
    def __init__(self, num_bacteria=20, domain_size=(2.2, 0.41), 
                 obstacle_center=(0.2, 0.2), obstacle_radius=0.05, 
                 bacteria_speed=0.0001):
        """
        Initialize the simulation
        
        Parameters:
            num_bacteria: Number of bacteria to simulate
            domain_size: (length, height) of the domain
            obstacle_center: (x, y) center of the cylindrical obstacle
            obstacle_radius: radius of the cylindrical obstacle
            bacteria_speed: Speed of bacteria's self-propulsion
        """
        print(f"Initializing simulation with {num_bacteria} bacteria")
        self.flow_field = SyntheticFlowField(domain_size, obstacle_center, obstacle_radius)
        self.bacteria = [Bacteria(self.flow_field, bacteria_speed) for _ in range(num_bacteria)]
        self.num_bacteria = num_bacteria
        self.obstacle_center = obstacle_center
        self.obstacle_radius = obstacle_radius
        self.domain_size = domain_size
    
    def run(self, steps=1000, dt=0.001, output_dir="results"):
        """Run the simulation for the specified number of steps"""
        print(f"Running simulation for {steps} steps...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize the flow field
        self.flow_field.visualize_velocity_field(output_dir)
        
        # Run simulation
        for i in range(steps):
            for bacteria in self.bacteria:
                bacteria.update(dt)
            
            # Optional: print progress every 10%
            if (i+1) % max(1, steps//10) == 0:
                print(f"Completed {i+1}/{steps} steps ({(i+1)/steps*100:.1f}%)")
        
        print("Simulation completed")
        
        # Plot results
        self._plot_trajectories(output_dir)
        self._plot_density(output_dir)
        self._plot_histograms(output_dir)
    
    def _plot_trajectories(self, output_dir):
        """Plot the trajectories of all bacteria"""
        plt.figure(figsize=(10, 4))
        
        # Plot each bacterium's trajectory
        for i, bacteria in enumerate(self.bacteria):
            plt.plot(bacteria.traj_x, bacteria.traj_y, linewidth=0.8, alpha=0.7)
        
        # Plot the obstacle
        circle = plt.Circle(self.obstacle_center, self.obstacle_radius, color='gray', fill=True)
        plt.gca().add_patch(circle)
        
        # Set axis limits
        plt.xlim(0, self.domain_size[0])
        plt.ylim(0, self.domain_size[1])
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Bacteria Trajectories (n={self.num_bacteria})')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "bacteria_trajectories.png"), dpi=300)
        plt.close()
    
    def _plot_density(self, output_dir):
        """Plot the density of bacteria positions"""
        # Collect all trajectory points
        all_x_points = []
        all_y_points = []
        
        for bacteria in self.bacteria:
            all_x_points.extend(bacteria.traj_x)
            all_y_points.extend(bacteria.traj_y)
        
        # Create density plot
        plt.figure(figsize=(10, 4))
        h = plt.hist2d(all_x_points, all_y_points, bins=[50, 20], 
                    range=[[0, self.domain_size[0]], [0, self.domain_size[1]]], 
                    cmap='viridis', norm=LogNorm())
        plt.colorbar(h[3], label='Bacteria Count')
        
        # Plot the obstacle
        circle = plt.Circle(self.obstacle_center, self.obstacle_radius, color='red', fill=True)
        plt.gca().add_patch(circle)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Bacteria Density Map')
        plt.savefig(os.path.join(output_dir, "bacteria_density.png"), dpi=300)
        plt.close()
    
    def _plot_histograms(self, output_dir):
        """Plot histograms of bacteria positions"""
        # Collect all trajectory points
        all_x_points = []
        all_y_points = []
        
        for bacteria in self.bacteria:
            all_x_points.extend(bacteria.traj_x)
            all_y_points.extend(bacteria.traj_y)
        
        # Create histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # X-axis histogram
        sns.histplot(all_x_points, ax=ax1, kde=True, color='blue')
        ax1.set_xlabel('x position')
        ax1.set_ylabel('Frequency')
        ax1.set_title('X-axis Distribution')
        
        # Y-axis histogram
        sns.histplot(all_y_points, ax=ax2, kde=True, color='green')
        ax2.set_xlabel('y position')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Y-axis Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bacteria_histograms.png"), dpi=300)
        plt.close()