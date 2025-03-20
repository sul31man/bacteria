from Simulation import Simulation

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate bacteria in a synthetic CFD flow field')
    parser.add_argument('--num_bacteria', type=int, default=20, help='Number of bacteria to simulate')
    parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps')
    parser.add_argument('--dt', type=float, default=0.001, help='Time step size')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--bacteria_speed', type=float, default=0.0001, help='Self-propulsion speed of bacteria')
    
    args = parser.parse_args()
    
    # Create and run the simulation
    sim = Simulation(
        num_bacteria=args.num_bacteria,
        bacteria_speed=args.bacteria_speed
    )
    
    sim.run(steps=args.steps, dt=args.dt, output_dir=args.output)
    print(f"Results saved to {args.output}/") 