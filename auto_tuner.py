"""
Automated parameter tuning system for RT-RRT* cost function.
Runs experiments and optimizes parameters for faster convergence and better paths.
"""

import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from rt_rrt_star import RTRRTStar
from boat_dynamics import MilliAmpere1Sim
from config import WORLD_BOUNDS
import matplotlib.pyplot as plt

# Try to import tqdm for progress bar, fallback to simple progress
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        """Simple fallback progress indicator"""
        total = kwargs.get('total', len(iterable) if hasattr(iterable, '__len__') else None)
        desc = kwargs.get('desc', '')
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc} [{i+1}/{total}] {100*(i+1)//total}%", end='', flush=True)
            yield item
        print()  # New line after completion


class ParameterTuner:
    """
    Automatically tunes cost function parameters by running experiments
    and analyzing convergence speed and path quality.
    """
    
    def __init__(self, results_dir="tuning_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Parameters to tune
        self.param_names = [
            # Cost function weights (for steer_dynamic)
            'position_weight',
            'heading_weight', 
            'velocity_weight',
            'progress_weight',
            # R-tree spatial index weights (for nearest neighbor search)
            'rtree_position_weight',
            'rtree_heading_weight',
            'rtree_velocity_weight'
        ]
        
        # Test scenarios
        self.scenarios = self._create_scenarios()
        
    def _create_scenarios(self):
        """Create diverse test scenarios"""
        return [
            {
                'name': 'simple_straight',
                'start': np.array([15.0, 15.0, 0.0, 0.0, 0.0, 0.0]),
                'goal': np.array([18.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
                'obstacles': [
                    (np.array([10.0, 8.0]), 1.5),
                    (np.array([10.0, -4.0]), 1.5),
                ]
            },
            {
                'name': 'narrow_passage',
                'start': np.array([15.0, 15.0, 0.0, 0.0, 0.0, 0.0]),
                'goal': np.array([18.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
                'obstacles': [
                    (np.array([10.0, 5.0]), 2.0),
                    (np.array([10.0, -1.0]), 2.0),
                ]
            },
            {
                'name': 'turn_required',
                'start': np.array([15.0, 15.0, 0.0, 0.0, 0.0, 0.0]),
                'goal': np.array([18.0, 18.0, np.pi/2, 0.0, 0.0, 0.0]),
                'obstacles': [
                    (np.array([8.0, 8.0]), 2.0),
                    (np.array([12.0, 12.0]), 2.0),
                ]
            },
            {
                'name': 'cluttered',
                'start': np.array([15.0, 15.0, 0.0, 0.0, 0.0, 0.0]),
                'goal': np.array([18.0, 18.0, 0.0, 0.0, 0.0, 0.0]),
                'obstacles': [
                    (np.array([6.0, 4.0]), 1.2),
                    (np.array([10.0, 6.0]), 1.0),
                    (np.array([14.0, 8.0]), 1.5),
                    (np.array([8.0, 12.0]), 1.3),
                    (np.array([12.0, 14.0]), 1.1),
                ]
            }
        ]
    
    def run_experiment(self, params, scenario, max_time=10.0, verbose=False):
        """
        Run a single experiment with given parameters.
        
        Returns metrics about convergence and path quality.
        """
        # Separate cost function weights from rtree weights
        cost_weights = {k: v for k, v in params.items() if not k.startswith('rtree_')}
        rtree_weights = {
            'position_weight': params.get('rtree_position_weight', 1.0),
            'heading_weight': params.get('rtree_heading_weight', 0.5),
            'velocity_weight': params.get('rtree_velocity_weight', 0.3)
        }
        
        # Update steer.py parameters
        import steer
        old_weights = self._backup_weights(steer)
        self._apply_weights(steer, cost_weights)
        
        try:
            # Initialize planner with rtree weights
            planner = RTRRTStar(WORLD_BOUNDS, scenario['start'], 
                              rtree_weights=rtree_weights)
            boat = MilliAmpere1Sim(scenario['start'][:2])
            
            x_agent = scenario['start'][:2]
            x_goal = scenario['goal']
            obstacles = scenario['obstacles']
            
            # Metrics
            path_found_time = None
            path_lengths = []
            costs = []
            iterations = 0
            
            start_time = time.perf_counter()
            
            # Run planning loop
            while time.perf_counter() - start_time < max_time:
                dt = 0.05  # Fixed timestep for consistency
                
                path = planner.step(x_agent, x_goal, obstacles, [], dt)
                iterations += 1
                
                # Record metrics
                if len(path) > 0:
                    if path_found_time is None:
                        path_found_time = time.perf_counter() - start_time
                    
                    # Calculate path length
                    path_length = sum(
                        np.linalg.norm(path[i].x[:2] - path[i+1].x[:2])
                        for i in range(len(path) - 1)
                    )
                    path_lengths.append(path_length)
                    
                    # Record best cost
                    if len(path) > 0:
                        costs.append(path[-1].cost)
                
                # Simulate boat movement
                if len(path) > 1:
                    eta_ref = path[0].x[:3]
                    boat.step(eta_ref)
                    x_agent = boat.x[:2]
                
                # Check if reached goal
                if np.linalg.norm(x_agent - x_goal[:2]) < 0.5:
                    break
            
            elapsed_time = time.perf_counter() - start_time
            
            # Calculate metrics
            metrics = {
                'success': path_found_time is not None,
                'convergence_time': path_found_time if path_found_time else max_time,
                'final_path_length': path_lengths[-1] if path_lengths else float('inf'),
                'min_path_length': min(path_lengths) if path_lengths else float('inf'),
                'iterations': iterations,
                'reached_goal': np.linalg.norm(x_agent - x_goal[:2]) < 0.5,
                'path_improvement': (path_lengths[0] - path_lengths[-1]) / path_lengths[0] if len(path_lengths) > 1 and path_lengths[0] > 0 else 0.0,
                'cost_history': costs
            }
            
            return metrics
            
        finally:
            # Restore original weights
            self._restore_weights(steer, old_weights)
    
    def _backup_weights(self, steer_module):
        """Backup current cost function weights"""
        return steer_module.get_cost_weights()
    
    def _apply_weights(self, steer_module, params):
        """Apply new weights to steer module"""
        steer_module.set_cost_weights(**params)
    
    def _restore_weights(self, steer_module, old_weights):
        """Restore original weights"""
        steer_module.set_cost_weights(**old_weights)
    
    def evaluate_params(self, params, verbose=False):
        """
        Evaluate a parameter set across all scenarios.
        Returns aggregate score.
        """
        scenario_results = []
        
        for scenario in self.scenarios:
            metrics = self.run_experiment(params, scenario, max_time=8.0, verbose=verbose)
            scenario_results.append(metrics)
        
        # Aggregate score (lower is better)
        # Prioritize: success rate, convergence speed, path quality
        success_rate = sum(r['success'] for r in scenario_results) / len(scenario_results)
        avg_convergence_time = np.mean([r['convergence_time'] for r in scenario_results])
        avg_path_length = np.mean([r['final_path_length'] for r in scenario_results if r['success']])
        
        # Score combines multiple factors
        # High penalty for failures, reward fast convergence and short paths
        if success_rate == 0:
            score = 1000.0  # Very bad
        else:
            score = (
                (1.0 - success_rate) * 100 +  # Failure penalty
                avg_convergence_time * 5 +      # Speed penalty
                avg_path_length * 0.5           # Path length penalty
            )
        
        return {
            'score': score,
            'success_rate': success_rate,
            'avg_convergence_time': avg_convergence_time,
            'avg_path_length': avg_path_length,
            'scenario_results': scenario_results
        }
    
    def random_search(self, n_iterations=20, verbose=False):
        """
        Random search for good parameters.
        """
        print("Starting random search...")
        
        best_score = float('inf')
        best_params = None
        all_results = []
        
        # Use progress bar
        pbar = tqdm(range(n_iterations), desc="Random Search", unit="iter")
        
        for i in pbar:
            # Sample random parameters
            params = {
                # Cost function weights
                'position_weight': np.random.uniform(1.0, 20.0),
                'heading_weight': np.random.uniform(0.5, 5.0),
                'velocity_weight': np.random.uniform(0.01, 1.0),
                'progress_weight': np.random.uniform(1.0, 10.0),
                # R-tree weights
                'rtree_position_weight': np.random.uniform(0.5, 2.0),
                'rtree_heading_weight': np.random.uniform(0.1, 1.5),
                'rtree_velocity_weight': np.random.uniform(0.05, 0.8)
            }
            
            result = self.evaluate_params(params, verbose=verbose)
            result['params'] = params
            result['iteration'] = i
            all_results.append(result)
            
            if result['score'] < best_score:
                best_score = result['score']
                best_params = params
                pbar.set_postfix({'best_score': f'{best_score:.3f}'})
        
        return best_params, best_score, all_results
    
    def grid_search(self, param_ranges, verbose=False):
        """
        Grid search over parameter ranges.
        """
        print("Starting grid search...")
        
        # Create grid
        from itertools import product
        
        param_grid = []
        for pos_w in param_ranges['position_weight']:
            for head_w in param_ranges['heading_weight']:
                for vel_w in param_ranges['velocity_weight']:
                    for prog_w in param_ranges['progress_weight']:
                        param_grid.append({
                            'position_weight': pos_w,
                            'heading_weight': head_w,
                            'velocity_weight': vel_w,
                            'progress_weight': prog_w
                        })
        
        best_score = float('inf')
        best_params = None
        all_results = []
        
        # Use progress bar
        pbar = tqdm(param_grid, desc="Grid Search", unit="config")
        
        for i, params in enumerate(pbar):
            result = self.evaluate_params(params, verbose=verbose)
            result['params'] = params
            result['iteration'] = i
            all_results.append(result)
            
            if result['score'] < best_score:
                best_score = result['score']
                best_params = params
                pbar.set_postfix({'best_score': f'{best_score:.3f}'})
        
        return best_params, best_score, all_results
    
    def save_results(self, results, method='random_search'):
        """Save tuning results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"tuning_{method}_{timestamp}.json"
        
        # Convert numpy types to native Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        with open(filename, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def plot_results(self, all_results):
        """Plot tuning results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        scores = [r['score'] for r in all_results]
        success_rates = [r['success_rate'] for r in all_results]
        
        # Score over iterations
        axes[0, 0].plot(scores)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Score (lower is better)')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].grid(True)
        
        # Success rate over iterations
        axes[0, 1].plot(success_rates)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_title('Success Rate Over Iterations')
        axes[0, 1].grid(True)
        
        # Parameter influence on score
        param_names = ['position_weight', 'heading_weight', 'velocity_weight', 'progress_weight']
        for idx, param_name in enumerate(param_names[:2]):
            param_values = [r['params'][param_name] for r in all_results]
            axes[1, 0].scatter(param_values, scores, alpha=0.5, label=param_name)
        axes[1, 0].set_xlabel('Parameter Value')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Parameter Influence (Position & Heading)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        for idx, param_name in enumerate(param_names[2:]):
            param_values = [r['params'][param_name] for r in all_results]
            axes[1, 1].scatter(param_values, scores, alpha=0.5, label=param_name)
        axes[1, 1].set_xlabel('Parameter Value')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Parameter Influence (Velocity & Progress)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"tuning_plot_{timestamp}.png"
        plt.savefig(filename, dpi=150)
        print(f"Plot saved to {filename}")
        plt.close()
    
    def plot_best_iteration(self, best_params):
        """
        Run the best parameters once and visualize the tree for each scenario.
        Creates a matplotlib plot similar to pygama_demo visualization.
        """
        print("\nGenerating visualization of best parameters...")
        
        # Separate cost function weights from rtree weights
        cost_weights = {k: v for k, v in best_params.items() if not k.startswith('rtree_')}
        rtree_weights = {
            'position_weight': best_params.get('rtree_position_weight', 1.0),
            'heading_weight': best_params.get('rtree_heading_weight', 0.5),
            'velocity_weight': best_params.get('rtree_velocity_weight', 0.3)
        }
        
        # Update steer.py parameters
        import steer
        old_weights = self._backup_weights(steer)
        self._apply_weights(steer, cost_weights)
        
        try:
            # Only test first scenario for debugging
            scenario = self.scenarios[0]
            
            # Run planner for this scenario
            planner = RTRRTStar(WORLD_BOUNDS, scenario['start'], 
                              rtree_weights=rtree_weights)
            boat = MilliAmpere1Sim(scenario['start'][:2])
            
            x_agent = scenario['start'][:2]
            x_goal = scenario['goal']
            obstacles = scenario['obstacles']
            
            print(f"Starting visualization for {scenario['name']}")
            print(f"Start: {scenario['start'][:2]}, Goal: {scenario['goal'][:2]}")
            print(f"Initial tree nodes: {len(planner.tree.index.nodes)}")
            
            # Run for shorter time to get tree visualization
            start_time = time.perf_counter()
            path = []
            iteration = 0
            while time.perf_counter() - start_time < 2.0:
                dt = 0.05
                path = planner.step(x_agent, x_goal, obstacles, [], dt)
                iteration += 1
                
                if iteration % 20 == 0:
                    print(f"Iteration {iteration}: {len(planner.tree.index.nodes)} nodes, path length: {len(path)}")
                
                # Simulate boat movement
                if len(path) > 1:
                    eta_ref = path[0].x[:3]
                    boat.step(eta_ref)
                    x_agent = boat.x[:2]
            
            print(f"Final tree nodes: {len(planner.tree.index.nodes)}")
            print(f"Final path length: {len(path)}")
            
            # Create single plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_xlim(WORLD_BOUNDS[0])
            ax.set_ylim(WORLD_BOUNDS[1])
            ax.set_aspect('equal')
            ax.set_title(f"{scenario['name']}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Draw obstacles
            for obs_center, obs_radius in obstacles:
                circle = plt.Circle(obs_center, obs_radius, color='red', alpha=0.5)
                ax.add_patch(circle)
            
            # Draw tree edges
            for node_id in planner.tree.index.nodes:
                node = planner.tree.index.nodes[node_id]
                if node.parent is not None:
                    if node.blocked or node.parent.blocked:
                        color = 'magenta'
                        alpha = 0.3
                    else:
                        color = 'gray'
                        alpha = 0.5
                    ax.plot([node.parent.x[0], node.x[0]], 
                           [node.parent.x[1], node.x[1]], 
                           color=color, alpha=alpha, linewidth=0.5)
            
            # Draw tree nodes
            for node_id in planner.tree.index.nodes:
                node = planner.tree.index.nodes[node_id]
                if node.cost == 0:  # Root
                    ax.plot(node.x[0], node.x[1], 'ro', markersize=6)
                elif node.cost == float("inf"):  # Blocked
                    ax.plot(node.x[0], node.x[1], 'mo', markersize=2, alpha=0.5)
                else:
                    ax.plot(node.x[0], node.x[1], 'b.', markersize=2)
            
            # Draw path
            if len(path) > 1:
                path_x = [n.x[0] for n in path]
                path_y = [n.x[1] for n in path]
                ax.plot(path_x, path_y, 'g-', linewidth=2, label='Path')
            
            # Draw start and goal
            ax.plot(scenario['start'][0], scenario['start'][1], 'go', 
                   markersize=10, label='Start')
            ax.plot(scenario['goal'][0], scenario['goal'][1], 'r*', 
                   markersize=15, label='Goal')
            
            # Draw boat position
            ax.plot(boat.x[0], boat.x[1], 'bs', markersize=8, label='Boat')
            
            ax.legend(loc='upper right', fontsize=8)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.results_dir / f"best_tree_visualization_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Tree visualization saved to {filename}")
            plt.close()
            
        finally:
            # Restore original weights
            self._restore_weights(steer, old_weights)


def main():
    """Run automated parameter tuning"""
    tuner = ParameterTuner()
    
    print("=" * 60)
    print("RT-RRT* AUTOMATED PARAMETER TUNING")
    print("=" * 60)
    
    # Option 1: Random search (faster, good for exploration)
    print("\nRunning random search...")
    best_params, best_score, results = tuner.random_search(n_iterations=15, verbose=True)
    
    print("\n" + "=" * 60)
    print("BEST PARAMETERS FOUND:")
    print("=" * 60)
    for param, value in best_params.items():
        print(f"  {param:20s}: {value:.3f}")
    print(f"\nBest Score: {best_score:.2f}")
    
    # Save results
    tuner.save_results({
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }, method='random_search')
    
    # Plot results
    tuner.plot_results(results)
    
    # Plot tree visualization for best parameters
    tuner.plot_best_iteration(best_params)
    
    # Option 2: Grid search (more thorough, slower)
    # Uncomment to run grid search instead
    """
    param_ranges = {
        'position_weight': [5.0, 10.0, 15.0],
        'heading_weight': [1.0, 2.0, 3.0],
        'velocity_weight': [0.05, 0.1, 0.2],
        'progress_weight': [3.0, 5.0, 7.0]
    }
    best_params, best_score, results = tuner.grid_search(param_ranges, verbose=True)
    tuner.save_results({
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }, method='grid_search')
    tuner.plot_results(results)
    """


if __name__ == "__main__":
    main()
