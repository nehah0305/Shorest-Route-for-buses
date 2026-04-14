"""
Reinforcement Learning Module: Dynamic route optimization using RL

Implements Q-Learning for improving routes over time based on:
- Traffic patterns
- Actual travel times
- Real-time delays
- Historical performance data
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class RLOptimizer:
    """Optimize routes dynamically using Reinforcement Learning (Q-Learning)."""
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.995
    ):
        """
        Initialize RL optimizer with Q-Learning parameters.
        
        Args:
            learning_rate: Learning rate for Q-value updates (0-1)
            discount_factor: Discount factor for future rewards (0-1)
            exploration_rate: Initial exploration probability (0-1)
            exploration_decay: Decay rate for exploration (0-1)
        """
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        
        self.q_table = {}  # State-action Q-values
        self.traffic_history = {}  # Historical traffic data
        self.reward_history = []  # Track rewards over time
        self.episode = 0
    
    def encode_state(self, route: Tuple[int, ...], time_of_day: int) -> str:
        """
        Encode route and time into a state representation.
        
        Args:
            route: Tuple of location indices
            time_of_day: Hour of day (0-23)
        
        Returns:
            State string for Q-table
        """
        return f"route_{route}_time_{time_of_day}"
    
    def calculate_reward(
        self,
        actual_distance: float,
        estimated_distance: float,
        actual_time: float,
        estimated_time: float,
        is_on_time: bool
    ) -> float:
        """
        Calculate reward for a route execution.
        
        Reward components:
        - Distance efficiency (negative if longer than estimated)
        - Time efficiency (negative if slower than estimated)
        - On-time bonus (positive if all pickups on time)
        
        Args:
            actual_distance: Actual travel distance
            estimated_distance: Estimated distance
            actual_time: Actual travel time
            estimated_time: Estimated time
            is_on_time: Whether all pickups were on time
        
        Returns:
            Reward value (higher is better)
        """
        # Distance penalty: -1 for every 10% over estimate
        distance_ratio = actual_distance / max(estimated_distance, 1)
        distance_reward = -max(0, (distance_ratio - 1) * 10)
        
        # Time penalty: -1 for every 10% over estimate
        time_ratio = actual_time / max(estimated_time, 1)
        time_reward = -max(0, (time_ratio - 1) * 10)
        
        # On-time bonus
        on_time_bonus = 50 if is_on_time else 0
        
        # Efficiency bonus (negative if too much deviation)
        efficiency_bonus = 10 if (distance_ratio < 1.1 and time_ratio < 1.1) else 0
        
        total_reward = distance_reward + time_reward + on_time_bonus + efficiency_bonus
        return total_reward
    
    def select_action(
        self,
        state: str,
        possible_actions: List[Tuple[int, ...]],
        use_exploration: bool = True
    ) -> Tuple[int, ...]:
        """
        Select next action using epsilon-greedy strategy.
        
        Args:
            state: Current state
            possible_actions: List of possible route actions
            use_exploration: Whether to use exploration
        
        Returns:
            Selected action (route)
        """
        if use_exploration and np.random.random() < self.epsilon:
            # Exploration: random action
            action = possible_actions[np.random.randint(len(possible_actions))]
        else:
            # Exploitation: best known action
            q_values = [self.q_table.get((state, action), 0) for action in possible_actions]
            max_q = max(q_values)
            best_actions = [
                action for action, q in zip(possible_actions, q_values) if q == max_q
            ]
            action = best_actions[np.random.randint(len(best_actions))]
        
        return action
    
    def update_q_value(
        self,
        state: str,
        action: Tuple[int, ...],
        reward: float,
        next_state: str,
        possible_next_actions: List[Tuple[int, ...]]
    ) -> None:
        """
        Update Q-value using Q-Learning update rule.
        
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Resulting state
            possible_next_actions: Possible actions in next state
        """
        current_q = self.q_table.get((state, action), 0)
        
        if possible_next_actions:
            next_q_values = [
                self.q_table.get((next_state, next_action), 0)
                for next_action in possible_next_actions
            ]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0
        
        # Q-Learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
    
    def learn_from_episode(
        self,
        routes_data: List[Dict],
        time_of_day: int
    ) -> float:
        """
        Learn from a complete episode (day of operations).
        
        Args:
            routes_data: List of route dictionaries with execution metrics:
                - route: Tuple of location indices
                - estimated_distance: Estimated distance
                - actual_distance: Actual distance
                - estimated_time: Estimated time
                - actual_time: Actual time
                - is_on_time: Whether all pickups were on time
            time_of_day: Hour of day
        
        Returns:
            Episode reward (average reward across all routes)
        """
        state = self.encode_state(tuple(range(len(routes_data))), time_of_day)
        episode_reward = 0
        
        for route_data in routes_data:
            action = tuple(route_data['route'])
            
            # Calculate reward
            reward = self.calculate_reward(
                route_data['actual_distance'],
                route_data['estimated_distance'],
                route_data['actual_time'],
                route_data['estimated_time'],
                route_data['is_on_time']
            )
            
            episode_reward += reward
            
            # Update Q-value
            next_state = self.encode_state(action, (time_of_day + 1) % 24)
            self.update_q_value(state, action, reward, next_state, [action])
            
            # Store traffic data
            self._store_traffic_data(action, route_data)
        
        # Decay exploration rate
        self.epsilon *= self.epsilon_decay
        
        self.episode += 1
        self.reward_history.append(episode_reward / max(len(routes_data), 1))
        
        return episode_reward / max(len(routes_data), 1)
    
    def _store_traffic_data(
        self,
        route: Tuple[int, ...],
        route_data: Dict
    ) -> None:
        """
        Store actual traffic/performance data for later analysis.
        
        Args:
            route: Route identifier
            route_data: Actual performance data
        """
        route_key = str(route)
        if route_key not in self.traffic_history:
            self.traffic_history[route_key] = []
        
        self.traffic_history[route_key].append({
            'timestamp': datetime.now().isoformat(),
            'actual_distance': route_data['actual_distance'],
            'actual_time': route_data['actual_time'],
            'estimated_distance': route_data['estimated_distance'],
            'estimated_time': route_data['estimated_time']
        })
    
    def predict_travel_time(
        self,
        route: Tuple[int, ...],
        time_of_day: int,
        base_time: float
    ) -> float:
        """
        Predict travel time using learned patterns.
        
        Args:
            route: Route identifier
            time_of_day: Hour of day
            base_time: Base estimated time
        
        Returns:
            Adjusted predicted time in minutes
        """
        # Look up historical data for similar routes/times
        adjustments = []
        
        for stored_route, history in self.traffic_history.items():
            # Simple similarity: same route or time
            if stored_route == str(route):
                adjustments.extend([
                    h['actual_time'] / max(h['estimated_time'], 1) for h in history
                ])
        
        if adjustments:
            # Use median multiplier from history
            avg_multiplier = np.median(adjustments)
            return base_time * avg_multiplier
        
        return base_time
    
    def get_learned_policy(self) -> Dict:
        """
        Extract learned policy from Q-table.
        
        Returns:
            Dictionary mapping states to best actions
        """
        policy = {}
        
        # Group by state
        states = set(state for state, _ in self.q_table.keys())
        
        for state in states:
            actions_q = {
                action: self.q_table[(state, action)]
                for action in set(action for s, action in self.q_table.keys() if s == state)
            }
            
            if actions_q:
                best_action = max(actions_q, key=actions_q.get)
                policy[state] = {
                    'action': str(best_action),
                    'q_value': float(actions_q[best_action])
                }
        
        return policy
    
    def save_learning(self, filepath: str) -> None:
        """
        Save learned Q-table and traffic history.
        
        Args:
            filepath: Path to save learning data
        """
        learning_data = {
            'q_table': {
                f"{state}_{action}": q_value
                for (state, action), q_value in self.q_table.items()
            },
            'traffic_history': self.traffic_history,
            'reward_history': self.reward_history,
            'episode': self.episode,
            'epsilon': float(self.epsilon)
        }
        
        with open(filepath, 'w') as f:
            json.dump(learning_data, f, indent=2)
        
        print(f"Learning data saved to {filepath}")
    
    def load_learning(self, filepath: str) -> None:
        """
        Load previously saved learning data.
        
        Args:
            filepath: Path to load learning data
        """
        with open(filepath, 'r') as f:
            learning_data = json.load(f)
        
        # Reconstruct Q-table
        self.q_table = {}
        for key, q_value in learning_data['q_table'].items():
            parts = key.rsplit('_', 1)
            state = parts[0]
            action = eval(parts[1]) if len(parts) > 1 else None
            if action:
                self.q_table[(state, action)] = q_value
        
        self.traffic_history = learning_data.get('traffic_history', {})
        self.reward_history = learning_data.get('reward_history', [])
        self.episode = learning_data.get('episode', 0)
        self.epsilon = learning_data.get('epsilon', self.epsilon)
        
        print(f"Learning data loaded from {filepath}")
    
    def get_learning_statistics(self) -> Dict:
        """
        Get statistics about the learning progress.
        
        Returns:
            Dictionary with learning metrics
        """
        if not self.reward_history:
            return {}
        
        rewards = np.array(self.reward_history)
        
        return {
            'episodes': self.episode,
            'avg_reward': float(np.mean(rewards)),
            'max_reward': float(np.max(rewards)),
            'min_reward': float(np.min(rewards)),
            'recent_avg_reward': float(np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)),
            'q_table_size': len(self.q_table),
            'current_epsilon': float(self.epsilon)
        }


if __name__ == "__main__":
    # Example usage
    rl = RLOptimizer(learning_rate=0.1, discount_factor=0.9)
    
    # Simulate learning from route executions
    for episode in range(5):
        routes_data = [
            {
                'route': [1, 2, 3, 4, 5],
                'estimated_distance': 25.0,
                'actual_distance': 24.5,
                'estimated_time': 45.0,
                'actual_time': 44.0,
                'is_on_time': True
            },
            {
                'route': [6, 7, 8, 9],
                'estimated_distance': 20.0,
                'actual_distance': 22.0,
                'estimated_time': 35.0,
                'actual_time': 38.0,
                'is_on_time': False
            }
        ]
        
        reward = rl.learn_from_episode(routes_data, time_of_day=8)
        print(f"Episode {episode + 1}: Average Reward = {reward:.2f}")
    
    # Get statistics
    stats = rl.get_learning_statistics()
    print(f"\nLearning Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
