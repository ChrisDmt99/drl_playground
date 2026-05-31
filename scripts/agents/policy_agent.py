import numpy as np
import core.policies as policies 

class PolicyAgent:
    def __init__(self, seed, num_states, num_actions, policy_name, policy_params):
        # Set the random seed for reproducibility
        np.random.seed(seed)

        # Set the agent's parameters
        self.num_states = num_states
        self.num_actions = num_actions
        self.policy_name = policy_name
        self.policy_params = policy_params

        # We generate a random Q-table for demonstration purposes. In a real implementation, this would be learned over time.
        self.q_table = np.random.uniform(low=0.1, high=1.0, size=(num_states, num_actions))

        # Set policy-specific parameters
        if self.policy_name == "epsilon_greedy":
            self.init_epsilon = policy_params.get("init_epsilon", 1.0)
            self.min_epsilon = policy_params.get("min_epsilon", 0.01)
            self.decay_rate = policy_params.get("decay_rate", 0.9) 
            self.epsilon = self.init_epsilon        

        elif self.policy_name == "softmax":
            self.init_temperature = policy_params.get("init_temperature", 1.0)
            self.min_temperature = policy_params.get("min_temperature", 0.01)
            self.decay_rate = policy_params.get("decay_rate", 0.9)
            self.temperature = self.init_temperature  

        elif self.policy_name == "ucb":
            self.c = policy_params.get("c", 1.0)
            self.state_count = np.zeros(self.num_states)
            self.action_counts = np.zeros((self.num_states, self.num_actions))

        elif self.policy_name == "thompson_sampling":
            self.alpha = policy_params.get("alpha", 1.0)
            self.beta = policy_params.get("beta", 1.0)
            self.state_count = np.zeros(self.num_states)
            self.action_counts = np.zeros((self.num_states, self.num_actions))

        else:
            raise ValueError(f"Invalid policy name: {policy_name}")

    def select_action(self, state, action_space):
        q_values = self.q_table[state]
        
        # Use the specified policy to select an action based on the Q-table
        if self.policy_name == "epsilon_greedy":
            # Implement epsilon-greedy action selection
            return policies.epsilon_greedy_policy(q_values, self.epsilon, action_space)
        
        elif self.policy_name == "softmax":
            # Implement softmax action selection
            return policies.softmax_policy(q_values, self.temperature)
        
        elif self.policy_name == "ucb":
            # Implement Upper Confidence Bound (UCB) action selection 
            action, reason = policies.ucb_policy(q_values, self.c, self.state_count[state], self.action_counts[state])
            self.state_count[state] += 1       
            self.action_counts[state][action] += 1
            return action, reason
        
        elif self.policy_name == "thompson_sampling":
            # Implement Thompson Sampling action selection
            action, reason = policies.thompson_sampling_policy(q_values, self.alpha, self.beta, self.action_counts[state])
            self.state_count[state] += 1       
            self.action_counts[state][action] += 1
            return action, reason
        
        else:
            raise ValueError("Unsupported policy")
        
    def end_of_episode(self):
        """
        Update the agent's parameters at the end of each episode (e.g., decay epsilon for epsilon-greedy).
        """
        if self.policy_name == "epsilon_greedy":
            self.epsilon = max(1e-8, max(self.min_epsilon, self.epsilon * self.decay_rate))
            
        elif self.policy_name == "softmax":
            self.temperature = max(1e-8, max(self.min_temperature, self.temperature * self.decay_rate))