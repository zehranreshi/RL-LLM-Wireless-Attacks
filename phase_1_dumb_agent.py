import random
from simulation_environment import mock_run_simulation_and_get_reward

class DumbAgent:
    def generate_action(self):
        config = {
            "center_frequency": random.uniform(3.5e9, 3.7e9),
            "bandwidth": random.choice([10e6, 20e6, 40e6]),
            "tx_gain": random.uniform(50, 90)
        }
        return config

    def update(self, config, reward):
        pass

def run_training_loop():
    agent = DumbAgent()
    num_episodes = 20
    
    print("--- Starting Phase 1 Training with DumbAgent ---")

    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        
        action_config = agent.generate_action()
        reward = mock_run_simulation_and_get_reward(action_config)
        agent.update(action_config, reward)

    print("--- Phase 1 Training Complete ---")

if __name__ == '__main__':
    run_training_loop()