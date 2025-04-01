import ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
import gym

# Step 1: Define your custom environment
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)  # Example: 4 actions (up, down, left, right)
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(3,))  # Example: 3D observation space

    def reset(self):
        return [0, 0, 0]  # Example: initial state

    def step(self, action):
        # Example logic for the environment step
        state = [0, 0, 0]  # Updated state after action
        reward = 1  # Example reward
        done = False  # Whether the episode is done
        return state, reward, done, {}

# Register your environment
register_env("custom_env", lambda config: CustomEnv())

# Step 2: Set up Ray and PPO Trainer
ray.init(ignore_reinit_error=True)

trainer = ppo.PPOTrainer(env="custom_env")

# Step 3: Train the model
for _ in range(100):  # Run for 100 iterations
    result = trainer.train()
    print(f"Iteration: {_}, Reward: {result['episode_reward_mean']}")

# Step 4: Save the trained model
trainer.save("ppo_model")

# Step 5: Evaluate the model (optional)
policy = trainer.get_policy()
state = [0, 0, 0]
action = policy.compute_single_action(state)
print(f"Action taken: {action}")
