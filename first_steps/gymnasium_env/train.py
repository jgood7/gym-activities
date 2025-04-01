from stable_baselines3 import PPO
from envs import GridWorldEnv  # Import your custom environment

# Create an instance of the environment
env = GridWorldEnv(render_mode='human'
                   )

# Train the agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_gridworld")