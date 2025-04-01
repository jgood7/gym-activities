from envs import GridWorldEnv
from stable_baselines3 import PPO

env = GridWorldEnv(render_mode='human')  # Reinitialize environment
model = PPO.load("ppo_gridworld")  # Load trained model

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # Print grid state
