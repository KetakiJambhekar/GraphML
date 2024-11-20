# Train RL Agent using PPO
model_rl = PPO("MlpPolicy", env, verbose=1)
model_rl.learn(total_timesteps=10000)

# Test RL Agent
obs = env.reset()
action, _states = model_rl.predict(obs)
new_positions, reward, done, info = env.step(action)
print(f"Final Reward (Minimized Crossings): {reward}")
