import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset()

print("Initial observation:", obs)
print("Environment created successfully!")

env.close()