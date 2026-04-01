import gymnasium as gym
import gym_line_follower

# Gymnasium render_mode API: call env.render() with no args.
env = gym.make("TurtleBot3LineFollower-v0", gui=True, render_mode="gui")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # cmd_vel: [vx, wz]
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()