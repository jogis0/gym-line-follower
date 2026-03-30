import gymnasium as gym
import gym_line_follower
import matplotlib.pyplot as plt

env = gym.make("TurtleBot3LineFollower-v0", gui=True, render_mode="gui")
obs, info = env.reset()

plt.ion()
fig, (ax_down, ax_pov) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Robot Camera Feed")

im_down = ax_down.imshow(obs)
ax_down.set_title("Down Camera (observation)")
ax_down.axis("off")

pov = env.unwrapped.follower_bot.get_pov_image()
im_pov = ax_pov.imshow(pov)
ax_pov.set_title("Forward POV")
ax_pov.axis("off")

plt.tight_layout()
plt.pause(0.001)

step = 0
total_reward = 0.0

for _ in range(2000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    step += 1
    total_reward += reward

    im_down.set_data(obs)
    pov = env.unwrapped.follower_bot.get_pov_image()
    im_pov.set_data(pov)
    fig.suptitle(f"Step {step} | Reward: {reward:.3f} | Total: {total_reward:.1f}")
    fig.canvas.flush_events()
    plt.pause(0.001)

    if terminated or truncated:
        obs, info = env.reset()
        step = 0
        total_reward = 0.0

env.close()
plt.ioff()
plt.show()
