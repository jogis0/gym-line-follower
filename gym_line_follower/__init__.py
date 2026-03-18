from gymnasium.envs.registration import register

register(
    id='LineFollower-v0',
    entry_point='gym_line_follower.envs:LineFollowerEnv',
    reward_threshold=700
)

register(
    id='LineFollowerCamera-v0',
    entry_point='gym_line_follower.envs:LineFollowerCameraEnv',
    reward_threshold=700
)

register(
    id='TurtleBot3LineFollower-v0',
    entry_point='gym_line_follower.envs:TurtleBot3LineFollowerEnv',
    reward_threshold=700
)