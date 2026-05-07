import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/turtlebot3/ros2_ws/src/gymnasium_line_follower/gym-line-follower/install/gym_line_follower'
