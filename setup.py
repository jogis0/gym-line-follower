from setuptools import setup

setup(name='gym_line_follower',
      version='0.2.0',
      install_requires=['gymnasium>=0.29.0',
                        'pybullet', 'opencv-python', 'shapely', 'numpy'],
      author="Nejc Planinsek",
      author_email="planinseknejc@gmail.com",
      description="Line follower simulator environment.",
      )
