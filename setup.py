from setuptools import find_packages, setup

setup(
    name="gym_line_follower",
    version="0.2.0",
    packages=find_packages(include=["gym_line_follower", "gym_line_follower.*"]),
    include_package_data=True,
    install_requires=[
        "gymnasium>=0.29.0",
        "pybullet",
        "opencv-python",
        "shapely",
        "numpy",
    ],
    author="Nejc Planinsek",
    author_email="planinseknejc@gmail.com",
    description="Line follower simulator environment.",
)
