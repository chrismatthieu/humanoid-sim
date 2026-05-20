from setuptools import setup

package_name = "humanoid_pose_estimator"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="humanoid-sim",
    maintainer_email="humanoid-sim@example.com",
    description="MediaPipe + RealSense depth 3D pose estimator for the G1 mimic demo.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "pose_estimator_node = humanoid_pose_estimator.pose_estimator_node:main",
        ],
    },
)
