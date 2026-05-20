from setuptools import setup

package_name = "humanoid_retargeter"

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
    description="Geometric upper-body retargeter for the Unitree G1.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "retargeter_node = humanoid_retargeter.retargeter_node:main",
        ],
    },
)
