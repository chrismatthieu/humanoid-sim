cmake_minimum_required(VERSION 3.16)
project(g1_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf meshes scripts
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/process_g1_urdf.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
