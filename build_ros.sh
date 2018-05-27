echo "Building ROS nodes"

cd Examples/ROS/ORB_SLAM2
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j
