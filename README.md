# Anki Vector Localization in GPS Denied Environments

- Placed 8 ArUco markers around the perimeter of a 400mm × 400mm test area for localization.
- Fused wheel odometry and gyroscope data with periodic pose estimates from visual landmarks to continuously refine the robot’s estimated 2D position and orientation. 
- Implemented an Extended Kalman Filter to fuse nonlinear odometry motion model predictions with camera-based pose corrections, treating visual landmarks as GPS-like measurements. 
- Learned and applied frame transformations and 3D rotations to algin all sensor data into a consistent global coordinate frame.



[![Watch the video](http://i3.ytimg.com/vi/cFxt5h_BBY8/hqdefault.jpg)](https://www.youtube.com/watch?v=QOBxQInFjsE)
