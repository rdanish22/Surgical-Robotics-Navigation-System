# Surgical-Robotics-Navigation-System

Overview: This project implements a high-precision navigation system for stereotactic surgery, bridging the gap between electromagnetic (EM) tracking and preoperative CT imaging. I developed the algorithmic framework to "de-warp" distorted sensor data and achieve clinical-grade accuracy. Key Technical Implementations3D Point-to-Point Registration: Developed algorithms to compute transformation frames ($F_{reg}$) between tracker bases and CT coordinate systems. Pivot Calibration: Built robust routines to solve for probe tip coordinates, ensuring 95%+ accuracy in tool tracking. Distortion Correction: Integrated polynomial fitting models to mitigate random noise (up to 0.3 mm) and uncharacterized EM distortions. Validation Pipeline: Engineered automated testing protocols to verify algorithmic correctness against "debug" datasets (e.g., pal-debug-f)

The source files are:
Alogrithms.py: contains all of the main algorithms (registration, calibration, computing C_expected), answers question 2 and 3
Alogrithms_test.py: tests Alogrithms file
Cartesian_Math.py: contains functions for rotations, frame transformations, and rotation generation, answers question 1
Cartesian_Math_test.py: tests Cartesian_Math file
Frame.py: File that contains the Frame class that has an inverse function and stores R an P
Frame_test.py: tests Frame class
IO.py: files to handle all the reading and writing of files. uses base path "PA1StudentData/..." to open all files (must be in same level), writes to OUTPUT which is automatically generated upon running main
Test.py: runs all the test files at once
main.py: creates an output folder with all the output files and prints error analysis values to terminal. Answers questions 4, 5, 6




To run the program, just run the main.py file
