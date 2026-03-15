# Surgical-Robotics-Navigation-System

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
