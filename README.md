# Fast-Vision-Tracking
This is the vision tracking code for my Trash Bin Robot. 

## Goal
The goal of this program was to implement a fast tracking algorithm for my Trash Bin Robot project. This had to be optimized to be fast enough to have time to move towards incoming objects immediately after detection. 

## Implemation
In order to achieve the goal, this code needed to be run on a raspberry pi 5 using OpenCV. After researching various tracking algorithms it was concluded that 
Farenback Optical Flow was the best tracking algorithm for this application. Some optimzation implmented were the use of mulitithreading and hardware optimization (removing FPS limit) to increase the FPS.

## Results
OVER 100FPS ACCURATE OBJECT TRACKING WOOOOO
