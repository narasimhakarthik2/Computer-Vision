# Parking Space Counter

## Overview
This project implements a parking space detection system using computer vision techniques. It analyzes video footage of a parking lot to determine which parking spaces are occupied and which are free.

## Features
- Real-time detection of free and occupied parking spaces
- Visual representation of parking space status
- Continuous monitoring with video feed
- Display of free space count

## Requirements
- Python 3.x
- OpenCV (cv2)
- cvzone
- numpy
- networkx

## How it works
1. The script reads pre-defined parking space positions from the `CarParkPos` file.
2. It processes each frame of the video through several image processing steps:
   - Convert to grayscale
   - Apply Gaussian blur
   - Use adaptive thresholding
   - Apply median blur
   - Dilate the image
3. For each defined parking space:
   - It crops the processed image to the parking space area
   - Counts non-zero pixels in the cropped area
   - Determines if the space is free or occupied based on the pixel count
4. The results are visualized on the original frame and displayed

## Customization
- Adjust the `width` and `height` variables to match the size of your parking spaces
- Modify the threshold value (`850`) in the `checkParkingSpace` function to fine-tune detection sensitivity

## Limitations
- The system assumes a fixed camera position
- Performance may vary under different lighting conditions or with different parking lot layouts
- The current setup requires manual definition of parking space positions

## Future Improvements
- Implement automatic parking space detection
- Add support for multiple camera feeds

