# ece5510-parallel-panorama
ECE 5510 Group 2 Project
<br />
<br />
Usage:
`.\ParallelPanorama.exe <num-stitcher-worker-threads> <stitcher-mode | (manual) (opencv)> <top-level-img-directory-path>`
<br />
NOTE: Top-level image directory must contain subdirectories that contain images portions of
<br />&nbsp;the desired image to be stitched. Each subdirectory must be labeled with a numeric value
<br />&nbsp;representing the stitch position of the image. These numeric values start from 1, which
<br />&nbsp;is the furtherst left image to be stitched, and end in the number of subdirectories
<br />&nbsp;there are for images to be stitched, which is the furthest right image to be stitched.
<br />
<br />&nbsp;Within each subdirectory, there must be images present that are named with a numeric
<br />&nbsp;value that corresponds with the images in the other subdirectories to be stitched with.
<br />
<br />
Output:
```
Loaded images from - <subdirectory path images were successfully loaded from>
(Repeated for number of subdirectories)

Average time elapsed from last final stitched image: <average-time-elasped-in-ms>
(Repeated every 10 stitched images while images are being processed)
````
