# Drone Hair Dryers

Have you ever come out of the shower and needed your hair drying only to realise you're too lazy to use a towel or even a hair dryer? Well this project is for you!

Originally intended as a silly idea, this project utilises the downdrafts of a drone (quadcopter) to act as a hair drier. Don't try this at home: it is a dumb idea!

# The Thought Process

The process would look something like this:

1. Boot up the drone
2. The drone would detect the face of the person it is to hover over
3. It would then begin tracking their face and use their position (and its position relative to them)
4. The positional information is relayed to the flight computer to generate a flight path
5. Once above their head it would maintain a fix as the target moves for maximum drying potential

We will be utilising dronekit where possible, a Python wrapper for Ardupilot. That being said, dronekit relies heavily on
a GPS fixture, something not possible in all buildings. Therefore this project will rely on MAVLink messages with a
relative reference frame instead. After all, you are the drone's whole world.

# Implementation Details
Computer vision is expensive for embedded boards. The vision processing will be done on a companion computer and sent to the flight computer via MAVlink messages so that the existing dronekit and Ardulink infrastructure can be used
Because the drone will be flying from a 'ground plane' or 'xy plane' overhead of the target into an 'xz plane'. Because of this conventional 2D object tracking won't be sufficient and 3D tracking is required.
Throw in the fact that the person won't stay stationary and we have both a moving camera and a moving target.

## Face Detection
At first Open CV's pre-trained Haar Cascade Classifier will be used to detect the face closest to the camera's centre

## Object Tracking
Initially Open CV's KCF algorithm was used because it provides an acceptable compromise between speed and accuracy, and because occlusions won't be considered.

KCF is a two dimensional tracker, however, and to obtain the desired drone motion a three dimensional tracker is required such as the model devised by [G. Brazil, G. Pons-Moll, X. Liu, and B. Schiele](http://cvlab.cse.msu.edu/pdfs/brazil_pons-moll_liu_Schiele_eccv2020.pdf).
