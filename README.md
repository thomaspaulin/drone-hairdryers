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