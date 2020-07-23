import sys
from math import sqrt, floor

import cv2

# This code combines face detection with open CV's code for object tracking to detect and track a face automatically

(major_ver, minor_ver, subminor_ver) = (cv2.getVersionString()).split('.')


draw_debug_elements = True
faces_for_debug = []


def draw_face_debug_objects(frame):
    centre = get_centre(frame)

    for (x, y, w, h) in faces_for_debug:
        # Bounding box
        p1 = (x, y)
        p2 = (x + w, y + h)
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

        # Line to centre
        line_p1 = (int(x + (w / 2)), int(y + (h / 2)))
        cv2.line(frame, line_p1, centre, (0, 127, 255))


def save_face_for_debug(face_bounding_box):
    faces_for_debug.append(face_bounding_box)


def get_centre(frame):
    (w, h) = get_frame_dimensions(frame)
    return (
        floor(w / 2),
        floor(h / 2)
    )


def get_frame_dimensions(frame) -> [int, int]:
    try:
        return frame.shape[1], frame.shape[0]
    except AttributeError:
        # yes, it's possible to be more defensive but when writing this only Tuples and numpy ndarrays are used
        if type(frame) is tuple:
            return frame[0], frame[1]
        elif type(frame) is None:
            raise TypeError("Frame cannot be of None type")


def choose_face(face_bounding_boxes, frame):
    """
    Tries to choose a face when there could be multiple. It does this by considering the face scores, as well as
    considering the closest to the frame_centre of the camera, seeing as the person of interest wants to be in the focus.

    :param face_bounding_boxes: all possible faces detected
    :param frame: the frame, used for determining the centre point
    :return: the face chosen for tracking
    """

    if len(face_bounding_boxes) == 0:
        raise ValueError("At least one bounding box must be provided")

    frame_centre = get_centre(frame)

    # closest bounding box to the frame_centre
    nearest_candidate = (999999999, None)

    for i in range(0, len(face_bounding_boxes)):
        (x, y, w, h) = face_bounding_boxes[i]
        box_centre = (x + (w/2), y + (h/2))

        # simple hypotenuse distance finding
        dist = sqrt(pow(box_centre[0] - frame_centre[0], 2) + pow(box_centre[1] - frame_centre[1], 2))

        save_face_for_debug(face_bounding_boxes[i])

        if dist < nearest_candidate[0]:
            nearest_candidate = dist, (x, y, w, h)

    return nearest_candidate[1]


def detect_face(video_capture):
    """
    Search for a face until one is found

    :param video_capture: the video capture source which can be read from to detect a face
    :return: the boundary box which contains the detected face
    """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ok, frame = video_capture.read()
        if not ok:
            break

        greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(greyscale, 1.1, 4)
        print("Detected %d faces." % len(faces))

        if len(faces) != 0:
            try:
                frame_dimensions = get_frame_dimensions(frame)
                return choose_face(faces, frame_dimensions)
            except ValueError:
                # there were faces but none were chosen
                continue

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit()


def track_face():
    # Set up tracker.
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    # KCF has the best of BOOSTING and MIL, a good starting point
    tracker_type = tracker_types[2]

    if int(minor_ver) < 3:
        sys.exit(1)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        else:
            print("Unknown tracker type, please use one of the following: %s" % tracker_types)
            sys.exit(2)

    # Read video
    video = cv2.VideoCapture(0)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit(3)

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        sys.exit(3)

    # Define an initial bounding box by detecting the face of interest
    face_bbox = detect_face(video)

    # Initialize tracker. Tracking is more performant than object detection, hence, it is used once the face has been
    # detected
    ok = tracker.init(frame, face_bbox)
    if not ok:
        print("Tracker failed to initialise")
        sys.exit(4)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        if draw_debug_elements:
            draw_face_debug_objects(frame)

        # Showing tracking information on screen
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    track_face()
