import sys
import cv2

# This code combines face detection with open CV's code for object tracking to detect and track a face automatically

(major_ver, minor_ver, subminor_ver) = (cv2.getVersionString()).split('.')


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
        print("Detected %d faces. Expects 1 face" % len(faces))

        for (x, y, w, h) in faces:
            # todo determine which face to choose when there are multiple
            return x, y, w, h

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

    # Read video
    video = cv2.VideoCapture(0)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit(2)

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        sys.exit(2)

    # Define an initial bounding box by detecting the face of interest
    bbox = detect_face(video)

    # Initialize tracker. Tracking is more performant than object detection, hence, it is used once the face has been
    # detected
    ok = tracker.init(frame, bbox)

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
