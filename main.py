import cv2
import numpy as np


class BackgroundCaptureState:
    RELEASED = 0
    CAPTURING = 1
    CAPTURED = 2


class Marker:
    def __init__(self, topLeft, topRight, bottomRight, bottomLeft):
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomRight = bottomRight
        self.bottomLeft = bottomLeft

    def divide(self, value):
        return Marker(
            (self.topLeft[0] / value, self.topLeft[1] / value),
            (self.topRight[0] / value, self.topRight[1] / value),
            (self.bottomRight[0] / value, self.bottomRight[1] / value),
            (self.bottomLeft[0] / value, self.bottomLeft[1] / value),
        )

    def __add__(self, other):
        return Marker(
            (self.topLeft[0] + other.topLeft[0], self.topLeft[1] + other.topLeft[1]),
            (
                self.topRight[0] + other.topRight[0],
                self.topRight[1] + other.topRight[1],
            ),
            (
                self.bottomRight[0] + other.bottomRight[0],
                self.bottomRight[1] + other.bottomRight[1],
            ),
            (
                self.bottomLeft[0] + other.bottomLeft[0],
                self.bottomLeft[1] + other.bottomLeft[1],
            ),
        )


class QuadMarkers:
    def __init__(
        self, topLeftMarker, topRightMarker, bottomRightMarker, bottomLeftMarker
    ):
        self.topLeftMarker = topLeftMarker
        self.topRightMarker = topRightMarker
        self.bottomRightMarker = bottomRightMarker
        self.bottomLeftMarker = bottomLeftMarker


def convert_color_to_gray(src):
    return cv2.cvtColor(src, cv2.COLOR_RGBA2GRAY)


def get_mat_from_image(image):
    yuvMat = np.zeros((image.height + image.height // 2, image.width), np.uint8)
    nv21 = np.frombuffer(
        image.planes[0].buffer + image.planes[1].buffer + image.planes[2].buffer,
        dtype=np.uint8,
    )
    yuvMat[: nv21.shape[0]] = nv21.reshape((yuvMat.shape[0], yuvMat.shape[1]))
    rgbaMat = cv2.cvtColor(yuvMat, cv2.COLOR_YUV2RGBA_NV21, 4)
    return rgbaMat


def absdiff(src1, src2):
    return cv2.absdiff(src1, src2)


def emphasize_contrast(src):
    return cv2.multiply(src, 6.0)


def draw_fps(frame, fps):
    cv2.putText(
        frame, fps, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
    )


def show_image(frame):
    cv2.imshow("Image", frame)


def draw_marker_circle(frame, point):
    color = (0, 0, 255)
    cv2.circle(frame, tuple(map(int, point)), 12, color, -1)


def draw_edges(frame, quad_markers):
    points = np.array(
        [
            quad_markers.topLeftMarker.topLeft,
            quad_markers.topRightMarker.topRight,
            quad_markers.bottomRightMarker.bottomRight,
            quad_markers.bottomLeftMarker.bottomLeft,
        ],
        dtype=np.int32,
    )
    color = (0, 0, 255)
    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)


def extract_marker_area(frame, quad_markers):
    src = np.float32(
        [
            quad_markers.topLeftMarker.topLeft,
            quad_markers.topRightMarker.topRight,
            quad_markers.bottomRightMarker.bottomRight,
            quad_markers.bottomLeftMarker.bottomLeft,
        ]
    )
    width = frame.shape[1]
    height = frame.shape[0]
    dest = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    mat = cv2.getPerspectiveTransform(src, dest)
    result = cv2.warpPerspective(frame, mat, (width, height))
    return result


def detect_quad_markers(gray):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    print(corners, ids)
    markers = {}
    for i in range(len(ids)):
        id = ids[i][0]
        markerCorners = corners[i][0]
        marker = create_marker_from_corners(markerCorners)
        markers[id] = marker
    return markers


def create_marker_from_corners(markerCorners):
    topLeft = tuple(map(int, markerCorners[0]))
    topRight = tuple(map(int, markerCorners[1]))
    bottomRight = tuple(map(int, markerCorners[2]))
    bottomLeft = tuple(map(int, markerCorners[3]))
    return Marker(topLeft, topRight, bottomRight, bottomLeft)


def average_recent_quad_markers(recentQuadMarkersQueue):
    size = len(recentQuadMarkersQueue)
    p = (0.0, 0.0)
    averageTopLeftMarker = (
        averageTopRightMarker
    ) = averageBottomRightMarker = averageBottomLeftMarker = p
    for quadMarkers in recentQuadMarkersQueue:
        averageTopLeftMarker += quadMarkers.topLeftMarker.topLeft
        averageTopRightMarker += quadMarkers.topRightMarker.topRight
        averageBottomRightMarker += quadMarkers.bottomRightMarker.bottomRight
        averageBottomLeftMarker += quadMarkers.bottomLeftMarker.bottomLeft
    return QuadMarkers(
        Marker(averageTopLeftMarker, p, p, p),
        Marker(averageTopRightMarker, p, p, p),
        Marker(averageBottomRightMarker, p, p, p),
        Marker(averageBottomLeftMarker, p, p, p),
    )


def add_to_recent_quad_markers(recentQuadMarkersQueue, quadMarkers):
    recentQuadMarkersQueue.append(quadMarkers)
    if len(recentQuadMarkersQueue) > 5:
        recentQuadMarkersQueue.pop(0)


def analyze_laminar_flow(
    image, backgroundCaptureState, recentQuadMarkersQueue, background
):
    originalFrame = get_mat_from_image(image)
    gray = convert_color_to_gray(originalFrame)
    if backgroundCaptureState == BackgroundCaptureState.CAPTURING:
        quadMarkers = detect_quad_markers(gray)
        if quadMarkers:
            add_to_recent_quad_markers(recentQuadMarkersQueue, quadMarkers)
            if len(recentQuadMarkersQueue) >= 10:
                background = extract_marker_area(
                    gray, average_recent_quad_markers(recentQuadMarkersQueue)
                )
                backgroundCaptureState = BackgroundCaptureState.CAPTURED
    elif backgroundCaptureState == BackgroundCaptureState.RELEASED:
        quadMarkers = detect_quad_markers(gray)
        if quadMarkers:
            draw_marker_circle(originalFrame, quadMarkers[0].topLeft)
            draw_marker_circle(originalFrame, quadMarkers[0].topRight)
            draw_marker_circle(originalFrame, quadMarkers[0].bottomRight)
            draw_marker_circle(originalFrame, quadMarkers[0].bottomLeft)
    elif backgroundCaptureState == BackgroundCaptureState.CAPTURED:
        quadMarkers = detect_quad_markers(gray)
        if quadMarkers:
            draw_marker_circle(originalFrame, quadMarkers[0].topLeft)
            draw_marker_circle(originalFrame, quadMarkers[0].topRight)
            draw_marker_circle(originalFrame, quadMarkers[0].bottomRight)
            draw_marker_circle(originalFrame, quadMarkers[0].bottomLeft)
            draw_edges(originalFrame, quadMarkers[0])
            markerArea = extract_marker_area(
                gray, average_recent_quad_markers(recentQuadMarkersQueue)
            )
            diff = absdiff(markerArea, background)
            contrast = emphasize_contrast(diff)
            show_image(contrast)


def main():
    # Initialize variables
    camera = cv2.VideoCapture(0)
    backgroundCaptureState = BackgroundCaptureState.RELEASED
    recentQuadMarkersQueue = []
    background = None

    while True:
        # Read frame from camera
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Frame", frame)

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Simulate button press for starting and stopping capture
        if backgroundCaptureState == BackgroundCaptureState.RELEASED:
            backgroundCaptureState = BackgroundCaptureState.CAPTURING
        else:
            backgroundCaptureState = BackgroundCaptureState.RELEASED

        # Simulate capturing background
        if backgroundCaptureState == BackgroundCaptureState.CAPTURING:
            quadMarkers = detect_quad_markers(gray)
            if quadMarkers:
                add_to_recent_quad_markers(recentQuadMarkersQueue, quadMarkers)
                if len(recentQuadMarkersQueue) >= 10:
                    background = extract_marker_area(
                        gray, average_recent_quad_markers(recentQuadMarkersQueue)
                    )
                    backgroundCaptureState = BackgroundCaptureState.CAPTURED

        # Simulate processing based on background
        if backgroundCaptureState == BackgroundCaptureState.CAPTURED:
            quadMarkers = detect_quad_markers(gray)
            if quadMarkers:
                draw_marker_circle(frame, quadMarkers[0].topLeft)
                draw_marker_circle(frame, quadMarkers[0].topRight)
                draw_marker_circle(frame, quadMarkers[0].bottomRight)
                draw_marker_circle(frame, quadMarkers[0].bottomLeft)
                draw_edges(frame, quadMarkers[0])
                markerArea = extract_marker_area(
                    gray, average_recent_quad_markers(recentQuadMarkersQueue)
                )
                diff = absdiff(markerArea, background)
                contrast = emphasize_contrast(diff)
                show_image(contrast)

        # Simulate waiting for user input (key press)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Release camera and close windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
