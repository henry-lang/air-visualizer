from typing import Optional
import cv2
import numpy as np
import os
from collections import deque
from enum import Enum, auto
from time import strftime

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


class RecentViewports:
    data: deque["Viewport"]
    size: int

    def __init__(self, size: int):
        self.data = deque()
        self.size = size

    def add(self, viewport: "Viewport"):
        self.data.append(viewport)
        if len(self.data) > self.size:
            self.data.popleft()

    def has_average(self) -> bool:
        return len(self.data) > 0

    def average(self) -> "Viewport":
        size = Marker(
            (self.size, self.size),
            (self.size, self.size),
            (self.size, self.size),
            (self.size, self.size),
        )
        z = (0, 0)
        avg = Viewport(
            Marker(z, z, z, z),
            Marker(z, z, z, z),
            Marker(z, z, z, z),
            Marker(z, z, z, z),
        )

        for v in self.data:
            avg.tl += v.tl / size
            avg.tr += v.tr / size
            avg.br += v.br / size
            avg.bl += v.bl / size

        return avg


class Marker:
    def __init__(self, tl, tr, br, bl):
        self.tl = tl
        self.tr = tr
        self.br = br
        self.bl = bl

    def from_corners(corners) -> "Marker":
        tl = tuple(map(int, corners[0][0]))
        tr = tuple(map(int, corners[0][1]))
        br = tuple(map(int, corners[0][2]))
        bl = tuple(map(int, corners[0][3]))
        return Marker(tl, tr, br, bl)

    def __add__(self, o: "Marker") -> "Marker":
        return Marker(
            (self.tl[0] + o.tl[0], self.tl[1] + o.tl[1]),
            (self.tr[0] + o.tr[0], self.tr[1] + o.tr[1]),
            (self.br[0] + o.br[0], self.br[1] + o.br[1]),
            (self.bl[0] + o.bl[0], self.bl[1] + o.bl[1]),
        )

    def __truediv__(self, factor: "Marker") -> "Marker":
        return Marker(
            (self.tl[0] / factor.tl[0], self.tl[1] / factor.tl[1]),
            (self.tr[0] / factor.tr[0], self.tr[1] / factor.tr[1]),
            (self.br[0] / factor.br[0], self.br[1] / factor.br[1]),
            (self.bl[0] / factor.bl[0], self.bl[1] / factor.bl[1]),
        )


class Viewport:
    def __init__(self, tl, tr, br, bl):
        self.tl = tl
        self.tr = tr
        self.br = br
        self.bl = bl


def detect_viewport(gray) -> Optional[Viewport]:
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    tl = tr = br = bl = None
    if not ids is None:
        for i in range(len(ids)):
            id = ids[i][0]
            marker = Marker.from_corners(corners[i])
            if id == 0:
                tl = marker
            if id == 1:
                tr = marker
            if id == 2:
                br = marker
            if id == 3:
                bl = marker
    if tl is None or tr is None or br is None or bl is None:
        return None
    return Viewport(tl, tr, br, bl)


def extract_viewport_area(frame, viewport):
    src = np.float32(
        [
            viewport.tl.tl,
            viewport.tr.tr,
            viewport.br.br,
            viewport.bl.bl,
        ]
    )
    width = frame.shape[1]
    height = frame.shape[0]
    dest = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    mat = cv2.getPerspectiveTransform(src, dest)
    result = cv2.warpPerspective(frame, mat, (width, height))
    return result


def diff(a: cv2.Mat, b: cv2.Mat) -> cv2.Mat:
    return cv2.absdiff(a, b)


def main():
    now = strftime("%Y-%m-%HT%H:%M:%S")
    images_path = f"out/{now}"
    os.makedirs(images_path, exist_ok=True)
    num_frames = 0
    recent_viewports = RecentViewports(10)

    print(f"Images Path: {images_path}")

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Failed to initialize capture")
        exit(1)

    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()
        # print(f"Reading frame {num_frames}")
        num_frames += 1

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        viewport = detect_viewport(gray)

        if viewport:
            recent_viewports.add(viewport)
            marker_area = extract_viewport_area(gray, viewport)
            cv2.imshow("frame", marker_area)
        else:
            if recent_viewports.has_average():
                average = recent_viewports.average()
                cv2.imshow("frame", extract_viewport_area(gray, average))
            else:
                cv2.imshow("frame", gray)

        if cv2.waitKey(1) == ord("q"):
            break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
