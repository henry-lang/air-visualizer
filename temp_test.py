from collections import deque
import os
from time import strftime
from typing import Optional
import cv2
import numpy as np


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
        s = len(self.data)
        size = Marker(
            (s, s),
            (s, s),
            (s, s),
            (s, s),
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


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
recent = RecentViewports(60)
capture = cv2.VideoCapture(0)
now = strftime("%Y-%m-%HT%H_%M_%S")
images_path = f"out/{now}"
os.makedirs(images_path, exist_ok=True)

if not capture.isOpened():
    print("Failed to open capture")
    exit(1)


def get_frame() -> cv2.Mat:
    ret, frame = capture.read()
    if not ret:
        print("Failed to get frame")
        exit(1)
    bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return bw


# Record a couple frames to warm up camera
for i in range(60):
    frame = get_frame()
    if cv2.waitKey(1) == ord("q"):
        break
    viewport = detect_viewport(frame)
    if not viewport is None:
        print("Found viewport, adding it to recents")
        recent.add(viewport)

if not recent.has_average():
    print("Failed to determine viewport, make sure it is in view")
    exit(1)

average_viewport = recent.average()
first = extract_viewport_area(get_frame(), average_viewport)
num_frames = 0

while True:
    frame = extract_viewport_area(get_frame(), average_viewport)
    diff = cv2.absdiff(frame, first)
    cv2.imshow("Preview", diff)
    cv2.imwrite(f"{images_path}/{num_frames}.png", diff)
    num_frames += 1
