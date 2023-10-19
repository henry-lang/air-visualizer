import cv2 as cv

channel = 0
webcam = cv.VideoCapture(0)

delay = 100
lx, ly = 445, 340
gain = 2


def make_600p():
    webcam.set(3, 800)
    webcam.set(4, 600)


def square_on_image(image, lx, ly, width, height):
    cx, cy = int(width / 2), int(height / 2)
    px1, px2 = cx - int(lx / 2), cx + int(lx / 2)
    qy1, qy2 = cy - int(ly / 2), cy + int(ly / 2)

    image[qy1:qy2, px1] = 255
    image[qy1:qy2, px2] = 25
    image[qy1, px1:px2] = 255
    image[qy2, px1:px2] = 255
    return image


def cropped_image(image, lx, ly, width, height):
    cx, cy = int(width / 2), int(height / 2)
    px1, px2 = cx - int(lx / 2), cx + int(lx / 2)
    qy1, qy2 = cy - int(ly / 2), cy + int(ly / 2)

    return image[qy1:qy2, px1:px2]


if webcam.isOpened():
    print("Starting...")
    make_600p()
    width = webcam.get(cv.CAP_PROP_FRAME_WIDTH)
    height = webcam.get(cv.CAP_PROP_FRAME_HEIGHT)
    val, first_frame = webcam.read()
    first_framebw = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    num = 0

    while val:
        val, rt_frame = webcam.read()
        rt_framebw = cv.cvtColor(rt_frame, cv.COLOR_BGR2GRAY)
        rt_framebwmrk = square_on_image(rt_framebw, lx, ly, width, height)
        cv.imshow("Imaged Frame", rt_framebwmrk)
        diff = cv.absdiff(rt_framebw, first_framebw)
        diffsub = cropped_image(diff, lx, ly, width, height)
        diffsub = cv.medianBlur(diffsub, 5)
        diffsub = cv.multiply(diffsub, gain)
        diffsub_color = cv.applyColorMap(diffsub, cv.COLORMAP_JET)
        cv.imshow("Image Difference", diffsub_color)
        key = cv.waitKey(delay)
        if key == 13:
            cv.imwrite("im\\diffsub_" + str(num) + "_color.jpg", diffsub_color)
            num += 1
        if key == 27:
            break
    webcam.release()
    cv.destroyAllWindows()
