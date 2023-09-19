import cv2

num_frames = 0


def main():
    global num_frames

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Failed to initialize capture")
        exit(1)
    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()
        print(f"Reading frame {num_frames}")
        num_frames += 1

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow("frame", gray)
        if cv2.waitKey(1) == ord("q"):
            break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
