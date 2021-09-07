import cv2
import numpy as np
import argparse

def shape_detector(args: argparse.Namespace):
    img = cv2.imread(args.image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    white = np.ones((img.shape[0], img.shape[1], 3))

    for c in contours:
        approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        elif len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / float(h)
            print(aspect_ratio)
            if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                cv2.putText(img, "Square", (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            else:
                cv2.putText(img, "Rectangle", (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        elif len(approx) == 5:
            cv2.putText(img, "Pentagon", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        elif len(approx) == 10:
            cv2.putText(img, "Star", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        else:
            cv2.putText(img, "Circle", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    cv2.imshow("Shapes", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="", type=str, help="Image file")
    args = parser.parse_args()
    shape_detector(args)
