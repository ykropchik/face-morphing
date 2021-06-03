from enum import Enum

import cv2 as cv
import dlib
import numpy as np
import sys


class Animation(Enum):
    DISABLE = 1
    PROCESS_ANIMATION = 2
    TRANSFORMATION_ANIMATION = 3


ANIMATION = Animation.PROCESS_ANIMATION
DRAW_POINTS = True
DRAW_TRIANGLES = True


def isRectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def drawPoints(img, points):
    for i in range(0, len(points)):
        cv.circle(img=img, center=points[i], radius=1, color=(255, 255, 0), thickness=2)
        cv.putText(img, str(i), (points[i][0] + 4, points[i][1] + 4), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def drawTriangle(img, points):
    with open("triangles.txt") as file:
        for line in file:
            x, y, z = line.split()

            x = int(x)
            y = int(y)
            z = int(z)

            cv.line(img, points[x], points[y], (255, 255, 0), 1, cv.LINE_AA)
            cv.line(img, points[y], points[z], (255, 255, 0), 1, cv.LINE_AA)
            cv.line(img, points[z], points[x], (255, 255, 0), 1, cv.LINE_AA)


def getPoints(points1, points2, alpha):
    points = []

    for i in range(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((x, y))

    return points


def getFaceLandmarks(img, points):
    faceDetector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    grayImg = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)
    faces = faceDetector(grayImg)

    for face in faces:
        landmarks = predictor(image=grayImg, box=face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))

        points.append((0, 0))
        points.append((0, int(img.shape[0] / 2)))
        points.append((0, img.shape[0] - 1))
        points.append((img.shape[1] - 1, 0))
        points.append((img.shape[1] - 1, int(img.shape[0] / 2)))
        points.append((img.shape[1] - 1, img.shape[0] - 1))
        points.append((int(img.shape[1] / 2), 0))
        points.append((int(img.shape[1] / 2), img.shape[0] - 1))


def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv.INTER_LINEAR,
                        borderMode=cv.BORDER_REFLECT_101)

    return dst


def morph(img1, img2, imgMorph, points1, points2, points, alpha):
    with open("triangles.txt") as file:
        for line in file:
            x, y, z = line.split()

            x = int(x)
            y = int(y)
            z = int(z)

            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]

            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)


def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    # Find bounding rectangle for each triangle
    r1 = cv.boundingRect(np.float32([t1]))
    r2 = cv.boundingRect(np.float32([t2]))
    r = cv.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask
    cv.imshow("Result", np.uint8(img))

    if ANIMATION == Animation.PROCESS_ANIMATION:
        cv.waitKey(50)


def main():
    alpha = 0.5

    img1 = cv.imread(cv.samples.findFile("face1.jpg"))
    img2 = cv.imread(cv.samples.findFile("face3.png"))
    imgResult = np.zeros((max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1]), 3), dtype=img1.dtype)
    if img1 is None:
        sys.exit("Could not read image1")

    if img2 is None:
        sys.exit("Could not read image2")

    points1 = []
    getFaceLandmarks(img1, points1)

    points2 = []
    getFaceLandmarks(img2, points2)

    points = getPoints(points1, points2, alpha)

    if ANIMATION == Animation.TRANSFORMATION_ANIMATION:
        alpha = 0

        while alpha < 1:
            points = getPoints(points1, points2, alpha)
            morph(img1, img2, imgResult, points1, points2, points, alpha)
            alpha += 0.05
            cv.waitKey(10)

        while alpha > 0:
            points = getPoints(points1, points2, alpha)
            morph(img1, img2, imgResult, points1, points2, points, alpha)
            alpha -= 0.05
            cv.waitKey(1)

        points = getPoints(points1, points2, 0.5)
        morph(img1, img2, imgResult, points1, points2, points, 0.5)
    else:
        morph(img1, img2, imgResult, points1, points2, points, alpha)

    if DRAW_POINTS:
        drawPoints(img1, points1)
        drawPoints(img2, points2)

    if DRAW_TRIANGLES:
        drawTriangle(img1, points1)
        drawTriangle(img2, points2)

    cv.imshow("Source img1", img1)
    cv.imshow("Source img2", img2)

    if ANIMATION == Animation.DISABLE:
        cv.imshow("Result", np.uint8(imgResult))

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
