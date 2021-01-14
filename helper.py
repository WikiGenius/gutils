import matplotlib.pyplot as plt
import imutils
import cv2


def plot_images(images, names, figsize=[15, 10], rows=1, cols=None):
    plt.figure(figsize=figsize)
    for i, (name, img) in enumerate(zip(names, images)):
        if len(img.shape) == 3:
            img = imutils.opencv2matplotlib(img)
        if cols is None:
            cols = len(images)//rows
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(name)
    plt.show()


def get_center_contour(cnt):
    """
    calculate center of the contour c
    input:
    @param c: contour array
    @return: center of the contour
    """

    moments = cv2.moments(cnt)
    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])
    center = (cx, cy)
    return center


def draw_defects_contour(cnt, canvas):
    """
    draw the (start, end, further-point) for each defect in the contour
    @param cnt: given contour
    @param canvas: given canvas image to draw

    @return: the sum of further distance defects for the given contour
    """
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        return 0
    sum_further_distance_defects = 0
    for j in range(defects.shape[0]):
        s, e, f, d = defects[j, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        sum_further_distance_defects += d
        cv2.line(canvas, start, end, (0, 255, 0), 3)
        cv2.circle(canvas, far, 3, (0, 0, 255), -1)
    return sum_further_distance_defects
