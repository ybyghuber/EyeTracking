import glob
import os
import cv2
import numpy as np
import math

DEBUG = False


# 　测地膨胀
def D_g(n, f, b, g):
    if n == 0:
        return f
    if n == 1:
        if DEBUG:
            cv2.imshow('g', g)
            cv2.imshow('img', cv2.dilate(f, b, iterations=1))
            cv2.imshow('min', np.min((cv2.dilate(f, b, iterations=1), g), axis=0))
            # cv2.imshow('c',np.min((cv2.dilate(f,b,iterations=1),g),axis=0)-cv2.dilate(f,b,iterations=1))
            cv2.waitKey()
            cv2.destroyAllWindows()
            # from IPython.core.debugger import Tracer; Tracer()()
            # print((cv2.dilate(f,b,iterations=1)<=g).all())
        return np.min((cv2.dilate(f, b, iterations=1), g), axis=0)
    return D_g(1, D_g(n - 1, f, b, g), b, g)


# 测地腐蚀
def E_g(n, f, b, g):
    if n == 0:
        return f
    if n == 1:
        return np.max((cv2.erode(f, b, iterations=1), g), axis=0)
    return E_g(1, E_g(n - 1, f, b, g), b, g)


# 膨胀重建
def R_g_D(f, b, g):
    if DEBUG:
        cv2.imshow('origin', f)
        cv2.waitKey()
        # cv2.destroyAllWindows()
    img = f
    while True:
        new = D_g(1, img, b, g)
        # cv2.destroyAllWindows()
        if (new == img).all():
            return img
        img = new


# 腐蚀重建
def R_g_E(f, b, g):
    img = f
    while True:
        new = E_g(1, img, b, g)
        if (new == img).all():
            return img
        img = new


# 重建开操作
def O_R(n, f, b, conn=np.ones((3, 3))):
    erosion = cv2.erode(f, b, iterations=n)
    return R_g_D(erosion, conn, f)


# 重建闭操作
def C_R(n, f, b, conn=np.ones((3, 3))):
    dilation = cv2.dilate(f, b, iterations=n)
    return R_g_E(dilation, conn, f)


images = glob.glob('../image/gray_eye/' + os.sep + '**.' + 'png')
for frame in images:
    img = cv2.imread(frame)
    img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_AREA)
    img = cv2.bilateralFilter(img, 10, 15, 15)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    h = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations=3)
    b = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations=3)
    result = img + h - b

    con_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    contruct = O_R(3, result, con_kernel)

    ret1, img_thres = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    ret2, result_thres = cv2.threshold(result, 80, 255, cv2.THRESH_BINARY)

    # canny = cv2.Canny(result, 50, 150)
    canny = cv2.Canny(result_thres, 30, 200)
    # circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=80, param2=10, minRadius=20, maxRadius=30)
    # 椭圆检测
    ret, thresh = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
    # print('circles=', circles)
    # if circles is not None and len(circles) > 0:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         # draw the outer circle
    #         cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #         # draw the center of the circle
    #         cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    #     print('circleNum= ', len(circles))
    for cnt in contours:
        print(len(cnt))
        if 60 < len(cnt) < 200:  # 30
            S1 = cv2.contourArea(cnt)  # 轮廓面积
            ell = cv2.fitEllipse(cnt)
            S2 = math.pi * ell[1][0] * ell[1][1]  # 计算椭圆面积
            if (S1 / S2) > 0.0001:  # 面积比例  0.003
                img_1 = cv2.ellipse(img, ell, (0, 255, 0), 1)
                print(str(S1) + "    " + str(S2) + "   " + str(ell[0][0]) + "   " + str(ell[0][1]))

    imgs = np.hstack([img, result, contruct, img_thres, result_thres])
    cv2.imshow('canny', canny)
    cv2.imshow('imgs', imgs)

    if 27 == cv2.waitKey(500):
        break
