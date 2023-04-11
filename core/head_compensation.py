import cv2
import face_recognition
import numpy as np
import math


def isolate_eye(frame, landmarks):
    region = np.array(landmarks)
    region = region.astype(np.int32)

    height, width = frame.shape[:2]
    black_frame = np.zeros((height, width), np.uint8)
    mask = np.full((height, width), 255, np.uint8)
    region[1][1] -= 5
    region[2][1] -= 5
    region[4][1] += 5
    region[5][1] += 5
    cv2.fillPoly(mask, [region], (0, 0, 0))
    eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

    min_x = np.min(region[:, 0])
    max_x = np.max(region[:, 0])
    min_y = np.min(region[:, 1])
    max_y = np.max(region[:, 1])

    frame = eye[min_y:max_y, min_x:max_x]

    height, width = frame.shape[:2]
    center = (width / 2 - 0.5, height / 2 - 0.5)
    return center


def eye_center(frame):
    right_eye_point = None
    left_eye_point = None
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) != 0:
        max_area = 0
        max_index = 0
        for i in range(len(face_locations)):
            top, right, bottom, left = face_locations[i]
            if math.fabs((top - bottom) * (right - left)) > max_area:
                max_area = math.fabs((top - bottom) * (right - left))
                max_index = i
        top, right, bottom, left = face_locations[max_index]
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5
        face_landmarks_list = face_recognition.face_landmarks(frame)
        for face_landmarks in face_landmarks_list:
            right_eye_point = face_landmarks['right_eye']
            left_eye_point = face_landmarks['left_eye']

    right_center = isolate_eye(frame, right_eye_point)
    left_center = isolate_eye(frame, left_eye_point)
    right_inner = right_eye_point[3][0] - right_eye_point[0][0]
    return right_center, left_center, right_inner


def eye_center_length(frame):
    length = abs(eye_center(frame)[0][0] - eye_center(frame)[1][0])
    return length


def return_s(frame1, frame2):
    if eye_center_length(frame1) != 0 and eye_center_length(frame2) != 0 and eye_center(frame1)[2] != 0 \
            and eye_center(frame2)[2] != 0:
        s = (eye_center_length(frame1) / eye_center_length(frame2))*(eye_center(frame2)[2] / eye_center(frame1)[2])
    else:
        s = 1
    return s

