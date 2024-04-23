import sys
import threading

import cv2
from cv2 import aruco

import math as m
from numpy import genfromtxt
import numpy as np
import pandas as pd
import time
import serial

import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, QTimer, Qt
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtGui
from bokeh.plotting import figure, output_file, show

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
board = aruco.CharucoBoard_create(3, 3, 1, 0.8, aruco_dict)
form_class = uic.loadUiType("gui.ui")[0]
isDragging = False
x0, y0, w, h = -1, -1, -1, -1
blue, red = (255, 0, 0), (0, 0, 255)


def mouse_event(event, x, y, flags, param):
    global isDragging, x0, y0, x1, y1, img
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            img_draw = img.copy()
            cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2)
            cv2.imshow("img", img_draw)
    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            w = x - x0
            h = y - y0
            x1 = x
            y1 = y
            if w > 0 and h > 0:
                img_draw = img.copy()
                cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2)
                cv2.imshow("img", img_draw)
                roi = img[y0 : y0 + h, x0 : x0 + w]
                cv2.imshow("cropped", roi)
                cv2.moveWindow("cropped", 0, 0)
            else:
                cv2.imshow("img", img)


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rot2Eul(R):
    assert isRotationMatrix(R)
    sy = m.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = m.atan2(R[2, 1], R[2, 2])
        y = m.atan2(-R[2, 0], sy)
        z = m.atan2(R[1, 0], R[0, 0])
    else:
        x = m.atan2(-R[1, 2], R[1, 1])
        y = m.atan2(-R[2, 0], sy)
        z = 0

    x = x / m.pi * 180
    y = y / m.pi * 180
    z = z / m.pi * 180

    return np.array([x, y, z])


class MyWindow(QMainWindow, form_class):
    def __init__(self, running):
        super().__init__()
        self.setupUi(self)
        self.scale = 1
        self.running = running
        self.upperRange = 1
        self.lowerRange = -1
        self.pw1 = pg.PlotWidget(title="변위 데이터")
        self.pw1.setYRange(self.lowerRange, self.upperRange)
        img = cv2.imread("wallpaper.png")
        img = cv2.resize(img, (960, 540))
        h, w, c = img.shape
        qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.label.setPixmap(pixmap)
        self.camid = 0

        self.hbox.addWidget(self.pw1)
        self.setWindowTitle("LAVOLUTION")

        self.dispRange0.setRange(1, 10)
        self.dispRange0.setSingleStep(1)
        self.dispRange1.setRange(1, 10)
        self.dispRange1.setSingleStep(1)

        self.dispRange0.valueChanged[int].connect(self.setYRange0)
        self.dispRange1.valueChanged[int].connect(self.setYRange1)
        self.btn_align.clicked.connect(self.align)
        self.btn_scale.clicked.connect(self.detect)
        self.btn_measure.clicked.connect(self.measure)
        self.btn_file.clicked.connect(self.file)
        self.btn_save.clicked.connect(self.save)
        self.btn_ang.clicked.connect(self.ang)
        self.zeroset.clicked.connect(self.zero)

    def setYRange0(self, value):
        self.lowerRange = -value
        self.pw1.setYRange(self.lowerRange, self.upperRange)

    def setYRange1(self, value):
        self.upperRange = value
        self.pw1.setYRange(self.lowerRange, self.upperRange)

    def alignlaser(self):
        cap = cv2.VideoCapture(self.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        camera_matrix = genfromtxt("const/mtx.csv", delimiter=",")
        dist_coeffs = genfromtxt("const/dst.csv", delimiter=",")
        squareLength, markerLength = 23, 18.5

        arucoParams = aruco.DetectorParameters_create()

        self.label.resize(960, 540)
        while self.running:
            ret, img = cap.read()
            h, w = img.shape[:2]
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                try:
                    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
                    )

                    dst = cv2.undistort(
                        img, camera_matrix, dist_coeffs, None, newcameramtx
                    )

                    corners, ids, rejectedImgPoints = aruco.detectMarkers(
                        dst, aruco_dict, parameters=arucoParams
                    )  # First, detect markers

                    diamondCorners, diamondIds = aruco.detectCharucoDiamond(
                        dst,
                        corners,
                        ids,
                        squareLength / markerLength,
                        diamondCorners=None,
                        diamondIds=None,
                        cameraMatrix=None,
                        distCoeffs=None,
                    )

                    dst = aruco.drawDetectedDiamonds(
                        dst, diamondCorners, diamondIds=diamondIds, borderColor=None
                    )

                    rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                        diamondCorners,
                        markerLength,
                        newcameramtx,
                        dist_coeffs,
                        rvecs=None,
                        tvecs=None,
                        _objPoints=None,
                    )

                    im_with_charuco_board = aruco.drawAxis(
                        dst, newcameramtx, dist_coeffs, rvecs[0], tvecs[0], 10
                    )  # axis length 100 can be changed according to your requirement
                    aruco_coord = np.reshape(diamondCorners, (4, 2))
                    im_with_charuco_board = cv2.circle(
                        im_with_charuco_board,
                        (int(aruco_coord[0, 0]), int(aruco_coord[0, 1])),
                        4,
                        (0, 255, 0),
                        -1,
                    )
                    im_with_charuco_board = cv2.circle(
                        im_with_charuco_board,
                        (int(aruco_coord[3, 0]), int(aruco_coord[3, 1])),
                        4,
                        (0, 255, 0),
                        -1,
                    )
                    rotMatrix = cv2.Rodrigues(src=rvecs[0])[0]
                    EULER_ANGLE = rot2Eul(rotMatrix)

                    if abs(EULER_ANGLE[0]) <= 180.5 and abs(EULER_ANGLE[0]) >= 179.5:
                        cpitch = (0, 255, 0)
                    else:
                        cpitch = (255, 0, 0)

                    if abs(EULER_ANGLE[1]) <= 0.5 and abs(EULER_ANGLE[0]) >= -0.5:
                        cyaw = (0, 255, 0)
                    else:
                        cyaw = (255, 0, 0)

                    if abs(EULER_ANGLE[2]) <= 0.5 and abs(EULER_ANGLE[0]) >= -0.5:
                        croll = (0, 255, 0)
                    else:
                        croll = (255, 0, 0)

                    im_with_charuco_board = cv2.putText(
                        im_with_charuco_board,
                        "pitch: {0}".format(EULER_ANGLE[0]),
                        (50, 880),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        cpitch,
                        2,
                        cv2.LINE_AA,
                    )
                    im_with_charuco_board = cv2.putText(
                        im_with_charuco_board,
                        "yaw: {0}".format(EULER_ANGLE[1]),
                        (50, 930),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        cyaw,
                        2,
                        cv2.LINE_AA,
                    )
                    im_with_charuco_board = cv2.putText(
                        im_with_charuco_board,
                        "roll: {0}".format(EULER_ANGLE[2]),
                        (50, 980),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        croll,
                        2,
                        cv2.LINE_AA,
                    )
                    pixelsize = aruco_coord[0, :] - aruco_coord[3, :]
                    pixelsize = m.sqrt(pixelsize[0] ** 2 + pixelsize[1] ** 2)
                    targetSize = 170  # m
                    pixelsize = targetSize / squareLength * pixelsize
                    points = np.array(
                        [
                            [pixelsize / 2, pixelsize / 2],
                            [-pixelsize / 2, pixelsize / 2],
                            [-pixelsize / 2, -pixelsize / 2],
                            [pixelsize / 2, -pixelsize / 2],
                        ]
                    )
                    points[:, 0] = points[:, 0] + width / 2
                    points[:, 1] = points[:, 1] + height / 2
                    point_1 = (int(points[0, 0]), int(points[0, 1]))
                    point_2 = (int(points[1, 0]), int(points[1, 1]))
                    point_3 = (int(points[2, 0]), int(points[2, 1]))
                    point_4 = (int(points[3, 0]), int(points[3, 1]))

                    roi_size = 10
                    align_threshold = 180
                    roi1 = im_with_charuco_board[
                        int(point_1[1] - roi_size) : int(point_1[1] + roi_size),
                        int(point_1[0] - roi_size) : int(point_1[0] + roi_size),
                    ]
                    roi2 = im_with_charuco_board[
                        int(point_2[1] - roi_size) : int(point_2[1] + roi_size),
                        int(point_2[0] - roi_size) : int(point_2[0] + roi_size),
                    ]
                    roi3 = im_with_charuco_board[
                        int(point_3[1] - roi_size) : int(point_3[1] + roi_size),
                        int(point_3[0] - roi_size) : int(point_3[0] + roi_size),
                    ]
                    roi4 = im_with_charuco_board[
                        int(point_4[1] - roi_size) : int(point_4[1] + roi_size),
                        int(point_4[0] - roi_size) : int(point_4[0] + roi_size),
                    ]

                    if roi1.mean() > align_threshold:
                        croi1 = (0, 255, 0)
                    else:
                        croi1 = (255, 0, 0)
                    if roi2.mean() > align_threshold:
                        croi2 = (0, 255, 0)
                    else:
                        croi2 = (255, 0, 0)
                    if roi3.mean() > align_threshold:
                        croi3 = (0, 255, 0)
                    else:
                        croi3 = (255, 0, 0)
                    if roi4.mean() > align_threshold:
                        croi4 = (0, 255, 0)
                    else:
                        croi4 = (255, 0, 0)

                    im_with_charuco_board = cv2.rectangle(
                        im_with_charuco_board,
                        (int(point_1[0] - roi_size), int(point_1[1] - roi_size)),
                        (int(point_1[0] + roi_size), int(point_1[1] + roi_size)),
                        croi1,
                        2,
                    )
                    im_with_charuco_board = cv2.rectangle(
                        im_with_charuco_board,
                        (int(point_2[0] - roi_size), int(point_2[1] - roi_size)),
                        (int(point_2[0] + roi_size), int(point_2[1] + roi_size)),
                        croi2,
                        2,
                    )
                    im_with_charuco_board = cv2.rectangle(
                        im_with_charuco_board,
                        (int(point_3[0] - roi_size), int(point_3[1] - roi_size)),
                        (int(point_3[0] + roi_size), int(point_3[1] + roi_size)),
                        croi3,
                        2,
                    )
                    im_with_charuco_board = cv2.rectangle(
                        im_with_charuco_board,
                        (int(point_4[0] - roi_size), int(point_4[1] - roi_size)),
                        (int(point_4[0] + roi_size), int(point_4[1] + roi_size)),
                        croi4,
                        2,
                    )

                    img = cv2.resize(im_with_charuco_board, (960, 540))
                    h, w, c = img.shape
                    qImg = QtGui.QImage(
                        img.data, w, h, w * c, QtGui.QImage.Format_RGB888
                    )
                    pixmap = QtGui.QPixmap.fromImage(qImg)
                    self.label.setPixmap(pixmap)
                except:
                    ret, img = cap.read()
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (960, 540))
                    h, w, c = img.shape
                    qImg = QtGui.QImage(
                        img.data, w, h, w * c, QtGui.QImage.Format_RGB888
                    )
                    pixmap = QtGui.QPixmap.fromImage(qImg)
                    self.label.setPixmap(pixmap)
        cap.release()

    def detectlaser(self):
        cap = cv2.VideoCapture(self.camid)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        while True:
            ret, frame = cap.read()
            if not (ret):
                break
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow("frame", frame)
            key = cv2.waitKey(33)

            if key == 27:
                img = frame
                break
        if cap.isOpened():
            cap.release()

        cv2.destroyAllWindows()

        device = select_device("")
        half = device.type != "cpu"
        imgsz = 640

        model = attempt_load("best.pt", map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)
        if half:
            model.half()

        names = model.module.names if hasattr(model, "module") else model.names

        if device.type != "cpu":
            model(
                torch.zeros(1, 3, imgsz, imgsz)
                .to(device)
                .type_as(next(model.parameters()))
            )

        # Load image
        img0 = img
        assert img0 is not None, "Image Not Found "

        # Padded resize
        img = letterbox(img0, imgsz, stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        # Process detections
        det = pred[0]
        arr = []
        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            i = 0
            # Write results
            for *xyxy, conf, cls in reversed(det):
                x_buffer = []
                y_buffer = []
                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())
                img0 = cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 0, 255), 3)
                img_ROI = img0[y1:y2, x1:x2]
                img_ROI = cv2.cvtColor(img_ROI, cv2.COLOR_BGR2GRAY)
                # binarize
                im_array = np.asarray(img_ROI)
                # for i in range(len(x2-x1)):
                #     for j in range(len(y2-y1)):
                #         img0[j,i]*
                kernel1d = cv2.getGaussianKernel(1, 3)
                kernel2d = np.outer(kernel1d, kernel1d.transpose())
                low_im_array = cv2.filter2D(im_array, -1, kernel2d)
                ret, dst = cv2.threshold(
                    low_im_array, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                x_ = np.mean(np.nonzero(dst)[1])
                y_ = np.mean(np.nonzero(dst)[0])

                ROI_origin_x = x1 + x_
                ROI_origin_y = y1 + y_
                arr.append([ROI_origin_x, ROI_origin_y])

                # -----put the output folder path here---####
                i += 1
        for i in range(len(arr)):
            img0 = cv2.circle(img0, (int(arr[i][0]), int(arr[i][1])), 4, (0, 0, 0), -1)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img0 = cv2.resize(img0, (960, 540))
        h, w, c = img0.shape
        qImg = QtGui.QImage(img0.data, w, h, w * c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)

        arr.sort(key=lambda x: x[0])
        self.angle = self.textEdit.toPlainText()
        pix_y = abs(arr[0][1] - arr[1][1])
        self.scale = 170 / pix_y / m.cos(float(self.angle) / 180 * m.pi)
        self.label.setPixmap(pixmap)
        cap.release()

    def measuredisp(self):
        nfeatures = 20
        frm = 0
        red = (0, 0, 225)
        intg = True

        cap = cv2.VideoCapture(self.camid)

        # Parameters for lucas kanade optical flow
        lk_params = dict(
            winSize=(50, 50),
            maxLevel=1,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Take first frame and find corners in it
        ret, old_frame = cap.read()

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", old_frame)
        cv2.setMouseCallback("img", mouse_event, old_frame)
        cv2.waitKey()
        cv2.destroyAllWindows()

        roi = old_frame[y0:y1, x0:x1]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures, edgeThreshold=0)
        kp, des = sift.detectAndCompute(roi_gray, None)
        p0 = cv2.KeyPoint_convert(kp)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(roi)

        self.start = time.time()

        while self.running:
            ret, frame = cap.read()

            if ret:
                frame = frame[y0:y1, x0:x1]
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    roi_gray, frame_gray, p0, None, **lk_params
                )

                good_new = p1
                good_old = p0

                buffer = []

                # draw the tracks`
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    frame = cv2.circle(frame, (int(a), int(b)), 5, red, -1)

                    buffer.append(b)

                img = cv2.add(frame, mask)
                if frm == 0:
                    self.disp_init = np.mean(buffer)
                self.disp.append(
                    (self.disp_init - np.mean(buffer))
                    * float(self.scale_txt.toPlainText())
                )
                self.time.append(time.time() - self.start)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (960, 540))
                h, w, c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                self.label.setPixmap(pixmap)

                frm = frm + 1
                if intg == True:
                    roi_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)

    @pyqtSlot()
    def get_data(self):
        self.pw1.clear()
        self.pl = self.pw1.plot(pen="y")
        self.range_time = 30

        if len(self.time) > 1:
            self.pl.setData(self.time, self.disp)
            self.disp_max.setText("현재변위: " + str(round(self.disp[-1], 3)) + " mm")
            self.disp_max.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            if self.time[-1] > self.range_time:
                k = self.time[-1] - self.range_time
                self.pw1.setXRange(k, k + self.range_time)
        self.show()

    def align(self):
        if self.running == False:
            self.running = True
            self.th_align = threading.Thread(target=self.alignlaser)
            self.th_align.start()
        else:
            self.running = False
            self.th_align.join()

    def detect(self):
        if self.running == False:
            self.running = True
            self.th = threading.Thread(target=self.detectlaser)
            self.th.start()
        else:
            self.running = False
            self.th.join()
            self.scale_txt.setText(str(round(self.scale, 3)))
            self.scale_txt.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)

    def file(self):
        self.path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

    def measure(self):
        if self.running == False:
            print(float(self.scale_txt.toPlainText()))
            self.pw1.clear()
            self.pw1.setXRange(0, 30)
            self.disp = []
            self.time = []
            self.running = True
            self.th_measure = threading.Thread(target=self.measuredisp)
            self.th_measure.start()
            self.mytimer = QTimer()
            self.mytimer.start(1 / 24 * 1000)  # 1초마다 차트 갱신 위함...
            self.mytimer.timeout.connect(self.get_data)
        else:
            self.running = False
            self.mytimer.stop()
            self.th_measure.join()

            if len(self.filename.toPlainText()) != 0:
                output_file(self.path + "/" + self.filename.toPlainText() + ".html")
            p = figure(
                title="교량 변위",
                x_axis_label="시간(sec)",
                y_axis_label="변위(mm)",
                width=800,
                height=300,
                sizing_mode="scale_width",
            )
            p.line(self.time, self.disp, line_width=3)
            show(p)

    def zero(self):
        self.disp_init = self.disp_init - self.disp[-1]

    def save(self):
        if len(self.disp) > 0:
            t100 = np.arange(0, int(self.time[len(self.time) - 1] * 100) + 1)
            t100 = t100 / 100
            d100 = np.interp(t100, self.time, self.disp)
            writer = pd.ExcelWriter(
                self.path + "/" + self.filename.toPlainText() + ".xlsx",
                engine="xlsxwriter",
            )
            df = pd.DataFrame({"시간": t100, "변위": d100})
            org = pd.DataFrame({"시간": self.time, "변위": self.disp})
            sf = pd.DataFrame(
                {
                    "스케일 팩터": float(self.scale_txt.toPlainText()),
                    "각도": float(self.textEdit.toPlainText()),
                },
                index=[0],
            )
            df.to_excel(writer, index=False, sheet_name="Data")
            sf.to_excel(writer, index=False, sheet_name="Scale")
            org.to_excel(writer, index=False, sheet_name="Original")
            writer.save()

    def ang(self):
        py_serial = serial.Serial(
            # Window
            port="COM3",
            # 보드 레이트 (통신 속도)
            baudrate=115200,
        )

        cmd = "a"
        py_serial.write(cmd.encode())

        time.sleep(0.1)

        if py_serial.readable():
            # 들어온 값이 있으면 값을 한 줄 읽음 (BYTE 단위로 받은 상태)
            # BYTE 단위로 받은 response 모습 : b'\xec\x97\x86\xec\x9d\x8c\r\n'
            response = py_serial.readline()
            angle = response[: len(response) - 2].decode().split(",")

            # 디코딩 후, 출력 (가장 끝의 \n을 없애주기위해 슬라이싱 사용)
            ax = float(angle[0])
            ay = float(angle[1])
            az = float(angle[2])
            pitch = m.atan(ax / m.sqrt(ay**2 + az**2))
            self.textEdit.setText(str(round(pitch / m.pi * 180, 3)))

    def closeEvent(self, QCloseEvent):
        re = QMessageBox.question(
            self, "종료 확인", "종료 하시겠습니까?", QMessageBox.Yes | QMessageBox.No
        )

        if re == QMessageBox.Yes:
            self.running = False
            self.th.join()
            self.th_align.join()
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow(False)
    myWindow.show()
    app.exec_()
