import sys

from PyQt5.QtWidgets import *

from PyQt5.QtCore import *

from PyQt5.QtGui import *

from PyQt5.uic import loadUi

import numpy as np

import cv2

import test

class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()

class MyWindow(QDialog):
    def __init__(self):
        super().__init__()
        loadUi('mainForm.ui',self)
        self.image = None
        self.hsvImage = None
        self.maskImage = None
        self.adjustImage = None
        self.resultImage = None
        self.Img = None
        self.mode = None
        self.selectedColor = None
        self.gray = None
        self.pre = False

        self.brushImage = None  # 최종 추가
        self.color = None  # 최종 추가
        self.stk = []  # 최종 추가

        self.drawing = False
        self.removeRegion = True
        self.addRegion = False
        self.current_x = 0
        self.current_y = 0
        self.brushSize = 3

        self.former_x = 0
        self.former_y = 0
        self.resultLabel.setMouseTracking(True)
        self.resultLabel.mousePressEvent = self.RemoveAdd2
        self.resultLabel.mouseMoveEvent = self.RemoveAdd
        self.resultLabel.mouseReleaseEvent = self.RemoveAdd3


        self.loadButton.clicked.connect(self.loadClicked) #불러오기 버튼
        self.saveButton.clicked.connect(self.saveClicked) #저장하기 버튼
        self.restoreButton.clicked.connect(self.restoreClicked)  # 저장하기 버튼

        self.pretreatmentButton.clicked.connect(self.pretreatmentClicked) #전처리 버튼
        self.coloringButton.clicked.connect(self.coloringClicked) #딥러닝 채색 버튼

        self.colorSelectButton.clicked.connect(self.selectColor) #수정할 컬러 선택 버튼

        self.setRange.valueChanged.connect(self.setColor2)  #수정할 컬러 범위 조정 슬라이더
        self.setBrushSize.valueChanged.connect(self.setBrush)

        self.hue.valueChanged.connect(self.resultDisplay) #컬러 바꾸기 슬라이더
        self.sat.valueChanged.connect(self.resultDisplay) #채도 바꾸기 슬라이더
        self.val.valueChanged.connect(self.resultDisplay) #밝기 바꾸기 슬라이더

    @pyqtSlot()
    def keyPressEvent(self,event):
        print(event.key())
        if self.maskImage is not None:
            if event.key() == 16777248:
                if self.mode == "selectColor":
                    self.mode = "changeColor"
                    self.resultImage = self.image.copy()
                    self.displayImage(2)
                elif self.mode == "changeColor":
                    self.mode = "selectColor"
                    self.resultImage = self.maskImage.copy()
                    self.displayImage(2)

            elif event.key() == 90: # 최종 추가
                img = self.stk.pop().copy()
                if not self.stk:
                    self.stk.append(img)
                self.image = img.copy()
                self.resultImage = self.image.copy()
                self.displayImage(2)

    @pyqtSlot()
    def setBrush(self):
        self.brushSize = self.setBrushSize.value()

    @pyqtSlot()
    def RemoveAdd(self,event):  # 최종 추가
        if self.mode == "selectColor":
            if self.maskImage is not None:
                h = self.Img.height() / self.resultLabel.height()
                w = self.Img.width() / self.resultLabel.width()
                self.former_x = int(event.pos().x() * w)
                self.former_y = int(event.pos().y() * h)

                if self.AddButton.isChecked():
                    if self.drawing == True:
                        cv2.circle(self.maskImage, (self.former_x, self.former_y), self.brushSize, (255, 255, 255), -1)
                        self.resultImage = self.maskImage.copy()
                        self.displayImage(2)

                elif self.RemoveButton.isChecked():
                    if self.drawing == True:
                        cv2.circle(self.maskImage, (self.former_x, self.former_y), self.brushSize, (0, 0, 0), -1)
                        self.resultImage = self.maskImage.copy()
                        self.displayImage(2)

        elif self.mode == "changeColor":   # 최종 추가
            if self.resultImage is not None and self.selectedColor is not None:
                h = self.Img.height() / self.resultLabel.height()
                w = self.Img.width() / self.resultLabel.width()

                self.former_x = int(event.pos().x() * w)
                self.former_y = int(event.pos().y() * h)

                if self.drawing == True:
                    self.brushImage = self.image.copy()

                    if self.image is not None: # 되돌리기
                        self.stk.append(self.image)

                    imCrop = self.brushImage[self.former_y - (self.brushSize): self.former_y + (self.brushSize), self.former_x - (self.brushSize): self.former_x + (self.brushSize)]
                    cv2.circle(imCrop, (self.brushSize-1, self.brushSize-1), self.brushSize - 2, (self.color.blue(), self.color.green(), self.color.red()), -1)
                    imBlur = cv2.GaussianBlur(imCrop,(3,3),0)

                    cropGray = cv2.cvtColor(imBlur, cv2.COLOR_BGR2GRAY)
                    mask = np.zeros(cropGray.shape, dtype=np.uint8)
                    cv2.circle(mask, (self.brushSize-1, self.brushSize-1), self.brushSize-1, 255 , -1)
                    mask_inv = cv2.bitwise_not(mask)
                    img1_fg = cv2.bitwise_and(imBlur, imBlur, mask=mask)
                    img2_bg = cv2.bitwise_and(imCrop, imCrop, mask=mask_inv)
                    dst = cv2.add(img1_fg, img2_bg)

                    self.brushImage[self.former_y - (self.brushSize): self.former_y + (self.brushSize), self.former_x - (self.brushSize): self.former_x + (self.brushSize)] = dst
                    self.image = self.brushImage.copy()
                    self.resultImage = self.image.copy()
                    self.displayImage(2)

    @pyqtSlot()
    def RemoveAdd2(self, event):
        self.drawing = True

    @pyqtSlot()
    def RemoveAdd3(self, event):
        self.drawing = False

    @pyqtSlot()
    def restoreClicked(self): # 최종 추가
        if self.image is None:
            messagebox = QMessageBox(QMessageBox.Information, "안내", "이미지 불러오기를 먼저 실행하세요.",
                                     buttons=QMessageBox.Ok | QMessageBox.Cancel, parent=self)
            messagebox.show()
        else:
            img_mark = self.image.copy()
            mark = np.zeros(self.image.shape[:2], np.uint8)
            sketch = Sketcher('img', [img_mark, mark], lambda: ((255, 255, 255), 255))
            self.res = None
            stack = []
            stack.append(self.image)
            while True:
                ch = cv2.waitKey()
                # esc 사용 and 창 꺼질시c
                if ch == 27 or cv2.getWindowProperty('img', 0) < 0:
                    if self.res is not None:
                        self.res = img_mark
                        self.image = self.res.copy()
                        self.resultImage = self.image.copy()
                        self.displayImage(2)
                        sketch.show()
                    break
                # space 바
                if ch == ord(' '):
                    if self.res is not None:
                        stack.append(self.res)

                    self.res = cv2.inpaint(img_mark, mark, 3, cv2.INPAINT_TELEA)
                    #cv2.namedWindow('inpaint', cv2.WINDOW_NORMAL)
                    #cv2.imshow('inpaint', res)
                    img_mark[:] = self.res.copy()
                    mark[:] = 0
                    sketch.show()
                # Z 키
                if ch == ord('z'):
                    img = stack.pop().copy()
                    if not stack:
                        stack.append(img)
                        self.res = None
                    img_mark[:] = img.copy()
                    mark[:] = 0
                    sketch.show()

            self.res = img_mark
            cv2.destroyAllWindows()

    @pyqtSlot()
    def pretreatmentClicked(self): # 전처리 버튼 누르면 실행될 내용
        if self.image is None:
            messagebox = QMessageBox(QMessageBox.Information, "안내", "이미지 불러오기를 먼저 실행하세요.", buttons=QMessageBox.Ok | QMessageBox.Cancel, parent=self)
            messagebox.show()
        elif self.pre == False:
            # 영역분할 필터
            #cv2.pyrMeanShiftFiltering(self.image, 1, 1, self.image)
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(self.gray, 0, 255,
                                        cv2.THRESH_BINARY_INV +
                                        cv2.THRESH_OTSU)
            thresh = abs(255 - thresh)
            kernel = np.ones((3, 3), np.uint8)

            # pre_result : 전처리 결과
            self.pre = True
            self.image = thresh#cv2.erode(thresh, kernel, iterations=1)
            self.resultImage = self.image.copy()
            self.displayImage(2)

    @pyqtSlot()
    def coloringClicked(self):  # 딥러닝 채색 버튼 누르면 실행될 내용
        if self.image is None:
            messagebox = QMessageBox(QMessageBox.Information, "안내", "이미지 불러오기를 먼저 실행하세요.",
                                     buttons=QMessageBox.Ok | QMessageBox.Cancel, parent=self)
            messagebox.show()
        else:
            write_img1 = self.image.copy()
            write_img2 = cv2.resize(write_img1, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imwrite('C:\\Users\\user\\Desktop\\final\\datasets\\colorize\\testA\\test.jpg', write_img2)
            test.starting()

            self.image = cv2.imread(
                'C:\\Users\\user\\Desktop\\final\\results\\colorize\\test_latest\\images\\test_fake_B.png')
            self.resultImage = self.image
            self.displayImage(2)

    # 결과 이미지 출력할 때 참고할 내용 : 딥러닝에 활용할 이미지는 QImage 형식으로 self.image에 저장돼있고,
    # 딥러닝 채색 결과를 self.image과 self.resultimage 에 초기화 해주고, self.displayImage(2) 함수 실행하면 됨.

    @pyqtSlot()
    def setColor2(self):
        if self.image is None:
            messagebox = QMessageBox(QMessageBox.Information, "안내", "딥러닝 채색을 먼저 실행하세요.",
                                     buttons=QMessageBox.Ok | QMessageBox.Cancel, parent=self)
            messagebox.show()

        elif self.selectedColor is None:
            messagebox = QMessageBox(QMessageBox.Information, "안내", "수정할 컬러 선택을 먼저 완료해주세요.",
                                     buttons=QMessageBox.Ok | QMessageBox.Cancel, parent=self)
            messagebox.show()

        else:
            self.mode = "selectColor"
            self.hsvImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            maxH = self.selectedColor + self.setRange.value()
            minH = self.selectedColor - self.setRange.value()

            minH2 = minH

            if minH < 0:
                minH2 = 0;

            color_lower = np.array([minH2, 0, 0], np.uint8)
            color_upper = np.array([maxH, 255, 255], np.uint8)
            color_mask = cv2.inRange(self.hsvImage, color_lower, color_upper)

            print(minH)
            print(maxH)

            if minH < 0:
                color_lower2 = np.array([180 + minH, 0, 0], np.uint8)
                color_upper2 = np.array([180, 255, 255], np.uint8)
                color_mask2 = cv2.inRange(self.hsvImage, color_lower2, color_upper2)
                color_mask = cv2.bitwise_or(color_mask, color_mask2)

            if maxH > 180:
                color_lower3 = np.array([0, 0, 0], np.uint8)
                color_upper3 = np.array([maxH - 180, 255, 255], np.uint8)
                color_mask3 = cv2.inRange(self.hsvImage, color_lower3, color_upper3)
                color_mask = cv2.bitwise_or(color_mask, color_mask3)

            self.maskImage = color_mask.copy()
            self.resultImage = color_mask.copy()
            self.displayImage(2)

    @pyqtSlot()
    def selectColor(self):
        self.color = QColorDialog.getColor()  # 최종 추가
        self.selectedColor = self.color.hsvHue() / 2
        if self.color.isValid():
            self.colorFrame.setStyleSheet("QWidget {background-color: %s}" % self.color.name())

    @pyqtSlot()
    def resultDisplay(self):
        if self.maskImage is None:
            messagebox = QMessageBox(QMessageBox.Information, "안내", "수정할 영역 선택을 먼저 진행하세요.", buttons=QMessageBox.Ok | QMessageBox.Cancel, parent=self)
            messagebox.show()

        else :
            self.mode = "changeColor"

            if self.image is not None:  # 최종 추가
                self.stk.append(self.image)

            h_ = self.hue.value()
            s_ = self.sat.value()
            v_ = self.val.value()

            if h_ > 0:
                h_ = h_ - 180

            img_hsv_replaced = self.hsvImage.copy()
            (h, s, v) = cv2.split(img_hsv_replaced)

            Hue = np.array(h)
            Hue2 = np.array(h)
            h[(Hue2 + h_ >= 0) & (Hue2 + h_ <= 180)] = Hue2[(Hue2 + h_ >= 0)] + h_
            h[(Hue + h_ < 0)] = Hue[(Hue + h_ < 0)] + (h_) + 180

            s = cv2.add(s, s_)
            v = cv2.add(v, v_)

            img_hsv_replaced = cv2.merge([h, s, v])
            img_hsv_replaced = cv2.cvtColor(img_hsv_replaced, cv2.COLOR_HSV2BGR)

            mask_inv = cv2.bitwise_not(self.maskImage)

            img1_bg = cv2.bitwise_and(self.image, self.image, mask = mask_inv)

            img2_fg = cv2.bitwise_and(img_hsv_replaced, img_hsv_replaced, mask = self.maskImage)
        
            self.resultImage = cv2.add(img1_bg, img2_fg)
            self.image = self.resultImage.copy()
            self.displayImage(2)

    """
    @pyqtSlot()
    def adjustClicked(self):
        #gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) >= 3 else self.image
        #self.resultImage = cv2.Canny(gray, 100, 200)
        self.resultImage = self.hsvImage.copy()
        self.displayImage(2)
    """

    @pyqtSlot()
    def loadClicked(self):
        fname, filter = QFileDialog.getOpenFileName(self, '이미지 불러오기', 'C:\\Users\\user\\Desktop\\test',
                                                       "JPEG Image (*.jpg);;PNG Image (*.png)")
        if fname:
            self.image = None
            self.hsvImage = None
            self.maskImage = None
            self.adjustImage = None
            self.resultImage = None
            self.Img = None
            self.mode = None
            self.selectedColor = None
            self.gray = None
            self.pre = False

            self.brushImage = None  # 최종 추가
            self.color = None  # 최종 추가
            self.stk = []  # 최종 추가

            self.drawing = False
            self.removeRegion = True
            self.addRegion = False
            self.current_x = 0
            self.current_y = 0
            self.brushSize = 3

            self.former_x = 0
            self.former_y = 0

            #self.displayImage(2)

            # self.label.setText(fname[0])
            self.loadImage(fname)
            # qp = QPixmap(fname[0])
            # self.label.setPixmap(qp)


        else:
            messagebox = QMessageBox(QMessageBox.Warning, "로드 실패", "이미지를 불러오는 데 실패하였습니다.",
                                     buttons=QMessageBox.Ok | QMessageBox.Cancel, parent=self)
            messagebox.show()

    @pyqtSlot()
    def saveClicked(self):
        # if self.label.pixmap():
        #    self.label.pixmap().save('result.jpg', 'jpg', 90)
        fname, filter = QFileDialog.getSaveFileName(self, '이미지 저장하기', 'C:\\',
                                                    "JPEG Image (*.jpg);;PNG Image (*.png)")
        if fname:
            cv2.imwrite(fname, self.resultImage)
            messagebox = QMessageBox(QMessageBox.Information, "저장 성공", "이미지가 성공적으로 저장되었습니다.",
                                     buttons=QMessageBox.Ok | QMessageBox.Cancel, parent=self)
            messagebox.show()
        else:
            messagebox = QMessageBox(QMessageBox.Warning, "저장 실패", "이미지를 저장하는 데 실패하였습니다.",
                                     buttons=QMessageBox.Ok | QMessageBox.Cancel, parent=self)
            messagebox.show()

    @pyqtSlot()
    def loadImage(self, fname):
        self.image = cv2.imread(fname, cv2.IMREAD_COLOR)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.resultImage = self.image.copy()
        self.displayImage(1)
        self.resultLabel.clear()

    @pyqtSlot()
    def displayImage(self, window):
        qformat = QImage.Format_Indexed8

        if len(self.resultImage.shape) == 3:
            if (self.resultImage.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        self.Img = QImage(self.resultImage, self.resultImage.shape[1], self.resultImage.shape[0], self.resultImage.strides[0], qformat)

        self.Img = self.Img.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(self.Img))
            self.imgLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)

        if window == 2:
            self.resultLabel.setPixmap(QPixmap.fromImage(self.Img))
            self.resultLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.resultLabel.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.setWindowTitle('불교 미술 채색 복원')
    window.setWindowIcon(QIcon('Logo.jpg'))
    window.show()
    sys.exit(app.exec_())

"""
    @pyqtSlot()
    def setColor1(self):
        self.hsvImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        color_lower = np.array([self.hMin.value(), self.sMin.value(), self.vMin.value()], np.uint8)
        color_upper = np.array([self.hMax.value(), self.sMax.value(), self.vMax.value()], np.uint8)
        color_mask = cv2.inRange(self.hsvImage, color_lower, color_upper)

        if self.hMax.value() > 180:
            color_lower = np.array([0, self.sMin.value(), self.vMin.value()], np.uint8)
            color_upper = np.array([self.hMax.value()-180, self.sMax.value(), self.vMax.value()], np.uint8)
            color_mask2 = cv2.inRange(self.hsvImage, color_lower, color_upper)
            color_mask = cv2.bitwise_or(color_mask, color_mask2)

        self.colorLabel.setText('Min : ' + str(color_lower) + ' / Max : ' + str(color_upper))

        #self.adjustImage = color_mask.copy()
        self.maskImage = color_mask.copy()
        self.resultImage = color_mask.copy()
        self.displayImage(2)
"""

"""
        def setupUI(self):
        self.setGeometry(400, 100, 800, 800)
        self.setWindowTitle("이미지 불러오기 연습")

        self.loadButton = QPushButton("File Open")
        self.loadButton.clicked.connect(self.loadClicked)

        self.saveButton = QPushButton("File Save")
        self.saveButton.clicked.connect(self.saveClicked)

        self.label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.loadButton)
        layout.addWidget(self.saveButton)
        layout.addWidget(self.label)

        self.setLayout(layout)

    def loadClicked(self):
        fname, filter = QFileDialog.getOpenFileName(self, '이미지 불러오기', 'C:\\', "JPEG Image (*.jpg);;PNG Image (*.png)")
        if fname:
            #self.label.setText(fname[0])
            self.loadImage(fname)
            #qp = QPixmap(fname[0])
            #self.label.setPixmap(qp)
        else:
            messagebox = QMessageBox(QMessageBox.Warning, "로드 실패", "이미지를 불러오는 데 실패하였습니다.",
                                     buttons = QMessageBox.Ok | QMessageBox.Cancel, parent=self)
            messagebox.show()

    def saveClicked(self):
        #if self.label.pixmap():
        #    self.label.pixmap().save('result.jpg', 'jpg', 90)
        fname, filter = QFileDialog.getSaveFileName(self, '이미지 저장하기', 'C:\\', "JPEG Image (*.jpg);;PNG Image (*.png)")

        if fname:
            cv2.imwrite(fname, self.image)
            messagebox = QMessageBox(QMessageBox.Information, "저장 성공", "이미지가 성공적으로 저장되었습니다.",
                                     buttons=QMessageBox.Ok | QMessageBox.Cancel, parent=self)
            messagebox.show()
        else:
            messagebox = QMessageBox(QMessageBox.Warning, "저장 실패", "이미지를 저장하는 데 실패하였습니다.",
                                     buttons=QMessageBox.Ok | QMessageBox.Cancel, parent=self)
            messagebox.show()

    def loadImage(self, fname):
        self.image = cv2.imread(fname, cv2.IMREAD_COLOR)
        self.displayImage()

    def displayImage(self):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)

        img = img.rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(img))
        #self.label.setAlignment(QtCore.Qt.AlignHCenter)
    

        self.pushButton3 = QPushButton("Drawing")
        self.pushButton3.clicked.connect(self.pushButtonClicked3)
    
        def mouseMoveEvent(self, event):
            txt = "Mouse 위치 ; x={0},y={1}, global={2},{3}".format(event.x(), event.y(), event.globalX(), event.globalY())
            self.statusbar.showMessage(txt)
            print(event.globalX())
        
            def pushButtonClicked3(self):
        pix = self.label.pixmap()
        mask = pix.createMaskFromColor(QColor(255, 255, 255), Qt.MaskOutColor)

        p = QPainter(pix)
        p.setPen(QColor(0, 0, 255))
        p.drawPixmap(pix.rect(), mask, mask.rect())
        p.end()
        
"""
"""
class MyMain(QMainWindow):
    def __init__(self):
        super().__init__()

        self.statusbar = self.statusBar()
        print(self.hasMouseTracking())
        self.setMouseTracking(False) # True 면, mouse button 안눌러도 , mouse move event 추적함.
        print(self.hasMouseTracking())
        self.setGeometry(300, 200, 400, 200)
        self.show()



    def mouseMoveEvent(self, event):
        txt = "Mouse 위치 ; x={0},y={1}, global={2},{3}".format(event.x(), event.y(), event.globalX(), event.globalY())
        self.statusbar.showMessage(txt)
        print(event.globalX())
"""