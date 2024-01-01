import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QLineEdit,QProgressBar
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, QTimer, Qt
from ultralytics import YOLO

SKELETON_EDGES = [
    # Aggiungi qui la tua lista di connessioni tra keypoints
    (0, 16), (1,16), (0, 2), (1, 3), (4, 5), (4,6),(6, 8), (5, 7), (7,9), (4,10),(10,12), (5,11), (11,10), (11,13), (13,15), (12,14)
]

SCREEN_SIZE = [1024, 769]

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 e YOLO Pose Detection'
        self.model = YOLO('yolov8n-pose.pt')  # Carica il modello
        self.cap = None  # Variabile per la cattura video
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setMinimumSize(SCREEN_SIZE[0],SCREEN_SIZE[1])  # Imposta la dimensione minima della finestra


        layout = QVBoxLayout()

        # Aggiungi una barra di progresso
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)  # Imposta il range per un effetto indeterminato
        layout.addWidget(self.progress_bar)
        self.progress_bar.hide()  # Nascondi inizialmente la barra di progresso

        self.video_label = QLabel(self)
        layout.addWidget(self.video_label)

        self.btn_webcam = QPushButton('Usa Webcam', self)
        self.btn_webcam.clicked.connect(lambda: self.start_video(0))
        layout.addWidget(self.btn_webcam)

        self.btn_video = QPushButton('Usa Video', self)
        self.btn_video.clicked.connect(lambda: self.start_video(1))
        layout.addWidget(self.btn_video)

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)


    @pyqtSlot(int)
    def start_video(self, source):
        # Mostra la barra di progresso e avvia un timer
        self.progress_bar.show()
        QTimer.singleShot(500, lambda: self.load_video(source))  # Parte dopo un ritardo


    def load_video(self, source):
            if self.cap:
                self.cap.release()

            if source == 0:
                self.cap = cv2.VideoCapture(0)
            else:
                self.cap = cv2.VideoCapture('./test_video.mp4')
            
            self.timer.start(30)
        
        
    def update_frame(self):
        # Capture frame-by-frame
        if self.progress_bar.isVisible():
            self.progress_bar.hide()  # Nascondi la barra di progresso

        ret, frame = self.cap.read()
        # Predict with the model
        #model.predict(source=source, show=True, conf=0.3, save=True)
        #results = model.predict(frame,show=False, conf=0.3, save=False)  # predict on a frame
        results = self.model.track(frame, verbose=False,  show=False, persist=True, conf=0)
        
        # For each person detected in the frame
        for result in results:
            kpts = result.keypoints
            boxes = result.boxes
            masks = result.masks
            nk = kpts.shape[1]

            if(boxes.id is not None):
                kpt_insts = kpts.data
                box_insts = boxes.xyxy[:, :4]
                confs_insts = boxes.data[:, 4]
                box_ids = boxes.id.tolist()  # Estrai gli ID delle bounding boxes

                #print(box_insts.tolist())
                #print(boxes.id.tolist())
                for box, box_id in zip(box_insts.tolist(), box_ids):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                     # Calcola la larghezza e l'altezza del testo
                    text = str(int(box_id))
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

                    # Disegna un rettangolo bianco dietro il testo
                    cv2.rectangle(frame, (int(x1)-20, int(y2) - text_height), (int(x1)-20 + text_width, int(y2)), (255, 255, 255), -1)

                    # Aggiungi il testo sopra il rettangolo bianco
                    cv2.putText(frame, text, (int(x1)-20, int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                #for box in box_insts.tolist():
                #    x1, y1, x2, y2 = box
                #    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                x = [0] * 17
                y = [0] * 17
                for kpt_inst in kpt_insts.tolist():
                    #calola l'indice del kpt_inst nel for
                    for indice in range(17):
                        x[indice], y[indice] = (int(kpt_inst[-16+indice][0]), int(kpt_inst[-16 + indice][1]))   
                        cv2.circle(frame, (x[indice], y[indice]), 5, (0, 255, 0), -1)
                        cv2.putText(frame, str(indice), (x[indice], y[indice]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    # Draw the skeleton
                    for edge in SKELETON_EDGES:
                        # Skip edges where either of the keypoints is not detected 
                        if edge[0] >= nk or edge[1] >= nk:
                            continue
                        if (x[edge[0]], y[edge[0]]) != (0, 0) and (x[edge[1]], y[edge[1]]) != (0, 0):
                            cv2.line(frame, (x[edge[0]], y[edge[0]]),(x[edge[1]],y[edge[1]]),(255, 0, 0), 2)      
        

            # Converti e mostra l'immagine
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(SCREEN_SIZE[0],SCREEN_SIZE[1], Qt.KeepAspectRatio)
            self.video_label.setPixmap(QPixmap.fromImage(p))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
