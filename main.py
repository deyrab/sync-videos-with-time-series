import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from viewer import BehaviorVisualizer

def main():
    app = QApplication(sys.argv)

    video_file, _ = QFileDialog.getOpenFileName(
        None, "Select Video File", "", "Video Files (*.avi *.mp4 *.mov)"
    )
    if not video_file:
        QMessageBox.critical(None, "Error", "No video file selected.")
        return

    data_file, _ = QFileDialog.getOpenFileName(
        None, "Select Data File", "", "Excel Files (*.xlsx *.xls)"
    )
    if not data_file:
        QMessageBox.critical(None, "Error", "No data file selected.")
        return

    viewer = BehaviorVisualizer(video_file, data_file, frame_scale=0.5)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
