
import os
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QListWidget, QListWidgetItem,
    QGroupBox, QSizePolicy, QSplitter, QFrame, QSpinBox, QFileDialog,
    QColorDialog, QProgressDialog, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class BehaviorVisualizer(QMainWindow):
    def __init__(self, video_file, data_file, frame_scale=0.5):
        super().__init__()
        self.setWindowTitle("Cuttlefish Behavior Visualizer")
        self.resize(1200, 800)

        self.frame_scale = frame_scale
        self.video_file = video_file
        self.enhanced_csv_path = data_file

        self.dlc_df = pd.read_excel(self.enhanced_csv_path)
        self.video_frames = []
        self.current_frame = 0

        self.plot_options = [col for col in self.dlc_df.columns if self.dlc_df[col].dtype in [np.float64, np.int64] and col != 'time_seconds']
        self.selected_plot_columns = []
        self.plot_colors = {}

        self.overlay_columns = [col for col in self.dlc_df.columns if col != 'time_seconds']
        self.selected_overlay_columns = []

        self.load_video_frames()
        self.init_ui()

    def load_video_frames(self):
        cap = cv2.VideoCapture(self.video_file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=self.frame_scale, fy=self.frame_scale)
            self.video_frames.append(frame)
        cap.release()
        print(f"Loaded {len(self.video_frames)} frames from video.")

    def init_ui(self):
        central_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(central_splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        content_splitter = QSplitter(Qt.Vertical)
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        content_splitter.addWidget(self.video_label)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_splitter.addWidget(self.canvas)
        left_layout.addWidget(content_splitter)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        overlay_group = QGroupBox("Overlay on Video")
        overlay_layout = QVBoxLayout()
        self.overlay_list = QListWidget()
        self.overlay_list.setSelectionMode(QListWidget.MultiSelection)
        for col in self.overlay_columns:
            self.overlay_list.addItem(QListWidgetItem(col))
        self.overlay_list.itemSelectionChanged.connect(self.update_selected_overlays)
        overlay_layout.addWidget(self.overlay_list)
        overlay_group.setLayout(overlay_layout)
        right_layout.addWidget(overlay_group)

        plot_group = QGroupBox("Plot Variables")
        plot_layout = QVBoxLayout()
        self.plot_list = QListWidget()
        self.plot_list.setSelectionMode(QListWidget.MultiSelection)
        for col in self.plot_options:
            self.plot_list.addItem(QListWidgetItem(col))
        self.plot_list.itemSelectionChanged.connect(self.update_selected_plots)
        plot_layout.addWidget(self.plot_list)

        color_button = QPushButton("Choose Plot Colors")
        color_button.clicked.connect(self.choose_colors)
        plot_layout.addWidget(color_button)

        plot_group.setLayout(plot_layout)
        right_layout.addWidget(plot_group)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.video_frames) - 1)
        self.slider.setValue(self.current_frame)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.slider_changed)
        right_layout.addWidget(self.slider)

        button_layout = QHBoxLayout()
        for label, func in [("Play", self.play_video), ("Pause", self.pause_video),
                            ("Forward", self.forward_video), ("Backward", self.backward_video)]:
            btn = QPushButton(label)
            btn.clicked.connect(func)
            button_layout.addWidget(btn)
        right_layout.addLayout(button_layout)

        scale_layout = QHBoxLayout()
        scale_label = QLabel("Video Size (%):")
        self.scale_spinbox = QSpinBox()
        self.scale_spinbox.setRange(10, 200)
        self.scale_spinbox.setValue(int(self.frame_scale * 100))
        self.scale_spinbox.setSuffix("%")
        self.scale_spinbox.valueChanged.connect(self.change_video_scale)
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_spinbox)
        right_layout.addLayout(scale_layout)

        save_btn = QPushButton("Save Plot")
        save_btn.clicked.connect(self.save_plot)
        right_layout.addWidget(save_btn)

        save_svg_btn = QPushButton("Export SVG")
        save_svg_btn.clicked.connect(self.save_plot_svg)
        right_layout.addWidget(save_svg_btn)

        # Time Range
        time_range_layout = QHBoxLayout()
        self.start_time_input = QLineEdit()
        self.end_time_input = QLineEdit()
        self.start_time_input.setPlaceholderText("Start Time (s)")
        self.end_time_input.setPlaceholderText("End Time (s)")
        time_range_layout.addWidget(QLabel("Start:"))
        time_range_layout.addWidget(self.start_time_input)
        time_range_layout.addWidget(QLabel("End:"))
        time_range_layout.addWidget(self.end_time_input)
        right_layout.addLayout(time_range_layout)

        export_video_btn = QPushButton("Export Video with Plot")
        export_video_btn.clicked.connect(self.export_video_with_plot)
        right_layout.addWidget(export_video_btn)

        central_splitter.addWidget(left_panel)
        central_splitter.addWidget(right_panel)
        central_splitter.setSizes([900, 300])

        self.timer = self.startTimer(30)
        self.playing = False

        self.update_frame(self.current_frame)

    def change_video_scale(self, value):
        self.frame_scale = value / 100.0
        self.video_frames = []
        self.load_video_frames()
        self.update_frame(self.current_frame)

    def update_selected_overlays(self):
        self.selected_overlay_columns = [item.text() for item in self.overlay_list.selectedItems()]
        self.update_frame(self.current_frame)

    def update_selected_plots(self):
        self.selected_plot_columns = [item.text() for item in self.plot_list.selectedItems()]
        self.update_frame(self.current_frame)

    def choose_colors(self):
        for col in self.selected_plot_columns:
            color = QColorDialog.getColor(title=f"Choose color for {col}")
            if color.isValid():
                self.plot_colors[col] = color.name()
        self.update_frame(self.current_frame)

    def save_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "plot.png", "PNG Files (*.png);;PDF Files (*.pdf)")
        if file_path:
            self.fig.savefig(file_path, format=os.path.splitext(file_path)[1][1:] or 'png')

    def save_plot_svg(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Plot as SVG", "plot.svg", "SVG Files (*.svg)")
        if file_path:
            self.fig.savefig(file_path, format='svg')

    def play_video(self):
        self.playing = True

    def pause_video(self):
        self.playing = False

    def forward_video(self):
        self.current_frame = min(self.current_frame + 1, len(self.video_frames) - 1)
        self.update_frame(self.current_frame)

    def backward_video(self):
        self.current_frame = max(self.current_frame - 1, 0)
        self.update_frame(self.current_frame)

    def slider_changed(self, value):
        self.current_frame = value
        self.update_frame(self.current_frame)

    def update_frame(self, frame_number):
        frame = self.video_frames[frame_number].copy()
        row = self.dlc_df.iloc[frame_number]
        time = row['time_seconds']

        text_labels = [f'Time: {time:.2f}s']
        for col in self.selected_overlay_columns:
            val = row.get(col, 'N/A')
            if isinstance(val, float):
                text_labels.append(f'{col}: {val:.2f}')
            else:
                text_labels.append(f'{col}: {val}')

        for i, text in enumerate(text_labels):
            cv2.putText(frame, text, (10, 30 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(image))

        self.update_plot(time)
        self.slider.setValue(frame_number)

    def update_plot(self, time):
        self.ax.clear()
        for col in self.selected_plot_columns:
            if col in self.dlc_df.columns:
                color = self.plot_colors.get(col, None)
                self.ax.plot(self.dlc_df['time_seconds'], self.dlc_df[col], label=col, color=color)
        if self.selected_plot_columns:
            self.ax.axvline(x=time, color='red', linestyle='--', label='Current Time')
            self.ax.set_xlim(0, self.dlc_df['time_seconds'].max())
            self.ax.set_xlabel('Time (s)')
            self.ax.set_title(" / ".join(self.selected_plot_columns))
            self.ax.legend()
        self.canvas.draw()

    def export_video_with_plot(self):
        output_path, _ = QFileDialog.getSaveFileName(self, "Export Video with Plot", "output.mp4", "MP4 files (*.mp4)")
        if not output_path:
            return

        try:
            start_time = float(self.start_time_input.text()) if self.start_time_input.text() else 0
            end_time = float(self.end_time_input.text()) if self.end_time_input.text() else self.dlc_df['time_seconds'].max()
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Start and End Time must be numbers.")
            return

        if start_time >= end_time:
            QMessageBox.critical(self, "Invalid Range", "Start Time must be less than End Time.")
            return

        indices_to_export = self.dlc_df[
            (self.dlc_df['time_seconds'] >= start_time) &
            (self.dlc_df['time_seconds'] <= end_time)
        ].index.tolist()

        if not indices_to_export:
            QMessageBox.critical(self, "No Data", "No frames found in the specified time range.")
            return

        height, width, _ = self.video_frames[0].shape
        dpi = 100
        fig_width = width / dpi
        fig_height = 2.5
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height + int(fig_height * dpi)))

        progress = QProgressDialog("Exporting video...", None, 0, len(indices_to_export), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        for i, idx in enumerate(indices_to_export):
            progress.setValue(i)
            if progress.wasCanceled():
                break

            frame = self.video_frames[idx].copy()
            time = self.dlc_df.iloc[idx]['time_seconds']

            text_labels = [f'Time: {time:.2f}s']
            for col in self.selected_overlay_columns:
                val = self.dlc_df.iloc[idx].get(col, 'N/A')
                if isinstance(val, float):
                    text_labels.append(f'{col}: {val:.2f}')
                else:
                    text_labels.append(f'{col}: {val}')

            for j, text in enumerate(text_labels):
                cv2.putText(frame, text, (10, 30 + j * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            subset_df = self.dlc_df.iloc[indices_to_export]
            for col in self.selected_plot_columns:
                if col in self.dlc_df.columns:
                    color = self.plot_colors.get(col, None)
                    ax.plot(subset_df['time_seconds'], subset_df[col], label=col, color=color)

            if self.selected_plot_columns:
                ax.axvline(x=time, color='red', linestyle='--', label='Current Time')
                ax.set_xlim(start_time, end_time)
                ax.set_xlabel('Time (s)')
                ax.set_title(" / ".join(self.selected_plot_columns))
                ax.legend()
            fig.tight_layout()
            fig.canvas.draw()

            plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            combined = np.vstack((frame, plot_img))
            combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            out.write(combined)

        progress.setValue(len(indices_to_export))
        out.release()
        print(f"Exported video saved to: {output_path}")

    def timerEvent(self, event):
        if self.playing:
            self.current_frame = (self.current_frame + 1) % len(self.video_frames)
            self.update_frame(self.current_frame)
