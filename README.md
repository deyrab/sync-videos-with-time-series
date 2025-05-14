A Python GUI app to synchronize animal behaviour videos with data plots. Designed for researchers studying animal behaviour, this tool allows overlaying behavioural metrics and exporting annotated visualizations.

## 🎯 Who is this for?
Researchers in:
- Ethology
- Neuroscience

## 📂 Input Requirements
You need:
1. A **video file** ('.avi', '.mp4', etc.)
2. An **Excel file** ('.xlsx') containing behavioral metrics:
    - Must include a 'time_seconds' column
    - Other columns can be numeric (for plotting) or categorical (for overlays)

## ▶️ How to Run

1. Clone or download this repo.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
python main.py
```

4. Select your video and data file when prompted. You can try the program using the video and xlsx file avalaible here: https://drive.google.com/drive/folders/1lFz60GDhI6C7_ixNKZkVHuPQf9KcqlHl?usp=sharing

## 💡 Features

- ⏯️ Video playback controls
- 📊 Plot 1 or more metrics over time
- 🎨 Color selection per variable
- 🧾 Overlay text from selected columns on video
- 💾 Export:
  - Current plot as PNG, PDF, or SVG
  - Combined video + plot as MP4
  - Time-range-limited exports

## 📸 Sample Output

- Annotated MP4 showing animal movement and plots
- Clean SVG or PNG plots ready for figures

Created by: Bastien S. Lemaire
License: MIT
