## Participant guide: record attention direction data (laptop webcam)

This takes about **8–10 minutes** and works on **macOS** and **Windows**.

### 1) Get the code
You can either:
- download the `pi-agent/` folder from the project owner, or
- clone the repo and use `pi-agent/`.

### 2) Run the guided session (web collector)

Data collection is done via the **web-based collector** (more reliable across OSes).

1) Run the web app and open:
- `http://localhost:5173/collect` (dev), or
- your deployed site at `/collect`

2) Fill in `participant` / `session` / `placement` and follow the guided flow.

macOS camera permission:
- If you see “not authorized to capture video”, enable camera access for your Terminal/Python in:
  - System Settings → Privacy & Security → Camera

Windows camera permission:
- Settings → Privacy & security → Camera
  - turn on **Camera access**
  - turn on **Let desktop apps access your camera**

Linux headless note:
- If you’re on a machine without a desktop display, use a machine with a browser + camera for collection.

During the flow you’ll be prompted to look in 5 directions:
- **screen**
- **away_left**
- **away_right**
- **away_up**
- **away_down**

### 4) Send the output back

The program prints a folder path like:

```
data/run_1234567890/
```

Please zip that entire folder and send it back.

It contains labeled face crops in:

```
face/<participant>/<session>/<placement_condition>/{screen,away_left,away_right,away_up,away_down}/*.jpg
```