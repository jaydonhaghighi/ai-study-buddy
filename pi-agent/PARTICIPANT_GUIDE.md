## Participant guide: record “looking at screen” data (laptop webcam)

This takes about **8–10 minutes** and works on **macOS** and **Windows**.

### 1) Get the code
You can either:
- download the `pi-agent/` folder from the project owner, or
- clone the repo and use `pi-agent/`.

### 2) Install dependencies (one-time)

#### macOS

```bash
cd pi-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-collect.txt
```

#### Windows (PowerShell)

```powershell
cd pi-agent
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-collect.txt
```

### 3) Run the guided session

#### macOS

```bash
cd pi-agent
python3 -m studybuddy_pi collect-data --participant pXX --session s01 --placement laptop_webcam
```

#### Windows (PowerShell)

```powershell
cd pi-agent
py -m studybuddy_pi collect-data --participant pXX --session s01 --placement laptop_webcam
```

If the webcam doesn't open, try a different index:

```bash
python -m studybuddy_pi collect-data --webcam 1 --participant pXX --session s01 --placement laptop_webcam
```

macOS camera permission:
- If you see “not authorized to capture video”, enable camera access for your Terminal/Python in:
  - System Settings → Privacy & Security → Camera

Windows camera permission:
- Settings → Privacy & security → Camera
  - turn on **Camera access**
  - turn on **Let desktop apps access your camera**

Linux headless note:
- If you’re on a machine without a desktop display, run with:

```bash
python -m studybuddy_pi collect-data --no-preview --participant pXX --session s01 --placement laptop_webcam
```

When the preview window opens:
- **Press Enter** to begin.
- Follow the on-screen prompts (“LOOK AT SCREEN”, “LOOK AWAY”, etc.).
- Keys:
  - `Q` quit
  - `S` skip the current condition (e.g., glasses/dim light if not applicable)

### 4) Send the output back

The program prints a folder path like:

```
data/run_1234567890/
```

Please zip that entire folder and send it back.

It contains labeled face crops in:

```
face/<participant>/<session>/<placement_condition>/{looking,not_looking}/*.jpg
```

