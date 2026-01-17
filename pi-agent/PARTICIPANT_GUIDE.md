## Participant guide: record “looking at screen” data (laptop webcam)

This takes about **8–10 minutes**.

### 1) Get the code
You can either:
- download the `pi-agent/` folder from the project owner, or
- clone the repo and use `pi-agent/`.

### 2) Install dependencies (one-time)

```bash
cd pi-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-collect.txt
```

### 3) Run the guided session

```bash
cd pi-agent
python -m studybuddy_pi collect-data --participant pXX --session s01 --placement laptop_webcam
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

