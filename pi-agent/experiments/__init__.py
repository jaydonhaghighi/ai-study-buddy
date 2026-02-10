"""
Local experiment automation for pi-agent training/evaluation.

Design goals:
- Reproducible: every run writes a self-contained folder with config + metrics + logs.
- Low-dependency: orchestrates training via subprocess (TensorFlow stays optional here).
- AI-friendly: report output includes a paste-ready "AI-ready" block.
"""

