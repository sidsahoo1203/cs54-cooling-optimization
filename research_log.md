# CS54 Research Log — Siddharth Sahoo
**Project:** AI-Based Cooling Optimization in Data Centers
**Guide:** Dr. P. Akilandeswari
**Reg. No:** RA2512005010041

---

## 2026-09-09 (Day 1) — Environment Setup

**Phase:** Phase 0 — Setup

**Goal:** Working Linux development environment for CS54 implementation.

**Done:**
- Found broken WSL Ubuntu registration left over from an old uninstall;
  unregistered and reinstalled clean.
- WSL2 + Ubuntu (resolute) installed, user account created.
- System updated: 152 packages upgraded.
- Installed build-essential (GCC 15.2.0), git 2.53.0, curl 8.18.0, wget.
- Installed Miniconda 26.7.1 at /home/siddharth/miniconda3.
- Created conda env `cs54` with Python 3.10.21.
- Created project structure and initialized Git repository.

**Environment decisions:**
- WSL2 rather than Colab: persistent filesystem, no session timeouts,
  standard Linux toolchain for research code.
- Conda pins Python 3.10 (SustainDC requirement) independently of
  Ubuntu's system Python 3.14.
- Accepted Anaconda default-channel ToS (free for academic use).

**Problems:**
- Broken pre-existing WSL distro: /bin/sh missing, getpwuid failures.
  Resolved via `wsl --unregister Ubuntu` then fresh install.

**Next:** Clone SustainDC, install requirements, run demo training.
