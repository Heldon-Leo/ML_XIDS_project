# Deploy the ML-XIDS dashboard on Streamlit Community Cloud

What runs in the cloud: `dashboard.py`, reading a saved snapshot of your own live-monitor run (`demo_data/live/`).
What stays on your laptop: training, the 4,500-tree model, `simulate_traffic.py`, `live_monitor.py`. Streamlit Cloud
cannot run those next to the dashboard, and the model/data files are too large for GitHub.

## A. One-time preparation (on your laptop, in the project folder, venv active)
1. Copy these files into the project root (overwrite `dashboard.py`, `requirements.txt`, `LIVE_MONITORING.md`):
   `dashboard.py`, `make_demo_snapshot.py`, `requirements.txt`, `requirements-full.txt`, `.gitignore`,
   `.streamlit/config.toml`, `DEPLOY_STREAMLIT_CLOUD.md`.
2. Produce a real run (the three commands you already use) and let it process a few hundred flows, then stop it:
       python simulate_traffic.py --reset --batch 20 --interval 2
       python live_monitor.py --from-start
3. Save the snapshot:
       python make_demo_snapshot.py
   Check that `demo_data/live/alerts.jsonl` and `demo_data/live/live_stats.json` exist.
4. Test the cloud mode locally: temporarily rename `outputs/live` to `outputs/live_x`, run
   `streamlit run dashboard.py`, confirm the blue "Cloud demo" banner and the alert table appear, then rename it back.

## B. Push to GitHub (account: Heldon-Leo)
       git init
       git add .
       git status          # must NOT list data/, models/*.pkl, outputs/, venv/ or capture/
       git commit -m "ML-XIDS: dashboard + cloud demo snapshot"
       git branch -M main
       git remote add origin https://github.com/Heldon-Leo/ML-XIDS.git
       git push -u origin main
   Create the empty repository `ML-XIDS` on github.com first (private is fine; Streamlit Cloud can read it after you
   authorise GitHub).
   The repo needs: `dashboard.py`, `config.py`, `requirements.txt`, `.streamlit/config.toml`, `demo_data/live/*`.
   (`src/`, `schemas/` and the other scripts are fine to include; the dashboard does not use them.)

## C. Deploy
1. Open https://share.streamlit.io and sign in with GitHub (allow access to the repository).
2. Create app -> "Deploy a public app from GitHub" (or "Yup, I have an app").
3. Repository `Heldon-Leo/ML-XIDS`, branch `main`, **Main file path: `dashboard.py`** (not app.py).
4. Open "Advanced settings": choose Python 3.11 or 3.12.
5. Click Deploy. The first build takes 1-3 minutes. The link looks like `https://<name>.streamlit.app` and
   works on any device. (Optional) Settings -> General lets you choose the sub-domain.

## D. Protect it (the dashboard shows IP addresses and alert details)
- Easiest: app Settings -> Sharing -> make it private and add the e-mail addresses of your guide / reviewers.
- Or a password: app Settings -> Secrets, paste
      DASH_PASSWORD = "choose-a-password"
  The dashboard then asks for it before showing anything.

## E. Updating
Run `python make_demo_snapshot.py` after a new run, then `git add demo_data` / `git commit` / `git push`.
Streamlit Cloud redeploys by itself. Free apps go to sleep after a period without visitors; the first visit
afterwards shows a "wake up" button.

## Troubleshooting
- "No data yet": `demo_data/live/live_stats.json` is missing from the repo (check .gitignore, run the snapshot).
- ModuleNotFoundError: add the package to `requirements.txt` (the cloud file is deliberately small).
- Push rejected for file size: a file over 100 MB; do not commit `models/*.pkl` or `data/`.
