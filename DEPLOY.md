# Putting Monday online (free, on Render)

Monday's hosted version runs on NumPy only (numpy_gpt.py), so it is small and fast to install.
Training stays on your laptop. Hosting only needs the finished weights.

## Part 1: Make the hosted brain (on your laptop, once, and again after every retrain)

With (venv) showing:

    python -m pip install -r requirements.txt
    python train_chat.py --retrain_tokenizer --steps 3000     (skip if you already trained)
    python export_weights.py

export_weights.py must print:  OK: the NumPy brain matches the PyTorch brain.
It creates monday_weights.npz. If it prints MISMATCH, stop and send me the message.

## Part 2: Test the hosted brain locally

    $env:MONDAY_BACKEND = "numpy"
    python app.py

Open http://127.0.0.1:5000 and chat. This is exactly what the server will run.
(To go back to the PyTorch brain later: close the terminal and open a new one.)

## Part 3: Put the project on GitHub

1. Install Git from git-scm.com if `git --version` does not work.
2. On github.com click New repository, name it `monday`, keep it empty (no README), create it.
3. In the terminal, inside the project folder:

    git init
    git add .
    git commit -m "Monday v0.1"
    git branch -M main
    git remote add origin https://github.com/YOUR-USERNAME/monday.git
    git push -u origin main

4. Check that these three files were uploaded (they are required):
   monday_weights.npz, tokenizer.json, data/chat.txt
   Quick check:  git ls-files
   (ckpt_chat.pt and venv are skipped on purpose, see .gitignore.)

## Part 4: Deploy on Render

1. Go to render.com and sign in with your GitHub account.
2. New + > Web Service > pick the `monday` repository.
3. Settings:
   - Language / Runtime: Python 3
   - Build Command:  pip install -r requirements-deploy.txt
   - Start Command:  gunicorn app:app --workers 1 --threads 2 --timeout 120
   - Instance Type:  Free
4. Click Deploy. When the log says the service is live, open the URL Render shows
   (something like https://monday-xxxx.onrender.com).

The Free plan sleeps after 15 minutes without visitors. The first visit after that takes
about a minute to wake up. That is normal.

## Updating Monday later (the daily upgrade loop)

1. Change data or code on your laptop (for example make_dataset.py).
2. python make_dataset.py, python train_chat.py --retrain_tokenizer, python export_weights.py
3. git add .   git commit -m "what you changed"   git push
Render sees the push and redeploys by itself.
