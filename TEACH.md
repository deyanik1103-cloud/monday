# Teaching Monday (the daily loop)

Monday logs every question it was unsure about. You answer them. Monday gets smarter.

## 1. See what Monday could not answer
- On your own computer: questions are saved in `unsure_log.txt` (not uploaded to GitHub).
- On Render: open your service, click Logs, and look for lines starting with `[unsure]`.
  Copy those lines into a text file, for example `render_logs.txt`.

## 2. Teach the answers
    python teach.py                       (questions from unsure_log.txt)
    python teach.py --file render_logs.txt

For each question, type a short answer (one or two sentences), then optionally other ways to ask it.
Enter skips it for now. `x` means never ask me about this again. `q` quits.
Answers are stored in `taught.json`. See them with `python teach.py --list`.

Type each answer in ONE line, in your own words. If you paste several lines, every line break counts as
pressing Enter, so the extra lines land in the next prompts, and after teach.py ends, PowerShell tries to run
them as commands. If you taught something by mistake:  python teach.py --remove "the wording"

## 3. Make it live right away
    git add .
    git commit -m "Taught Monday 5 new answers"
    git push

Monday now answers those questions from taught.json. The page marks them "taught answer".
(This layer only fills in when the model says "I am not sure", so it never overrides what the model knows.)

## 4. Teach the neural network too (when you have time, about 20 minutes)
    python teach.py --build
    python train_chat.py --retrain_tokenizer --steps 3000
    python export_weights.py
    python evaluate.py                    (check nothing got worse)
    git add .  /  git commit  /  git push

After this the model itself knows the answers, and the question is no longer "taught", just "known".

## Notes
- If you run `python make_dataset.py` by itself, taught answers are left out of data/chat.txt.
  Always use `python teach.py --build` instead (it runs your make_dataset.py and adds the taught answers).
- The page tells visitors that unsure questions are logged. Keep that line if you keep logging.
- Check answers for accuracy before you teach them. Monday repeats exactly what you write.
- Tests:  python test_teach.py
