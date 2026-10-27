# Monday (built from scratch)

## One-time setup
1. Install Python 3.10 or newer.
2. In a terminal inside this folder:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. `pip install -r requirements.txt`

## Milestone 1: tiny GPT on Shakespeare (letters)
    python download_data.py
    python train.py --steps 1000
    python generate.py --prompt "ROMEO:"

## Milestone 2: Monday chat (our own tokenizer + Q&A data)
    python make_dataset.py        # builds data/chat.txt
    python train_chat.py          # trains tokenizer + model (saves tokenizer.json, ckpt_chat.pt)
    python chat.py                # talk to Monday

To make Monday smarter, add entries to ENTRIES in make_dataset.py, then run
make_dataset.py again and `python train_chat.py --retrain_tokenizer`.

## Hosting
See DEPLOY.md (export_weights.py makes the small NumPy brain used online).

## Web app (Monday in your browser)
    python -m pip install -r requirements.txt     # adds Flask
    python app.py
Then open http://127.0.0.1:5000 in your browser. (Train the model first: python train_chat.py)

## "I am not sure" check
Monday compares your question with what it was trained on (familiarity.py). If it contains
topics or words it has never seen, Monday says it is not sure instead of guessing.
See it working:  `python chat.py --debug`

## Files
- model.py: the transformer (attention, blocks, GPT)
- tokenizer.py: our own BPE tokenizer
- make_dataset.py: the Monday question-and-answer data
- train.py / generate.py: milestone 1
- numpy_gpt.py, brain_np.py, export_weights.py: the small NumPy brain used for hosting
- app.py + static/index.html: the web app (Flask backend + chat page)
- familiarity.py: the "do I know this?" check
- brain.py: Monday's brain, one function: reply(question) (the web app will use it)
- train_chat.py / chat.py: milestone 2

## Evaluation
    python evaluate.py            # full test: new wordings, new topics, out-of-scope questions
    python evaluate.py --gate     # only the "do I know this?" check (fast, no model run)
Run it before and after every retrain, and write down the score.
