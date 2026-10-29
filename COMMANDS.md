# Monday's commands

Commands run as plain code (commands.py) BEFORE the neural model answers. If a message is
not a command, it goes to the model as usual.

| Say something like | What happens |
|---|---|
| What time is it? / what is the date? | Tells the time or date in the visitor's own time zone |
| Set a timer for 5 minutes / 1 hour 30 minutes / half an hour | Timer in the browser; a beep and a message when it ends |
| Remind me in 10 minutes | Same as a timer |
| Cancel all timers | Stops the running timers |
| What is 12 times 7? / (2+3)*4 / 2^10 / 20% of 50 / square root of 81 | Calculator (our own parser, no eval) |
| Flip a coin / roll a dice | Random result |
| Open YouTube / go to GitHub | Opens the site in a new tab (fixed list of safe sites) |
| Help / what commands can you do | Lists the commands |

Notes
- Timers live in the open browser tab. Closing the tab cancels them, and a tab in the
  background may fire up to a minute late (browsers slow down background tabs).
- To add a command: write a small function in commands.py, call it from handle(), and add
  test cases to test_commands.py. Run:  python test_commands.py
