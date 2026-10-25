"""Downloads a small practice text file (Tiny Shakespeare) into data.txt.
Later we will replace this with our own Monday dataset."""
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

print("Downloading practice dataset...")
urllib.request.urlretrieve(URL, "data.txt")
text = open("data.txt", encoding="utf-8").read()
print(f"Done. data.txt has {len(text):,} characters.")
