"""Last Colab step: save a copy of the executed notebook next to the results and zip everything
(run with %run inside the notebook)."""
import json
import subprocess

from google.colab import _message

OUT = "/content/drive/MyDrive/Deep Learning Projects/BoltaNote/outputs"
nb = _message.blocking_request("get_ipynb", timeout_sec=120)["ipynb"]
json.dump(nb, open(f"{OUT}/BoltaNote_train.ipynb", "w"))
r = subprocess.run(f'cd "{OUT}/.." && rm -f boltanote_outputs.zip && zip -qr boltanote_outputs.zip outputs && ls -la', shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr)
print("DONE")
