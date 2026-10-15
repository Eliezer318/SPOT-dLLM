#!/usr/bin/env python
"""Create the Hugging Face model repo for the SPOT drafters and upload them (private by default).

    huggingface-cli login                      # once, with a write token
    python hf/upload_drafters.py --repo-id <hf-user>/SPOT-dLLM

The repo is created private; make it public later from the Hugging Face settings page.
"""
import argparse, os, shutil, tempfile
from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--repo-id", required=True)
ap.add_argument("--public", action="store_true", help="create the repo public (default: private)")
args = ap.parse_args()

api = HfApi()
api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    shutil.copytree(os.path.join(ROOT, "drafters"), os.path.join(tmp, "drafters"))
    card = open(os.path.join(ROOT, "hf", "MODEL_CARD.md"), encoding="utf-8").read().replace("HF_REPO_ID", args.repo_id)
    open(os.path.join(tmp, "README.md"), "w", encoding="utf-8").write(card)
    for name in os.listdir(os.path.join(tmp, "drafters")):
        shutil.move(os.path.join(tmp, "drafters", name), os.path.join(tmp, name))
    os.rmdir(os.path.join(tmp, "drafters"))
    os.makedirs(os.path.join(tmp, "assets"))
    shutil.copy(os.path.join(ROOT, "docs", "assets", "spot_replay.gif"), os.path.join(tmp, "assets", "spot_replay.gif"))
    api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=tmp,
                      commit_message="SPOT drafter checkpoints")
print("uploaded to https://huggingface.co/" + args.repo_id, "(private)" if not args.public else "(public)")
