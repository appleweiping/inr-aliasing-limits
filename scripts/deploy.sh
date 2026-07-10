#!/usr/bin/env bash
# Sync the code tree to the GPU server over SFTP (SFTP lands in /root, shell moves it into
# /root/autodl-tmp). Data files fetched on the server (data/*.npz) are preserved -- only
# code is shipped. Requires AL_SSH_PASS in the environment.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-C:/Python314/python.exe}"
WINPWD="$(pwd -W 2>/dev/null || pwd)"
tar czf ./_upload.tgz src experiments data/fetch_real.py tests pyproject.toml README.md \
  $( [ -d paper ] && echo paper )
"$PY" scripts/_ssh.py --put "$WINPWD/_upload.tgz" inr.tgz >/dev/null
"$PY" scripts/_ssh.py "mkdir -p /root/autodl-tmp/inr-aliasing-limits && mv -f /root/inr.tgz /root/autodl-tmp/inr.tgz && cd /root/autodl-tmp/inr-aliasing-limits && tar xzf /root/autodl-tmp/inr.tgz && echo 'synced to server'"
rm -f ./_upload.tgz
