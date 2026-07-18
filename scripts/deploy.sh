#!/usr/bin/env bash
# Sync the code tree to the GPU server over SFTP (SFTP lands in /root, shell moves it into
# /root/autodl-tmp). Data files fetched on the server (data/*.npz) are preserved -- only
# code is shipped. Requires AL_SSH_PASS in the environment.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-C:/Python314/python.exe}"
WINPWD="$(pwd -W 2>/dev/null || pwd)"
# ship the current commit + dirty flag so server-produced JSONs carry honest provenance.
# Compute the dirty flag BEFORE creating GIT_COMMIT: the `> GIT_COMMIT` redirect creates the
# (gitignored) file first, and an un-ignored stamp would otherwise register as a dirty change.
SRV_SHA="$(git rev-parse HEAD 2>/dev/null || true)"
SRV_STATE="clean"; [ -n "$(git status --porcelain 2>/dev/null)" ] && SRV_STATE="dirty"
printf '%s\n%s\n' "$SRV_SHA" "$SRV_STATE" > GIT_COMMIT
tar czf ./_upload.tgz src experiments data/fetch_real.py tests pyproject.toml README.md \
  requirements.lock GIT_COMMIT $( [ -d paper ] && echo paper )
rm -f GIT_COMMIT
"$PY" scripts/_ssh.py --put "$WINPWD/_upload.tgz" inr.tgz >/dev/null
"$PY" scripts/_ssh.py "mkdir -p /root/autodl-tmp/inr-aliasing-limits && mv -f /root/inr.tgz /root/autodl-tmp/inr.tgz && cd /root/autodl-tmp/inr-aliasing-limits && tar xzf /root/autodl-tmp/inr.tgz && echo 'synced to server'"
rm -f ./_upload.tgz
