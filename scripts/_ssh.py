#!/usr/bin/env python
"""Minimal paramiko SSH helper for the GPU server.

Credentials are read from the environment so nothing secret is committed:

    AL_SSH_HOST  (default: connect.westb.seetacloud.com)
    AL_SSH_PORT  (default: 49844)
    AL_SSH_USER  (default: root)
    AL_SSH_PASS  (required)

Usage:
    python scripts/_ssh.py "nvidia-smi -L && nproc"
    python scripts/_ssh.py --put local.txt /root/remote.txt
    python scripts/_ssh.py --get /root/remote.txt local.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import paramiko


def _client() -> paramiko.SSHClient:
    host = os.environ.get("AL_SSH_HOST", "connect.westb.seetacloud.com")
    port = int(os.environ.get("AL_SSH_PORT", "49844"))
    user = os.environ.get("AL_SSH_USER", "root")
    password = os.environ.get("AL_SSH_PASS")
    if not password:
        sys.exit("AL_SSH_PASS not set in environment")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, port=port, username=user, password=password, timeout=30,
                banner_timeout=30, auth_timeout=30)
    return cli


def run(cmd: str, timeout: int = 600) -> int:
    cli = _client()
    try:
        stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=True)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        if out:
            sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        return code
    finally:
        cli.close()


def put(local: str, remote: str) -> int:
    cli = _client()
    try:
        sftp = cli.open_sftp()
        sftp.put(local, remote)
        sftp.close()
        print(f"put {local} -> {remote}")
        return 0
    finally:
        cli.close()


def get(remote: str, local: str) -> int:
    cli = _client()
    try:
        sftp = cli.open_sftp()
        sftp.get(remote, local)
        sftp.close()
        print(f"get {remote} -> {local}")
        return 0
    finally:
        cli.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", help="remote command to run")
    ap.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    if args.put:
        return put(*args.put)
    if args.get:
        return get(*args.get)
    if args.cmd:
        return run(args.cmd, timeout=args.timeout)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
