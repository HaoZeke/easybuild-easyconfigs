#!/usr/bin/env python3
"""Capture bounded ELF metadata after a successful SeisSol test link."""
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time

LIMIT = 65536


def inspect_command(argv):
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
    selector = selectors.DefaultSelector()
    os.set_blocking(process.stdout.fileno(), False)
    selector.register(process.stdout, selectors.EVENT_READ)
    data = bytearray()
    deadline = time.monotonic() + 10
    eof = False
    timed_out = False
    try:
        while len(data) <= LIMIT and not eof:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            for key, _ in selector.select(min(remaining, 0.25)):
                chunk = os.read(key.fd, min(16384, LIMIT + 1 - len(data)))
                if chunk:
                    data.extend(chunk)
                else:
                    eof = True
        if not eof:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
    finally:
        selector.close()
        process.stdout.close()
    return {"command": argv, "returncode": process.returncode,
            "output": bytes(data[:LIMIT]).decode("utf-8", errors="replace"),
            "truncated": len(data) > LIMIT, "timed_out": timed_out}


def link_output(argv):
    output = None
    for index, value in enumerate(argv):
        if value in ("-o", "--output") and index + 1 < len(argv):
            output = argv[index + 1]
        elif value.startswith("--output="):
            output = value.split("=", 1)[1]
    return Path(output).absolute() if output else None


def capture(argv):
    output = link_output(argv)
    directory = os.environ.get("SEISSOL_LINK_METADATA_DIR")
    if not directory or output is None or output.name != "SeisSol-serial-test":
        return
    with output.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("linked test output is not ELF")
        stream.seek(0)
        digest_state = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_state.update(chunk)
        digest = digest_state.hexdigest()
    root = Path(directory)
    if not root.is_absolute():
        raise ValueError("metadata directory must be absolute")
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise ValueError("metadata directory must belong to the caller")
    readelf, ldd = shutil.which("readelf"), shutil.which("ldd")
    if readelf is None or ldd is None:
        raise ValueError("readelf and ldd must be available")
    loader = inspect_command([ldd, str(output)])
    dependencies = []
    seen = set()
    for match in re.finditer(r"=>\s+(/[^\s()]+)", loader["output"]):
        path = Path(match.group(1))
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if path.name.startswith(("libomp", "libgomp", "libacpp", "libsycl",
                                 "libLLVM", "libur_", "libSeisSol", "libdevice")):
            dependencies.append({"path": str(path),
                                 "dynamic": inspect_command([readelf, "-d", str(path)])})
            if len(dependencies) == 8:
                break
    record = {"schema": 1, "compiler_status": 0, "output": str(output),
              "elf_sha256": digest,
              "argv_sha256": hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
              "ld_library_path_present": bool(os.environ.get("LD_LIBRARY_PATH")),
              "dynamic": inspect_command([readelf, "-d", str(output)]),
              "loader": loader, "dependencies": dependencies}
    destination = root / ("serial-test-link-" + str(os.getpid()) + ".json")
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    argv = sys.argv[1:]
    if not argv:
        return 64
    status = subprocess.call(argv)
    if status == 0:
        try:
            capture(argv)
        except Exception as error:
            print("SeisSol link metadata: " + str(error), file=sys.stderr)
    if status < 0:
        signum = -status
        if signum not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    return status


if __name__ == "__main__":
    sys.exit(main())
