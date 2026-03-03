import argparse
import os
import time
import shutil
from pathlib import Path

def default_common_files_dir() -> Path:
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        raise RuntimeError("APPDATA not set; cannot locate MetaQuotes Common\\Files")
    return Path(appdata) / "MetaQuotes" / "Terminal" / "Common" / "Files"

def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    # Copy bytes to temp, then atomic replace
    with src.open("rb") as fsrc, tmp.open("wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)
        fdst.flush()
        os.fsync(fdst.fileno())
    os.replace(tmp, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="", help="Full path to MT5 Common\\Files CSV. If empty, uses APPDATA default + filename.")
    ap.add_argument("--src_name", type=str, default="eurusd_m5_latest.csv", help="Filename inside Common\\Files when --src not provided.")
    ap.add_argument("--dst", type=str, required=True, help="Destination path inside repo.")
    ap.add_argument("--interval", type=float, default=2.0, help="Polling interval seconds.")
    ap.add_argument("--once", action="store_true", help="Copy once and exit.")
    args = ap.parse_args()

    if args.src:
        src = Path(args.src)
    else:
        src = default_common_files_dir() / args.src_name

    dst = Path(args.dst)

    print(f"[sync] src = {src}")
    print(f"[sync] dst = {dst}")
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    last_sig = None
    while True:
        st = src.stat()
        sig = (st.st_size, int(st.st_mtime))
        if sig != last_sig:
            atomic_copy(src, dst)
            print(f"[sync] copied size={st.st_size} mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}")
            last_sig = sig

        if args.once:
            break

        time.sleep(args.interval)

if __name__ == "__main__":
    main()