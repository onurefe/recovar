#!/usr/bin/env python3
"""
recovar_playback — Run the full scautopick + recovar pipeline over a
folder of miniSEED files and export scored picks to CSV.

Usage:
    seiscomp exec recovar_playback --input /path/to/mseeds
    seiscomp exec recovar_playback --input /path/to/mseeds --output results.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import time

SEISCOMP_ROOT = os.environ.get("SEISCOMP_ROOT", os.path.expanduser("~/seiscomp"))
CFG_FILE      = os.path.join(SEISCOMP_ROOT, "etc", "recovar_pick_filter.cfg")
LOG_FILE      = os.path.expanduser("~/.seiscomp/log/recovar_pick_filter.log")
DB_URL        = "mysql://sysop:sysop@localhost/seiscomp"

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, **kwargs):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)


def seiscomp(args):
    return run(f"seiscomp {args}")


def log(msg):
    print(f"  {msg}")


def update_record_stream(path):
    """Replace the recordStream line in recovar_pick_filter.cfg."""
    with open(CFG_FILE) as f:
        content = f.read()
    content = re.sub(r"(?m)^recordStream\s*=.*$",
                     f"recordStream      = {path}", content)
    with open(CFG_FILE, "w") as f:
        f.write(content)


def wait_for_ready(timeout=60):
    """Wait until recovar_pick_filter logs 'ready'."""
    log_offset = os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0
    deadline   = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                f.seek(log_offset)
                if any("pick_filter: ready" in line for line in f):
                    return True
        time.sleep(1)
    return False


def wait_for_scores(n_picks, timeout=120):
    """Wait until at least n_picks recovar_score lines appear in the log."""
    log_offset = os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0
    deadline   = time.time() + timeout
    while time.time() < deadline:
        with open(LOG_FILE) as f:
            f.seek(log_offset)
            scored = sum(1 for l in f if "recovar_score=" in l)
        if scored >= n_picks:
            return scored
        time.sleep(1)
    return scored


def count_picks_in_log(log_offset):
    """Count recovar_score lines written after log_offset."""
    if not os.path.exists(LOG_FILE):
        return 0
    with open(LOG_FILE) as f:
        f.seek(log_offset)
        return sum(1 for l in f if "recovar_score=" in l)


def export_csv(output):
    import pymysql as MySQLdb
    conn = MySQLdb.connect(host="localhost", user="sysop",
                           password="sysop", database="seiscomp")
    query = """
        SELECT
            po.publicID,
            CONCAT(p.time_value, '.', LPAD(p.time_value_ms, 3, '0')),
            p.waveformID_networkCode,
            p.waveformID_stationCode,
            p.waveformID_locationCode,
            p.waveformID_channelCode,
            CAST(REPLACE(c.text, 'recovar_score:', '') AS DECIMAL(6,4))
        FROM Pick p
        JOIN PublicObject po ON po._oid = p._oid
        JOIN Comment c       ON c._parent_oid = p._oid
        WHERE c.text LIKE 'recovar_score:%%'
        ORDER BY p.time_value
    """
    with conn:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pick_id", "pick_time", "net", "sta", "loc", "cha", "score"])
        w.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run scautopick + recovar over a folder of miniSEED files.")
    p.add_argument("--input",  required=True,
                   help="Folder containing miniSEED files (*.mseed / *.MSEED)")
    p.add_argument("--output", default="scored_picks.csv",
                   help="Output CSV file (default: scored_picks.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.expanduser(args.input)

    if not os.path.isdir(input_dir):
        sys.exit(f"Input folder not found: {input_dir}")

    # ── 1. Collect miniSEED files ─────────────────────────────────────────────
    mseed_files = sorted(
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith((".mseed", ".ms", ".miniseed"))
    )
    if not mseed_files:
        sys.exit(f"No miniSEED files found in {input_dir}")

    print(f"[1/5] Found {len(mseed_files)} miniSEED file(s) in {input_dir}")

    # ── 2. Concatenate into a single playback file ────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".mseed", delete=False)
    tmp.close()
    playback = tmp.name

    print(f"[2/5] Concatenating into {playback} ...")
    with open(playback, "wb") as out:
        for path in mseed_files:
            with open(path, "rb") as f:
                out.write(f.read())

    record_url = f"file://{playback}"

    # ── 3. Point recovar at the playback file and restart ────────────────────
    print(f"[3/5] Updating recovar config → {record_url}")
    original_stream = None
    with open(CFG_FILE) as f:
        for line in f:
            if line.strip().startswith("recordStream"):
                original_stream = line.strip()
                break

    update_record_stream(record_url)
    seiscomp("stop recovar_pick_filter")
    time.sleep(1)
    seiscomp("start recovar_pick_filter")

    print("       Waiting for model to load (~30 s) ...")
    if not wait_for_ready(timeout=90):
        sys.exit("recovar_pick_filter did not reach ready state. "
                 f"Check {LOG_FILE}")
    print("       recovar_pick_filter: ready")

    # ── 4. Run scautopick ─────────────────────────────────────────────────────
    log_offset = os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0

    print(f"[4/5] Running scautopick on {playback} ...")
    result = run(
        f"seiscomp exec scautopick --playback -I {record_url} -d {DB_URL}"
    )
    if result.returncode not in (0, 1):
        print(f"       scautopick exited with code {result.returncode}")

    # Count how many picks scautopick emitted from its output
    n_emitted = result.stderr.count("emit detection") + \
                result.stdout.count("emit detection")

    # Wait for recovar to score them (up to 60 s)
    n_scored = wait_for_scores(max(n_emitted, 1), timeout=60)
    print(f"       scautopick emitted picks, recovar scored {n_scored}")

    # ── 5. Export CSV ─────────────────────────────────────────────────────────
    print(f"[5/5] Exporting results to {args.output} ...")
    n_rows = export_csv(args.output)
    print(f"       {n_rows} scored pick(s) written to {args.output}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    os.unlink(playback)
    if original_stream:
        with open(CFG_FILE) as f:
            content = f.read()
        content = re.sub(r"(?m)^recordStream\s*=.*$", original_stream, content)
        with open(CFG_FILE, "w") as f:
            f.write(content)
        seiscomp("stop recovar_pick_filter")
        time.sleep(1)
        seiscomp("start recovar_pick_filter")

    print("\nDone.")


if __name__ == "__main__":
    main()
