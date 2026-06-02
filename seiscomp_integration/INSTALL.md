# RECOVAR — SeisComP Integration: Installation Guide

Tested on **Ubuntu 22.04** with **SeisComP 7.x**.

---

## Step 1 — Install system packages

```bash
sudo apt-get install -y libboost-program-options1.74.0 mariadb-server mariadb-client
sudo systemctl start mariadb && sudo systemctl enable mariadb
```

---

## Step 2 — Install SeisComP

Download `seiscomp-7.x.x-ubuntu22.04-x86_64.tar.gz` from `seiscomp.de/downloader/` (free account), then:

```bash
tar xzf ~/Downloads/seiscomp-7.x.x-ubuntu22.04-x86_64.tar.gz -C ~/
```

Add the following to `~/.bashrc`, then run `source ~/.bashrc`:

```bash
export SEISCOMP_ROOT=~/seiscomp
export PATH=/usr/bin:$SEISCOMP_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$SEISCOMP_ROOT/lib
```

```bash
source ~/.bashrc
```

---

## Step 3 — Run SeisComP setup

```bash
seiscomp setup
```

When prompted, use these values (press Enter for anything not listed):

| Prompt | Value |
|---|---|
| Agency ID | `TEST` |
| Enable database storage | `yes` |
| Database backend | `0` (mysql/mariadb) |
| Create the SeisComP database | `yes` |
| Run as super user | `yes` |
| RW user / password | `sysop` / `sysop` |
| Public hostname | `localhost` |
| RO user / password | `sysop` / `sysop` |
| Final prompt | `P` |

Then start the message server:

```bash
seiscomp start scmaster
```

---

## Step 4 — Clone RECOVAR and run the installer

```bash
git clone git@github.com:onurefe/recovar.git ~/recovar
cd ~/recovar && git checkout seiscomp-integration
```

Then run the installer script — it handles the venv, dependencies, pick filter installation, init descriptor, and all config files automatically:

```bash
bash ~/recovar/seiscomp_integration/install.sh
```

To use a custom waveform source pass `--record-stream`:

```bash
bash ~/recovar/seiscomp_integration/install.sh --record-stream sdsarchive:///data/sds
```

The script will print `All imports OK` at the end if everything succeeded. You can re-run it safely at any time — it skips steps that are already done.

### What the installer does

| Step | Action |
|---|---|
| 1 | Installs system packages (libboost, mariadb) |
| 2 | Creates Python 3.10 venv at `~/recovar-seiscomp` with tensorflow, numpy, scipy, obspy |
| 3 | Installs `recovar_pick_filter`, `recovar_batch_test`, and `recovar_playback` binaries to `$SEISCOMP_ROOT/bin/` |
| 3 | Installs the SeisComP init descriptor to `$SEISCOMP_ROOT/etc/init/` and enables the module |
| 4 | Writes `recovar_pick_filter.cfg` and `~/.seiscomp/global.cfg` |
| 5 | Appends SeisComP + RECOVAR environment variables to `~/.bashrc` |

---

## Step 5 — Run

```bash
source ~/.bashrc
seiscomp start recovar_pick_filter
```

Because the module is registered with SeisComP's process manager, you can also start it together with all other enabled modules:

```bash
seiscomp start
```

Watch the log for the ready message:

```bash
tail -f ~/.seiscomp/log/recovar_pick_filter.log
```

Expected:
```
Connection to localhost/production established
recovar_pick_filter: ready
```

TensorFlow model loading takes up to ~30 seconds on a CPU-only machine.

The module subscribes to the PICK messaging group and attaches a `recovar_score:[0–1]` comment to every incoming pick. Scores are persisted to the SeisComP MariaDB database by `scdb` alongside the original pick.

### Module management

```bash
seiscomp start   recovar_pick_filter
seiscomp stop    recovar_pick_filter
seiscomp status  recovar_pick_filter
seiscomp disable recovar_pick_filter   # prevent auto-start
seiscomp enable  recovar_pick_filter   # re-enable auto-start
```

---

## Exporting scored picks

Scored picks are stored in the SeisComP MariaDB database. Use the export script to query them:

```bash
# All scored picks → scored_picks.csv
~/recovar-seiscomp/bin/python3 ~/recovar/seiscomp_integration/export_scored_picks.py

# Filter by date range
~/recovar-seiscomp/bin/python3 ~/recovar/seiscomp_integration/export_scored_picks.py \
    --start 2024-01-01 --end 2024-12-31

# Only high-confidence picks
~/recovar-seiscomp/bin/python3 ~/recovar/seiscomp_integration/export_scored_picks.py \
    --min-score 0.7 --output high_confidence.csv
```

Output CSV columns: `pick_id, pick_time, net, sta, loc, cha, score`

---

## Testing

### Quick checks

**1. scmaster is running:**
```bash
seiscomp check scmaster
```
Expected: `Summary: 1 started modules checked`

**2. All Python imports work:**
```bash
~/recovar-seiscomp/bin/python3 -c "
import seiscomp.client, seiscomp.datamodel, seiscomp.io
import tensorflow, recovar, recovar_scorer
print('All imports OK — tensorflow', tensorflow.__version__)
"
```
Expected: `All imports OK — tensorflow 2.14.0`

**3. Pick filter reaches ready state:**

Start the pick filter (Step 5) and wait up to 30 seconds for:
```
recovar_pick_filter: ready
```

### Batch scoring validation

Validates the model's discrimination ability by scoring real earthquake P-arrivals fetched from IRIS and synthetic noise waveforms. Requires internet access.

```bash
seiscomp exec recovar_batch_test
```

Or with options:

```bash
~/recovar-seiscomp/bin/python3 ~/recovar/seiscomp_integration/batch_score_test.py \
    --events 50 --min-mag 6.5 --output batch_results.csv
```

Expected output (values will vary):
```
Earthquakes (n=~10): mean≈0.52  std≈0.28  min≈0.08  max≈0.90
Noise       (n=10):  mean≈0.07  std≈0.15  min≈0.00  max≈0.51
Accuracy (eq≥0.5, noise<0.5): ~70%
```

Earthquake scores vary with epicentral distance — near-regional events (< 30°) score higher than teleseismic events (> 80°) because the P-wave shape differs. This is expected behaviour, not a bug.

### End-to-end test

This injects a real pick into SeisComP messaging and checks that the pick filter responds with a score.

**Download test waveform data** (3-component, 100 Hz, 150 s around the 2019 M7.1 Ridgecrest earthquake P-wave):
```bash
mkdir -p ~/seiscomp_test/data
wget -O ~/seiscomp_test/data/IU.ANMO.HH.mseed \
  "https://service.iris.edu/fdsnws/dataselect/1/query?network=IU&station=ANMO&location=00&channel=HH%3F&starttime=2019-07-06T03:20:30&endtime=2019-07-06T03:23:00&format=miniseed"
```

**Point the pick filter at the test data** — edit `~/seiscomp/etc/recovar_pick_filter.cfg`:
```
recordStream = file:///home/<user>/seiscomp_test/data/IU.ANMO.HH.mseed
```

**Start the filter and run the test in one shot:**
```bash
LOG=~/.seiscomp/log/recovar_pick_filter.log
LOG_OFFSET=$(wc -c < "$LOG" 2>/dev/null || echo 0)

~/recovar-seiscomp/bin/python3 ~/seiscomp/bin/recovar_pick_filter &

until python3.10 -c "
import sys
with open('$LOG') as f:
    f.seek($LOG_OFFSET)
    sys.exit(0 if any('pick_filter: ready' in l for l in f) else 1)
" 2>/dev/null; do sleep 1; done

~/recovar-seiscomp/bin/python3 ~/recovar/seiscomp_integration/test_pick_injection.py
```

Expected output:
```
Sent pick recovar_test_pick_<timestamp>
  station : IU.ANMO.00.HHZ
  time    : 2019-07-06T03:22:00.000Z
Waiting up to 60s for recovar_score in log...

result : recovar_score:0.9444
PASS   — score 0.9444 >= 0.5 (high-confidence seismic signal)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `All imports` check fails | Check that PYTHONPATH in `~/.bashrc` has correct paths and has been sourced |
| `No plugins loaded` / app hangs | Ensure `core.plugins = dbmysql` is in `~/.seiscomp/global.cfg` |
| `mysql://mysql://` in log | Remove `mysql://` prefix from `dbstore.read/write` in `~/seiscomp/etc/scmaster.cfg` |
| `Client name not unique` | A stale instance is running: `seiscomp stop recovar_pick_filter` |
| `recovar_pick_filter: not running` after `seiscomp start` | Check log at `~/.seiscomp/log/recovar_pick_filter.log` for model load errors |
