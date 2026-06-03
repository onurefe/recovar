# Running RECOVAR with SeisComP

---

## Prerequisites (one-time setup per station)

**1. Start scmaster:**
```bash
seiscomp start scmaster
```

**2. Load station inventory:**
```bash
~/recovar-seiscomp/bin/python3 -c "
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
inv = Client('IRIS').get_stations(
    network='IU', station='ANMO', location='00', channel='HH?',
    level='response',
    starttime=UTCDateTime('2018-01-01'), endtime=UTCDateTime('2025-01-01'))
inv.write('/tmp/inventory.xml', format='STATIONXML')
" 2>/dev/null
seiscomp exec import_inv fdsnxml /tmp/inventory.xml ~/seiscomp/etc/inventory/station.xml
seiscomp exec scinv sync --filebase ~/seiscomp/etc/inventory/ \
    -d mysql://sysop:sysop@localhost/seiscomp
```

**3. Configure scautopick bindings:**
```bash
cat > ~/seiscomp/etc/key/station_IU_ANMO << 'EOF'
global:default
scautopick:default
EOF

mkdir -p ~/seiscomp/etc/key/scautopick
cat > ~/seiscomp/etc/key/scautopick/profile_default << 'EOF'
detecStream    = HH
detecLocid     = 00
detecFilter    = "RMHP(10)>>ITAPER(30)>>BW(4,1,20)>>STALTA(0.5,10)"
trigOn         = 3.0
trigOff        = 1.5
timeCorr       = -0.8
picker         = AIC
useSquaredness = true
EOF

seiscomp update-config scautopick
```

---

## Starting the integration

```bash
source ~/.bashrc
seiscomp start scmaster scdb recovar_pick_filter
```

Wait for the model to load (~30 s):
```bash
tail -f ~/.seiscomp/log/recovar_pick_filter.log
# Expected: recovar_pick_filter: ready
```

Once ready, `recovar_pick_filter` listens on the PICK messaging group.
Every pick detected by scautopick receives a `recovar_score:[0–1]` comment
attached to the pick object and persisted to the database by scdb.

---

## Module management

```bash
seiscomp start   recovar_pick_filter
seiscomp stop    recovar_pick_filter
seiscomp status  recovar_pick_filter
seiscomp enable  recovar_pick_filter   # auto-start with seiscomp start
seiscomp disable recovar_pick_filter
```

---

## Testing with the test archive

Use `create_test_archive.py` to download a set of earthquake and noise
waveforms from IRIS into an SDS archive:

```bash
~/recovar-seiscomp/bin/python3 ~/recovar/seiscomp_integration/create_test_archive.py \
    --output ~/seiscomp_test/sds
```

Point the init descriptor's `record_stream` at the SDS archive by editing
`~/seiscomp/etc/init/recovar_pick_filter.py` and changing:

```python
record_stream = "sdsarchive:///home/<user>/seiscomp_test/sds"
```

Restart the daemon to pick up the change:

```bash
seiscomp stop recovar_pick_filter && seiscomp start recovar_pick_filter
tail -f ~/.seiscomp/log/recovar_pick_filter.log
# Wait for: recovar_pick_filter: ready
```

Inject picks by running scautopick in playback mode over the archive:

```bash
find ~/seiscomp_test/sds -type f | sort | xargs cat > /tmp/playback.mseed

seiscomp exec scautopick \
    --playback \
    -I file:///tmp/playback.mseed \
    -d mysql://sysop:sysop@localhost/seiscomp
```

Check the recovar log for scored picks:

```bash
grep "recovar_score" ~/.seiscomp/log/recovar_pick_filter.log
```

Expected output (scores vary by event and epicentral distance):
```
recovar: Pick/... recovar_score=0.6058
recovar: Pick/... recovar_score=0.7436
```

Restore the live record stream when done:

```python
record_stream = "slink://localhost:18000"
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No stations added` in scautopick | Re-run `seiscomp update-config scautopick` |
| `waveform unavailable` in recovar log | Check `record_stream` in the init descriptor |
| No scored picks in database | Confirm `scdb` is running: `seiscomp status scdb` |
| `recovar_pick_filter` not starting | Check log: `~/.seiscomp/log/recovar_pick_filter.log` |
