# Running RECOVAR on miniSEED files

---

## Prerequisites (one-time setup per station)

**1. Start scmaster** (needed for scautopick to read inventory):
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

## Running

```bash
seiscomp exec recovar_playback --input /path/to/mseeds --output scored_picks.csv
```

That's it. The module detects picks with scautopick, scores each one with
RecovAR, and writes the results to CSV.

---

## Output

`scored_picks.csv` — one row per scored pick:

```
pick_id,pick_time,net,sta,loc,cha,score
Pick/...,2022-09-19T18:11:07Z,IU,ANMO,00,HHZ,0.9941
```

A score near **1** = high-confidence seismic signal. Near **0** = noise.

---

## Building test data

To download a mixed set of earthquake and noise waveforms from IRIS:

```bash
~/recovar-seiscomp/bin/python3 ~/recovar/seiscomp_integration/create_test_archive.py \
    --output ~/seiscomp_test/sds
```

Then copy the day files into a flat folder to use with `recovar_playback`:

```bash
find ~/seiscomp_test/sds -type f | xargs -I{} cp {} ~/seiscomp_test/flat/
seiscomp exec recovar_playback --input ~/seiscomp_test/flat --output results.csv
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No miniSEED files found` | Check file extensions: `.mseed`, `.ms`, or `.miniseed` |
| `0 pick(s) detected` | Re-run `seiscomp update-config scautopick`; confirm scmaster is running |
| Waveform unavailable for a pick | The pick window extends before the file start — normal for picks near the file boundary |
