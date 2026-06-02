# Running SeisComP + RECOVAR on a folder of miniSEED files

---

## Prerequisites (one-time setup)

These steps only need to be done once after installation.

**Load station inventory:**
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

**Create scautopick bindings:**
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

**1. Start SeisComP and RECOVAR:**
```bash
source ~/.bashrc
seiscomp start scmaster scdb recovar_pick_filter
```

**2. Run the playback module on your miniSEED folder:**
```bash
seiscomp exec recovar_playback --input /path/to/mseeds --output scored_picks.csv
```

That's it. The module handles everything: concatenating the files, picking, scoring, and exporting.

---

## Output

`scored_picks.csv` — one row per scored pick:

```
pick_id,pick_time,net,sta,loc,cha,score
Pick/...,2022-09-19 18:11:07,IU,ANMO,00,HHZ,0.9941
```

A score near **1** indicates a high-confidence seismic signal. Near **0** indicates noise.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No miniSEED files found` | Check file extensions are `.mseed`, `.ms`, or `.miniseed` |
| `recovar_pick_filter did not reach ready state` | Check `~/.seiscomp/log/recovar_pick_filter.log` |
| `No stations added` in scautopick | Re-run `seiscomp update-config scautopick` |
| 0 scored picks in output | Confirm `scdb` is running: `seiscomp status scdb` |
