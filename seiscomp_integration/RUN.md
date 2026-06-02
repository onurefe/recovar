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

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No stations added` in scautopick | Re-run `seiscomp update-config scautopick` |
| `waveform unavailable` in recovar log | Check `recordStream` in `~/seiscomp/etc/recovar_pick_filter.cfg` |
| No scored picks in database | Confirm `scdb` is running: `seiscomp status scdb` |
| `recovar_pick_filter` not starting | Check log: `~/.seiscomp/log/recovar_pick_filter.log` |
