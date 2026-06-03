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

Add to `~/.bashrc` and source it:

```bash
export SEISCOMP_ROOT=~/seiscomp
export PATH=/usr/bin:$SEISCOMP_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$SEISCOMP_ROOT/lib
```

---

## Step 3 — Run SeisComP setup

```bash
seiscomp setup
```

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

---

## Step 4 — Clone RECOVAR and run the installer

```bash
git clone git@github.com:onurefe/recovar.git ~/recovar
cd ~/recovar && git checkout seiscomp-integration
bash ~/recovar/seiscomp_integration/install.sh
```

The installer creates the Python venv at `~/recovar-seiscomp`, installs
dependencies (tensorflow, obspy, scipy, numpy), and installs the
`recovar_pick_filter` daemon into `$SEISCOMP_ROOT/bin/`.

It will print `All imports OK` at the end if everything succeeded.
