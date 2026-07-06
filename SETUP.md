# DeepSwing — Raspberry Pi 5 Setup Guide

> **Rebuilding after an SD-card failure?** Follow this top to bottom. The failure
> predated the offsite backup, so there's nothing to restore — you come up on a
> clean sim (both tracks at 100,000 SEK, no heuristics, no history), which is the
> fair starting point anyway after the risk/exit/sizing changes. Once you're
> running, set up the Google Drive backup (§4b) so the next card failure is a
> 10-minute restore instead of a total loss.

## Hardware

| Component | Recommendation |
|---|---|
| Raspberry Pi 5 | 8 GB RAM model (MIPRO optimization uses ~600 MB) |
| **Storage** | **USB SSD strongly recommended over microSD** — constant 15-min writes wear microSD out; this is what killed the last card. If you must use microSD, A2 class, 64 GB+, and rely on the offsite backup |
| Power | Official Pi 5 PSU (5V 5A USB-C) — cheap PSUs cause instability |
| Case | Any with active cooling (fan or heatsink) — Pi 5 throttles under load |

---

## 1. OS Setup

Download **Raspberry Pi OS Lite (64-bit)** — Debian Bookworm.
Use Raspberry Pi Imager to flash. Before writing, click the gear icon and set:
- Hostname: `deepswing`
- **Username: `alexander`** — the systemd units and scripts assume this user and
  the path `/home/alexander/Documents/DeepSwing`. If you pick a different name,
  update `systemd/*.service`, `systemd/*.timer`, and `deploy/deploy.sh` to match.
- Enable SSH (password or key)
- Set your WiFi credentials (or use Ethernet)

Boot, SSH in:

```bash
ssh alexander@deepswing.local
```

Update:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git sqlite3
```

Raspberry Pi OS Bookworm ships with Python 3.11 — verify with `python3 --version`.

---

## 2. Clone DeepSwing + the FinanceData dependency

DeepSwing depends on **`financedata`**, a sibling library that is **not on PyPI** —
it must be cloned next to DeepSwing and installed editable. Both live under
`~/Documents`:

```bash
mkdir -p ~/Documents && cd ~/Documents

# The shared data library (OHLCV, news, FX, VIX, fundamentals)
git clone https://github.com/ACWesterberg/FinanceData.git

# DeepSwing itself — the deploy branch is the live line
git clone https://github.com/ACWesterberg/DeepSwing.git
cd DeepSwing
git checkout deploy
```

> If your `FinanceData` repo has a different name/URL, clone it as
> `~/Documents/FinanceData` anyway (or edit the `../FinanceData` path in
> `deploy/deploy.sh`) — the editable install expects it there.

---

## 3. Virtualenv + dependencies

```bash
cd ~/Documents/DeepSwing
python3 -m venv venv
venv/bin/pip install --upgrade pip setuptools wheel
venv/bin/pip install -e ../FinanceData
venv/bin/pip install -r requirements.txt
```

> **If the `ta` or `sgmllib3k` wheels fail to build** with an `install_layout`
> error (a known Debian/setuptools quirk), force PEP 517 for them:
> `venv/bin/pip install --use-pep517 ta sgmllib3k` then re-run the requirements
> install.

Create your `.env` and fill in keys:

```bash
cp .env.example .env
nano .env
```

Required for the sim to trade: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. Strongly
recommended: `ALPHA_VANTAGE_API_KEY` (Nordic OHLCV), `NEWS_API_KEY` (news),
`FRED_API_KEY` (US macro). Everything else has a free fallback or a default.

**Before exposing the dashboard, set these two** (the tunnel makes it public):
- `DASHBOARD_PASSWORD=...` — enables login + WebSocket auth
- `RESET_PIN=...` — change it off the committed default

Smoke-test it starts (Ctrl+C to stop):

```bash
venv/bin/python main.py
# visit http://deepswing.local:8000 from another device on the LAN
```

---

## 4. Run as a systemd Service (autostart on boot)

The unit file already targets `alexander` / `/home/alexander/Documents/DeepSwing`:

```bash
sudo cp ~/Documents/DeepSwing/systemd/deepswing.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now deepswing
```

Check status and logs:

```bash
sudo systemctl status deepswing
journalctl -u deepswing -f       # follow live logs
```

---

## 4b. Offsite Backup to Google Drive (rclone)

**Set this up now, before you rely on the sim.** The app writes a nightly local
snapshot to `data/backups/`, but that lives on the same SD card — it protects
against a bad DB write, *not* against card failure (that's what wiped the last
install). This copies the portfolio DB, heuristics, compiled MIPRO programs, and
(optionally) `.env` to Google Drive every night, independently of the app.

Install rclone and create a Google Drive remote named `gdrive`:

```bash
sudo apt install -y rclone
rclone config
#   n) new remote  →  name: gdrive  →  storage: drive (Google Drive)
#   Accept defaults; for a headless Pi choose "N" at the auto-config prompt
#   and run the printed `rclone authorize` command on a machine with a browser,
#   then paste the token back.
rclone mkdir gdrive:DeepSwingBackups
```

Point the backup at that remote via an environment file:

```bash
sudo tee /etc/default/deepswing-backup >/dev/null <<'EOF'
RCLONE_REMOTE=gdrive:DeepSwingBackups
BACKUP_KEEP=14
BACKUP_INCLUDE_ENV=true
EOF
```

Install and enable the nightly timer:

```bash
sudo cp ~/Documents/DeepSwing/systemd/deepswing-backup.service /etc/systemd/system/
sudo cp ~/Documents/DeepSwing/systemd/deepswing-backup.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now deepswing-backup.timer
```

Verify it works right now (don't wait for midnight):

```bash
sudo systemctl start deepswing-backup.service   # run once immediately
journalctl -u deepswing-backup -n 20            # see the upload line
rclone ls gdrive:DeepSwingBackups               # confirm the archive landed
systemctl list-timers deepswing-backup.timer    # confirm next run is scheduled
```

**Restoring on a future rebuild** (after §2–3, before starting the service):

```bash
cd ~/Documents/DeepSwing
export RCLONE_REMOTE=gdrive:DeepSwingBackups
./deploy/restore_from_gdrive.sh                 # newest archive
# or: ./deploy/restore_from_gdrive.sh deepswing_backup_20260702_233000.tar.gz
sudo systemctl start deepswing
```

> The rclone config lives in the service user's home (`~/.config/rclone/`), so
> run `rclone config` as `alexander`, not root.

---

## 5. Custom Domain via Cloudflare Tunnel

Cloudflare Tunnel exposes DeepSwing on your own domain (`trading.yourdomain.com`)
with no open firewall ports, no static IP, no VPS.

**Prerequisites:** a domain added to Cloudflare (free plan is fine) + a Cloudflare account.

### Install cloudflared on the Pi

```bash
curl -L https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg > /dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared bookworm main' | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install cloudflared
```

### Authenticate

```bash
cloudflared tunnel login
```

Opens a browser link — paste it on your computer, log in to Cloudflare, select your domain.

### Create the tunnel

```bash
cloudflared tunnel create deepswing
```

Note the tunnel UUID printed (looks like `abc123-...`).

### Configure the tunnel

```bash
mkdir -p ~/.cloudflared
nano ~/.cloudflared/config.yml
```

Paste (replace `YOUR_TUNNEL_UUID` and `trading.yourdomain.com`):

```yaml
tunnel: YOUR_TUNNEL_UUID
credentials-file: /home/alexander/.cloudflared/YOUR_TUNNEL_UUID.json

ingress:
  - hostname: trading.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

### Add the DNS record

```bash
cloudflared tunnel route dns deepswing trading.yourdomain.com
```

Creates a CNAME pointing to your tunnel; Cloudflare handles SSL automatically.

### Run as a systemd service

```bash
sudo cloudflared service install
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
```

Your dashboard is now live at `https://trading.yourdomain.com`.

```
Browser → Cloudflare Edge → Encrypted tunnel → Pi → DeepSwing :8000
```

No router ports open; traffic encrypted in transit; Cloudflare is the reverse proxy.

---

## 6. Optional: Restrict Access

The dashboard has its own password auth (§3), but you can add Cloudflare Access
(free for 1 seat) as a second layer:

1. Cloudflare dashboard → Zero Trust → Access → Applications
2. Add Application → Self-hosted
3. Domain: `trading.yourdomain.com`
4. Policy: Email → `your@email.com`

---

## 7. Updating DeepSwing

`deploy/deploy.sh` fetches the `deploy` branch, reinstalls deps, and restarts:

```bash
cd ~/Documents/DeepSwing
./deploy/deploy.sh
```

Or manually:

```bash
cd ~/Documents/DeepSwing
git fetch origin deploy && git reset --hard origin/deploy
venv/bin/pip install -e ../FinanceData -q
venv/bin/pip install -r requirements.txt -q
sudo systemctl restart deepswing
```

---

## Quick Reference

| Task | Command |
|---|---|
| Start / stop / restart | `sudo systemctl {start,stop,restart} deepswing` |
| Live logs | `journalctl -u deepswing -f` |
| Run backup now | `sudo systemctl start deepswing-backup.service` |
| List cloud backups | `rclone ls gdrive:DeepSwingBackups` |
| Restore latest | `RCLONE_REMOTE=gdrive:DeepSwingBackups ./deploy/restore_from_gdrive.sh` |
| Reset sim (fresh start) | `curl -X POST localhost:8000/api/reset -H 'Content-Type: application/json' -d '{"pin":"<PIN>"}'` |
| Tunnel status | `sudo systemctl status cloudflared` |
| Dashboard | `https://trading.yourdomain.com` (or `http://deepswing.local:8000` on LAN) |
