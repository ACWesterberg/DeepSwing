# DeepSwing — Raspberry Pi 5 Setup Guide

## Hardware

| Component | Recommendation |
|---|---|
| Raspberry Pi 5 | 8 GB RAM model (MIPRO optimization uses ~600 MB) |
| Storage | 64 GB+ microSD A2 class, or USB SSD for reliability |
| Power | Official Pi 5 PSU (5V 5A USB-C) — cheap PSUs cause instability |
| Case | Any with active cooling (fan or heatsink) — Pi 5 throttles under load |

---

## 1. OS Setup

Download **Raspberry Pi OS Lite (64-bit)** — Debian Bookworm.
Use Raspberry Pi Imager to flash. Before writing, click the gear icon and set:
- Hostname: `deepswing`
- Enable SSH (password or key)
- Set your WiFi credentials (or use Ethernet)

Boot, SSH in:

```bash
ssh pi@deepswing.local
```

Update:

```bash
sudo apt update && sudo apt upgrade -y
```

---

## 2. Python 3.11

Raspberry Pi OS Bookworm ships with Python 3.11 — verify:

```bash
python3 --version   # should print 3.11.x
```

Install pip and venv:

```bash
sudo apt install -y python3-pip python3-venv git
```

---

## 3. Deploy DeepSwing

Clone the repo:

```bash
cd ~
git clone https://github.com/ACWesterberg/DeepSwing.git
cd DeepSwing
```

Create virtualenv and install dependencies:

```bash
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

Create your `.env` from the example:

```bash
cp .env.example .env
nano .env   # fill in your API keys
```

Test that it starts:

```bash
venv/bin/python main.py
```

Visit `http://deepswing.local:8000` from another device on your network — the dashboard should load.

Press `Ctrl+C` to stop.

---

## 4. Run as a systemd Service (autostart on boot)

Copy the service file:

```bash
sudo cp ~/DeepSwing/systemd/deepswing.service /etc/systemd/system/deepswing.service
```

Edit the paths if you cloned somewhere other than `/home/pi/DeepSwing`:

```bash
sudo nano /etc/systemd/system/deepswing.service
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable deepswing
sudo systemctl start deepswing
```

Check status and logs:

```bash
sudo systemctl status deepswing
journalctl -u deepswing -f       # follow live logs
```

---

## 5. Custom Domain via Cloudflare Tunnel

Cloudflare Tunnel lets you expose DeepSwing on your own domain (`trading.yourdomain.com`) without:
- Opening firewall ports
- Having a static IP
- Paying for a VPS

**Prerequisites:**
- A domain added to Cloudflare (free plan is fine)
- A Cloudflare account

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

This opens a browser link — paste it on your computer, log in to Cloudflare, select your domain. A credentials file will be saved on the Pi.

### Create the tunnel

```bash
cloudflared tunnel create deepswing
```

Note the tunnel UUID printed (looks like `abc123-...`).

### Configure the tunnel

Create the config file:

```bash
mkdir -p ~/.cloudflared
nano ~/.cloudflared/config.yml
```

Paste (replace `YOUR_TUNNEL_UUID` and `trading.yourdomain.com`):

```yaml
tunnel: YOUR_TUNNEL_UUID
credentials-file: /home/pi/.cloudflared/YOUR_TUNNEL_UUID.json

ingress:
  - hostname: trading.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

### Add the DNS record

```bash
cloudflared tunnel route dns deepswing trading.yourdomain.com
```

This creates a CNAME in Cloudflare's DNS pointing to your tunnel. Cloudflare handles SSL automatically.

### Run as a systemd service

```bash
sudo cloudflared service install
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
```

Your dashboard is now live at `https://trading.yourdomain.com`.

### Tunnel architecture

```
Browser → Cloudflare Edge → Encrypted tunnel → Pi → DeepSwing :8000
```

No ports are open on your router. Traffic is encrypted in transit. Cloudflare acts as the reverse proxy.

---

## 6. Optional: Restrict Access

Since the dashboard is now public, add Cloudflare Access (free for 1 seat) to require login:

1. Cloudflare dashboard → Zero Trust → Access → Applications
2. Add Application → Self-hosted
3. Domain: `trading.yourdomain.com`
4. Policy: Email → `your@email.com`

You'll be prompted to log in via a one-time code sent to your email before seeing the dashboard.

---

## 7. Updating DeepSwing

```bash
cd ~/DeepSwing
git pull
venv/bin/pip install -r requirements.txt   # if dependencies changed
sudo systemctl restart deepswing
```

---

## Quick Reference

| Task | Command |
|---|---|
| Start | `sudo systemctl start deepswing` |
| Stop | `sudo systemctl stop deepswing` |
| Restart | `sudo systemctl restart deepswing` |
| Live logs | `journalctl -u deepswing -f` |
| Tunnel status | `sudo systemctl status cloudflared` |
| Dashboard | `https://trading.yourdomain.com` (or `http://deepswing.local:8000` on LAN) |
