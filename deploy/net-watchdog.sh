#!/bin/bash
# Self-heal the Pi's network: the Wi-Fi driver can drop the association and
# never rejoin, which kills SSH and both Cloudflare tunnels while the box
# otherwise runs fine. Bounce the interface when the gateway is unreachable;
# reboot after 3 consecutive failed recoveries (portfolio state is DB-backed,
# so a reboot is safe).
set -u

STATE=/run/net-watchdog.fails
MAX_FAILS=3

iface=$(ip route show default | awk '{print $5; exit}')
gw=$(ip route show default | awk '{print $3; exit}')
if [ -z "$iface" ]; then
    iface=$(ls /sys/class/net | grep -E '^(wlan|eth)' | head -1)
fi

online() {
    if [ -n "$gw" ] && ping -c 2 -W 3 "$gw" > /dev/null 2>&1; then
        return 0
    fi
    ping -c 2 -W 3 1.1.1.1 > /dev/null 2>&1
}

if online; then
    rm -f "$STATE"
    exit 0
fi

fails=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$fails" > "$STATE"

if [ "$fails" -ge "$MAX_FAILS" ]; then
    logger -t net-watchdog "network still down after $MAX_FAILS recovery attempts — rebooting"
    systemctl reboot
    exit 0
fi

logger -t net-watchdog "network down (attempt $fails/$MAX_FAILS) — bouncing $iface"
ip link set "$iface" down
sleep 5
ip link set "$iface" up

if systemctl is-enabled --quiet NetworkManager 2>/dev/null; then
    systemctl restart NetworkManager
elif systemctl is-enabled --quiet dhcpcd 2>/dev/null; then
    systemctl restart dhcpcd
fi
