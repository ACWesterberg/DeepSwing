#!/bin/bash
set -e

cd ~/Documents/DeepSwing

git fetch origin deploy
git reset --hard origin/deploy

# Install shared data library (editable, lives at ~/FinanceData on the Pi)
if [ -d ~/FinanceData ]; then
    venv/bin/pip install -e ~/FinanceData -q
else
    echo "WARNING: ~/FinanceData not found — financedata not installed"
fi

venv/bin/pip install -r requirements.txt -q

sudo systemctl restart deepswing
sleep 5
sudo systemctl is-active deepswing
