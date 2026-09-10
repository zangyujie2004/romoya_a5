#!/usr/bin/env bash
set -euo pipefail

readonly CAN_BITRATE_CODE="s8"
readonly -a CAN_IDS=(1 3 5 7)

for id in "${CAN_IDS[@]}"; do
    sudo ip link set "can${id}" down 2>/dev/null || true
done
sudo pkill slcand 2>/dev/null || true

sleep 0.5

for id in "${CAN_IDS[@]}"; do
    device="/dev/arxcan${id}"
    interface="can${id}"

    if [[ ! -e "${device}" ]]; then
        echo "Missing ${device}; check USB connection and udev rules." >&2
        exit 1
    fi

    sudo slcand -o -f "-${CAN_BITRATE_CODE}" "${device}" "${interface}"
    sudo ip link set "${interface}" up
done

sleep 0.2

for id in "${CAN_IDS[@]}"; do
    interface="can${id}"
    state=$(<"/sys/class/net/${interface}/operstate")
    if [[ "${state}" != "up" ]]; then
        echo "${interface} failed to reach UP state (state=${state})." >&2
        exit 1
    fi
done

ip -brief link show type can
