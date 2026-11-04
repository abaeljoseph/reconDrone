#!/bin/sh
# At boot: stay on the home WiFi if it connects within 30 s, otherwise start the drone's own network (drone-ap,
# NetworkManager shared mode: the Pi is 10.42.0.1, the page http://10.42.0.1:8080). Other known networks
# (UTS-WiFi etc.) do not count: the phone usually cannot reach the Pi through them.
# Set HOME_WIFI to your network's connection name.
HOME_WIFI=your-home-wifi
for i in $(seq 30); do
  [ "$(nmcli -g GENERAL.CONNECTION dev show wlan0)" = "$HOME_WIFI" ] && exit 0
  sleep 1
done
nmcli con up drone-ap
