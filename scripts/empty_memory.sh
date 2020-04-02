#!/usr/bin/env bash#

sync; echo 1 > /proc/sys/vm/drop_caches
swapoff -a; swapon -a
