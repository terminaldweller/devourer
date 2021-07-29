#!/usr/bin/env sh
set -e
set -x

# sniff --src https://github.com/coinpride/CryptoList --url | ./main.py > out2.txt
./main.py --source https://github.com/coinpride/CryptoList > out.txt
# cat out2.txt| pico2wave --wave=out2.wav
