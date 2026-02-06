#!/usr/bin/bash

# Continuous 15 minute recording from Linux audio "card 3"
# (USB audio input from lav mic in this case)  

# $ arecord -l
# **** List of CAPTURE Hardware Devices ****
# card 2: i2smic [i2smic], device 0: bcm2835-i2s-dir-hifi dir-hifi-0 [bcm2835-i2s-dir-hifi dir-hifi-0]
#  Subdevices: 1/1
#  Subdevice #0: subdevice #0
# card 3: Creation [Cable Creation], device 0: USB Audio [USB Audio]
#  Subdevices: 1/1
#  Subdevice #0: subdevice #0

arecord -D plughw:3,0 -c 1 -r 24000 -f S16_LE -t raw - | \
ffmpeg \
  -f s16le \
  -ar 24000 \
  -ac 1 \
  -i - \
  -filter:a "highpass=f=20" \
  -c:a libmp3lame -b:a 64k \
  -write_xing 1 \
  -f segment \
  -segment_time 900 \
  -segment_atclocktime 1 \
  -reset_timestamps 1 \
  -strftime 1 \
  "ch4_%Y%m%d_%H%M%S.mp3"
  
