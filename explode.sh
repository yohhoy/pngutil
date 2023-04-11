#!/bin/bash
#
# Explode filesize of PNG format.
#
# (requires ImageMagick to encode original PNG)
#
if [ $# -ne 1 ]; then
  echo "$0 <input.png>"
  exit 1
fi
INPUT=$1

magick $INPUT -quality 96 -define png:exclude-chunk="cHRM,bKGD,tEXt,zTXt,tIME" PNG24:0-original.png
./pngutil.py 0-original.png --recompress=0                1-nocomp.png
./pngutil.py 0-original.png --recompress=0 --split-idat=1 2-nocomp-split.png
./pngutil.py 0-original.png --bloat                       3-bloat.png
./pngutil.py 0-original.png --bloat --split-idat=1        4-bloat-split.png
./pngutil.py 0-original.png --explode                     5-explode.png
./pngutil.py 0-original.png --explode --split-idat=1      6-explode-split.png

ls -l [0-9]*.png

