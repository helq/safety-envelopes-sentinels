#!/usr/bin/env zsh

for i in {000..099}; do
#  convert 6ms/$i-*.png 20ms/$i-*.png all/$i-*.png +append 3-together/$i-plot.png
  convert 20ms/$i-*.png all/$i-*.png -append 2-together-vertical/$i-plot.png
done

ffmpeg -r 12 -i 2-together-vertical/%03d-plot.png z-pred-vertical.m4v
ffmpeg -i "z-pred-vertical.m4v" -c:v libx264 -preset slow -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac "z-pred-vertical.mp4"
