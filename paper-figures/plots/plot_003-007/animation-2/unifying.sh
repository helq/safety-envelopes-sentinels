#!/usr/bin/env zsh

#for i in {000..099}; do
#  convert 6ms/$i-*.png 20ms/$i-*.png all/$i-*.png +append 3-together/$i-plot.png
#done

mkdir 2-together-vertical
for i in {000..099}; do
  #convert 20ms/$i-*.png all/$i-*.png -append 2-together-horizontal/$i-plot.png
  convert 20ms/$i-*.png all/$i-*.png +append 2-together-vertical/$i-plot.png
done

#ffmpeg -r 6 -i 2-together-horizontal/%03d-plot.png tau-conf-horizontal.m4v
ffmpeg -r 6 -i 2-together-vertical/%03d-plot.png tau-conf-vertical.m4v
