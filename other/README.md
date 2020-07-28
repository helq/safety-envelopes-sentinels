To recreate the plots run:

~~~bash
cd other
~~~

- Generate data

    ~~~bash
    cd other
    python generate-simulation-data.py .1 > data/generated-10percent-noise.txt
    python generate-simulation-data.py .2 > data/generated-20percent-noise.txt
    python generate-simulation-data.py .5 > data/generated-50percent-noise.txt
    python generate-simulation-data.py .8 > data/generated-80percent-noise.txt
    python generate-simulation-data.py 1 > data/generated-100percent-noise.txt
    python generate-simulation-data.py 2 > data/generated-200percent-noise.txt
    python generate-simulation-data.py 5 > data/generated-500percent-noise.txt
    ~~~

- Process data

    ~~~
    stack exec -- sentinels-exe simple < data/generated-10percent-noise.txt > data/simple-cf-10percent-noise.txt
    stack exec -- sentinels-exe simple < data/generated-20percent-noise.txt > data/simple-cf-20percent-noise.txt
    stack exec -- sentinels-exe simple < data/generated-50percent-noise.txt > data/simple-cf-50percent-noise.txt
    stack exec -- sentinels-exe simple < data/generated-80percent-noise.txt > data/simple-cf-80percent-noise.txt
    stack exec -- sentinels-exe simple < data/generated-100percent-noise.txt > data/simple-cf-100percent-noise.txt
    stack exec -- sentinels-exe simple < data/generated-200percent-noise.txt > data/simple-cf-200percent-noise.txt
    stack exec -- sentinels-exe simple < data/generated-500percent-noise.txt > data/simple-cf-500percent-noise.txt

    stack exec -- sentinels-exe sample < data/generated-10percent-noise.txt > data/sample-cf-10percent-noise.txt
    stack exec -- sentinels-exe sample < data/generated-20percent-noise.txt > data/sample-cf-20percent-noise.txt
    stack exec -- sentinels-exe sample < data/generated-50percent-noise.txt > data/sample-cf-50percent-noise.txt
    stack exec -- sentinels-exe sample < data/generated-80percent-noise.txt > data/sample-cf-80percent-noise.txt
    stack exec -- sentinels-exe sample < data/generated-100percent-noise.txt > data/sample-cf-100percent-noise.txt
    stack exec -- sentinels-exe sample < data/generated-200percent-noise.txt > data/sample-cf-200percent-noise.txt
    stack exec -- sentinels-exe sample < data/generated-500percent-noise.txt > data/sample-cf-500percent-noise.txt
    ~~~

- Create plots

    ~~~
    python create-plot.py data/generated-10percent-noise.txt  data/simple-cf-10percent-noise.txt  data/sample-cf-10percent-noise.txt  plots/consistency-ms=4-10-percent-noise
    python create-plot.py data/generated-20percent-noise.txt  data/simple-cf-20percent-noise.txt  data/sample-cf-20percent-noise.txt  plots/consistency-ms=4-20-percent-noise
    python create-plot.py data/generated-50percent-noise.txt  data/simple-cf-50percent-noise.txt  data/sample-cf-50percent-noise.txt  plots/consistency-ms=4-50-percent-noise
    python create-plot.py data/generated-80percent-noise.txt  data/simple-cf-80percent-noise.txt  data/sample-cf-80percent-noise.txt  plots/consistency-ms=4-80-percent-noise
    python create-plot.py data/generated-100percent-noise.txt data/simple-cf-100percent-noise.txt data/sample-cf-100percent-noise.txt plots/consistency-ms=4-100-percent-noise
    python create-plot.py data/generated-200percent-noise.txt data/simple-cf-200percent-noise.txt data/sample-cf-200percent-noise.txt plots/consistency-ms=4-200-percent-noise
    python create-plot.py data/generated-500percent-noise.txt data/simple-cf-500percent-noise.txt data/sample-cf-500percent-noise.txt plots/consistency-ms=4-500-percent-noise
    ~~~
