# Scripts and figures generated in exploring Safety Envelopes

First of all, sorry! If you are reading this because you want to replicate some of the
figures in the paper, you are in good luck. The code works (assuming everything is up to
date to the current Python version (3.7+) and the libraries are installed correctly).
Unfortunately, the code is research code, and as such it might be incomplete,
undocumented, buggy and weird. It's incomplete as I left many TODOs before moving on. It
might steal your dog. It could even take your wallet from your pocket as it runs down the
street with your life's savings and dignity. There is no warranty, but thanks for passing
by to give it a chance.

The best place to start is taking a look under `plots/`. Once you have found a plot that
seems promising, check the associated `plotting-scripts/plot_XXX.py` Python file.

Scripts named `plot_00X.py` (notice the two leading zeroes) are old and should be ignored
at all cost, but if you insist using them, make sure to comment the line
`data.legacy_data(...)` under `__main__`. This line allows the use of a faulty
implementation of standard deviation. I decided to compute standard deviation on my own
and as per Murphy I got it wrong. The new code (evading/commenting the `legacy_data()`)
uses NumPy's procedure which is correct. If you don't comment that line, you'll get the
same plots that are under `plots/plot_003-007`. If you comment it, you'll get what those
plots should look like.

To run a script, navigate to this path (wherever this README is located) and type in the
console:

```bash
python plotting-scripts/plot_XXX.py
```

Happy hacking :)
