# Safety Envelopes Sentinel

This project contains the Agda and Haskell code mentioned in the [paper](http://wcl.cs.rpi.edu/bib/Year/2020.complete.html#cruz-dddas-2020).

## To compile

To compile the code you need [stack](https://docs.haskellstack.org/en/stable/README/).

### Generating code from Agda

To typecheck and compile the Agda code into Haskell code run the following in the command line:

```sh
rm -r src/MAlonzo
cd agda
stack exec -- agda -c --ghc-dont-call-ghc --no-main --compile-dir=../src Avionics/SafetyEnvelopes/ExtInterface.agda
cd ..
```

If the previous lines run without errors it means that the Agda code typedchecked.

A new folder should have appeared `src/Alonzo`. It contains the code generated from Agda.

### Compiling Haskell code

The generated code needs to be compiled to be run:

```sh
stack build
```

## To run the code

```sh
stack exec -- sentinels-exe single
```

Once the sentinel is running it will read from stdin a list of floating point numbers and
it determine if the value is z-predictable, i.e., falls within certain confidence region
where the models make sense.

## To produce LaTeX code

- Copy file to generate LaTex from `.agda` to `.lagda`. Rename `.agda` file into
    `.agda.old` to evade name collisions.
- Surround code with `\begin{code} \end{code}`

```sh
cd agda
stack exec -- agda --latex Avionics/MyFileToGenerateLaTeX.lagda
cd ..
```

- Delete `.lagda` file and restore `.agda` file from `.agda.old`

---

_This article is a [stub](https://en.wikipedia.org/wiki/Wikipedia:Stub). You can help us
by [expanding it](https://en.wikipedia.org/wiki/Open-source_software)._
