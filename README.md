# sentinels

To check and compile Agda code into Haskell code:

```sh
rm -r src/MAlonzo
cd agda
stack exec -- agda -c --ghc-dont-call-ghc --no-main --compile-dir=../src Avionics/SafetyEnvelopes/ExtInterface.agda
cd ..
```

To compile Haskell code:

~~~sh
stack build
~~~

To run Haskell code:

```sh
stack exec -- sentinels-exe
```

To produce LaTeX text:

- Copy file to generate LaTex from `.agda` to `.lagda`. Rename `.agda` file into
    `.agda.old` to evade name collisions.
- Surround code with `\begin{code} \end{code}`

```sh
cd agda
stack exec -- agda --latex Avionics/MyFileToGenerateLaTeX.lagda
cd ..
```

- Delete `.lagda` file and restore `.agda` file from `.agda.old`

Batch renaming:

```bash
find . -name '*.agda' -exec bash -c '
  file="$0"
  cp "$file" "${file%.agda}.lagda"
  mv "$file" "${file}.old"
' {} \;
```

... Add header and footer

```bash
# Generating tex files
find . -name '*.lagda' -exec echo stack exec -- agda --latex {} \;

# Restoring agda files
find . -name '*.agda.old' -exec bash -c '
  file="$0"
  mv "$file" "${file%.old}"
  rm "${file%.agda.old}.lagda"
' {} \;
```
