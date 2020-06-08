# sentinels

To check and compile Agda code into Haskell code:

~~~sh
rm -r src/MAlonzo
cd agda
stack exec -- agda -c --ghc-dont-call-ghc --no-main --compile-dir=../src Avionics/SafetyEnvelopes.agda
cd ..
~~~

To compile Haskell code:

~~~sh
stack build
~~~

To run Haskell code:

~~~sh
stack exec -- sentinels-exe
~~~
