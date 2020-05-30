module Lib
    ( exampleRun
    ) where

import MAlonzo.Code.Avionics.SafetyEnvelopes

exampleRun = putStrLn . show $ consistencyenvelope 0 1 3 2
