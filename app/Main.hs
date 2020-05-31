module Main where

import Pipes
import qualified Pipes.Prelude as P

import SafetyEnvelopes (check)

-- TODO: Change read parser for something that handles errors well
--   In fact, the usage of String is a bad practice. Use ByteString and read
--   data from file as binary data not as text data

main :: IO ()
main = do
  -- TODO: From input arguments get files to input and output

  -- TODO: Open files to read and write
  -- TODO: Replace P.stdinLn with P.withHandle
  -- TODO: Replace P.stdoutLn with custom function to output more than one value
  let multiplier = 4.0
      airspeed_i = 2
      c = check airspeed_i multiplier

  runEffect $ P.stdinLn >-> P.map (\x-> show . c $ read x) >-> P.stdoutLn
