module Main where

import Pipes
import qualified Pipes.Prelude as P
import Data.List (tails)
import Control.Monad (forM)
import System.IO.Error (catchIOError)

import SafetyEnvelopes (checkMean, checkSample)

-- TODO: Change read parser for something that handles errors well
--   In fact, the usage of String is a bad practice. Use ByteString and read
--   data from file as binary data not as text data

--main :: IO ()
main = do
  -- TODO: From input arguments get files to input and output

  -- TODO: Open files to read and write
  -- TODO: Replace P.stdinLn with P.withHandle
  -- TODO: Replace P.stdoutLn with custom function to output more than one value
  let multiplier = 4.0
      mul_var = 2.0
      airspeed_i = 2
      sample_n = 10
  --    c = checkMean airspeed_i multiplier
  --
  --runEffect $ P.stdinLn
  --          >-> P.map (\x-> show . c $ read x)
  --          >-> P.stdoutLn

  -- TODO: Change this to use pipes!
  lines <- readLines
  let samples = takeWhile ((==10) . length) $ (take 10) <$> tails (map read lines)
  forM samples $ \sample-> do
    --print $ sample
    print $ checkSample airspeed_i multiplier mul_var sample

-- TODO: This is not lazy. This should go away once pipes are used
readLines :: IO [String]
readLines = do
  line <- catchIOError (Just <$> getLine) (\e -> return Nothing)
  case line of
    Just l  -> (l:) <$> readLines
    Nothing -> return []
