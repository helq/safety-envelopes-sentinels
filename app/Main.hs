module Main where

import Pipes
import qualified Pipes.Prelude as P
import Data.List (tails)
import Control.Monad (forM)
import System.IO.Error (catchIOError)
import System.Environment (getArgs)

import SafetyEnvelopes (checkMean, checkSample)

-- TODO: Change read parser for something that handles errors well
--   In fact, the usage of String is a bad practice. Use ByteString and read
--   data from file as binary data not as text data

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["mean"] -> main_mean
    ["sample"] -> main_sample
    _ -> putStrLn "You need to supply a parameter to check data. Either `mean` or `sample`"


main_mean :: IO ()
main_mean = do
  -- TODO: From input arguments get files to input and output
  -- TODO: Open files to read and write
  -- TODO: Replace P.stdinLn with P.withHandle
  -- TODO: Replace P.stdoutLn with custom function to output more than one value
  let multiplier = 4.0
      airspeed_i = 2
      c = checkMean airspeed_i multiplier

  runEffect $ P.stdinLn
            >-> P.map (\x-> show . c $ read x)
            >-> P.stdoutLn


main_sample :: IO ()
main_sample = do
  let multiplier = 4.0
      mul_var = 4.0
      airspeed_i = 2
      sample_n = 10

  -- TODO: Change this to use pipes!
  lines <- readLines
  let samples = takeWhile ((==sample_n) . length) $ (take sample_n) <$> tails (map read lines)
  forM samples $ \sample-> do
    --print $ sample
    print $ checkSample airspeed_i multiplier mul_var sample
  return ()

-- TODO: This is not lazy. This should go away once pipes are used
readLines :: IO [String]
readLines = do
  line <- catchIOError (Just <$> getLine) (\e -> return Nothing)
  case line of
    Just l  -> (l:) <$> readLines
    Nothing -> return []
