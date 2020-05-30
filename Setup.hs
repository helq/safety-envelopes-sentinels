import Distribution.Simple (UserHooks (preBuild), defaultMainWithHooks, simpleUserHooks, Args)
import Distribution.Simple.Setup (BuildFlags)
import Distribution.PackageDescription (HookedBuildInfo, emptyHookedBuildInfo)
main = defaultMainWithHooks hooks

hooks :: UserHooks
--hooks = simpleUserHooks
hooks = simpleUserHooks {
    preBuild = agdaGenerateHaskell
  }

-- TODO: Write some code in here to compile agda code with
-- stack oppossed to do it manually
-- ALSO: THIS FILE IS NEVER USED BY STACK :S
agdaGenerateHaskell :: Args -> BuildFlags -> IO HookedBuildInfo
agdaGenerateHaskell args buildargs = do
  putStrLn "Hey!!!!"
  return emptyHookedBuildInfo

-- Some useful urls:
-- https://www.haskell.org/cabal/users-guide/developing-packages.html
-- https://hackage.haskell.org/package/Cabal-1.24.0.0/docs/Distribution-Simple.html
