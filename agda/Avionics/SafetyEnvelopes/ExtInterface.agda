module Avionics.SafetyEnvelopes.ExtInterface where

open import Data.Bool using (Bool)
open import Data.Float using (Float)
open import Data.List using (List; map)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Product using (_×_; _,_)

open import Avionics.Probability using (Dist; NormalDist; ND)
open import Avionics.Real renaming (fromFloat to ff; toFloat to tf)
open import Avionics.SafetyEnvelopes using (z-predictable'; sample-z-predictable)

open import ExtInterface.Data.Maybe using (just; nothing) renaming (Maybe to ExtMaybe)
open import ExtInterface.Data.Product as Ext using (⟨_,_⟩)

fromFloats-z-predictable : List (Float Ext.× Float) → Float → Float → ExtMaybe (Float Ext.× Bool)
fromFloats-z-predictable means×stds z x =
    let
      ndists = map (λ{⟨ mean , std ⟩ → ND (ff mean) (ff std) ?}) means×stds
      (m , b) = z-predictable' ndists (ff z) (ff x)
    in
      just ⟨ tf m , b ⟩
{-# COMPILE GHC fromFloats-z-predictable as zPredictable #-}

fromFloats-sample-z-predictable :
    List (Float Ext.× Float)
    → Float → Float → List Float → ExtMaybe (Float Ext.× Float Ext.× Bool)
fromFloats-sample-z-predictable means×stds zμ zσ xs =
    let
      ndists = map (λ{⟨ mean , std ⟩ → ND (ff mean) (ff std) ?}) means×stds
    in
      return (sample-z-predictable ndists (ff zμ) (ff zσ) (map ff xs))
  where
    return : Maybe (ℝ × ℝ × Bool) → ExtMaybe (Float Ext.× Float Ext.× Bool)
    return nothing = nothing
    return (just (m' , v' , b)) = just ⟨ tf m' , ⟨ tf v' , b ⟩ ⟩
{-# COMPILE GHC fromFloats-sample-z-predictable as sampleZPredictable #-}
