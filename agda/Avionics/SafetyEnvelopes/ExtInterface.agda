module Avionics.SafetyEnvelopes.ExtInterface where

open import Data.Bool using (Bool)
open import Data.Float using (Float)
open import Data.List using (List; map)
open import Data.Maybe using (Maybe; just; nothing)

open import Avionics.Product using (_×_; ⟨_,_⟩; proj₁; proj₂)
open import Avionics.Probability using (Dist; NormalDist; ND)
open import Avionics.Real renaming (fromFloat to ff; toFloat to tf)
open import Avionics.SafetyEnvelopes using (z-predictable; sample-cf)

-- TODO: Change names
fromFloatsMeanCF : List (Float × Float) → Float → Float → Maybe (Float × Bool)
fromFloatsMeanCF means×stds z x =
    let
      ndists = map (λ{⟨ mean , std ⟩ → ND (ff mean) (ff std) ?}) means×stds
      res = z-predictable ndists (ff z) (ff x)
    in
      just ⟨ tf (proj₁ res) , proj₂ res ⟩
{-# COMPILE GHC fromFloatsMeanCF as meanCF #-}

fromFloatsSampleCF : List (Float × Float) → Float → Float → List Float → Maybe (Float × Float × Bool)
fromFloatsSampleCF means×stds zμ zσ xs =
    let
      ndists = map (λ{⟨ mean , std ⟩ → ND (ff mean) (ff std) ?}) means×stds
    in
      return (sample-cf ndists (ff zμ) (ff zσ) (map ff xs))
  where
    return : Maybe (ℝ × ℝ × Bool) → Maybe (Float × Float × Bool)
    return nothing = nothing
    return (just res) =
      let
        m' = proj₁ res
        v' = proj₁ (proj₂ res)
        b = proj₂ (proj₂ res)
      in
        just ⟨ tf m' , ⟨ tf v' , b ⟩ ⟩
{-# COMPILE GHC fromFloatsSampleCF as sampleCF #-}
