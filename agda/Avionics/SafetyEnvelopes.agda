module Avionics.SafetyEnvelopes where

-- Compile using:
-- stack exec -- agda -c --ghc-dont-call-ghc --no-main Avionics/SafetyEnvelopes.agda

open import Data.Nat using (ℕ; zero; suc)
open import Data.Bool using (Bool; true; false; _∧_)
open import Data.Float using (Float)
--open import Data.List using (List; []; _∷_)

open import Avionics.Real using (ℝ; _+_; _-_; _*_; _<_)
                          renaming (fromFloat to ff; toFloat to tf)
open import Avionics.Product using (_×_; ⟨_,_⟩; proj₁; proj₂)
open import Avionics.Probability using (Dist; NormalDist)

-- TODO: Change `NormalDist` for `List NormalDist`. A good implementation
--       probably requires Data.List.Any and _<?_ to make proofs better/easier
cf-signal : NormalDist → ℝ → ℝ → ℝ × Bool
cf-signal nd m x = ⟨ x , inside x ⟩
  where
    open NormalDist nd -- makes μ, σ and dist accessible
    open Dist dist -- makes pdf and cdf accessible

    -- TODO: Consider replacing _<_ for _<?_. _<?_ could allow better/easier proofs
    inside : ℝ → Bool
    inside x = ((μ - m * σ) < x) ∧ (x < (μ + m * σ))

fromFloatsCF : Float → Float → Float → Float → Float × Bool
fromFloatsCF mean std m x =
    let
      ndist = record { μ = ff mean; σ = ff std }
      res = cf-signal ndist (ff m) (ff x)
    in
      ⟨ tf (proj₁ res) , proj₂ res ⟩

{-# COMPILE GHC fromFloatsCF as consistencyenvelope #-}
