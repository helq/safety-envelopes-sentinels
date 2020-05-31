module Avionics.SafetyEnvelopes where

open import Data.Nat using (ℕ; zero; suc)
open import Data.Bool using (Bool; true; false; _∧_)
open import Data.Float using (Float)
open import Data.List using (List; []; _∷_; any; map)

open import Avionics.Real using (ℝ; _+_; _-_; _*_; _<_; _≤_; 0ℝ)
                          renaming (fromFloat to ff; toFloat to tf)
open import Avionics.Product using (_×_; ⟨_,_⟩; proj₁; proj₂)
open import Avionics.Probability using (Dist; NormalDist; ND)

-- TODO: Change `NormalDist` for `List NormalDist`. A good implementation
--       probably requires Data.List.Any and _<?_ to make proofs better/easier
mean-cf : List NormalDist → ℝ → ℝ → ℝ × Bool
mean-cf nds m x = ⟨ x , any (inside x) nds ⟩
  where
    -- TODO: Consider replacing _<_ for _<?_. _<?_ could allow better/easier proofs
    inside : ℝ → NormalDist → Bool
    inside x nd = ((μ - m * σ) < x) ∧ (x < (μ + m * σ))
      where open NormalDist nd using (μ; σ)

nonneg-cf : ℝ → ℝ × Bool
nonneg-cf x = ⟨ x , 0ℝ ≤ x ⟩

fromFloatsCF : List (Float × Float) → Float → Float → Float × Bool
fromFloatsCF means×stds m x =
    let
      ndists = map (λ{⟨ mean , std ⟩ → ND (ff mean) (ff std)}) means×stds
      res = mean-cf ndists (ff m) (ff x)
    in
      ⟨ tf (proj₁ res) , proj₂ res ⟩

{-# COMPILE GHC fromFloatsCF as consistencyenvelope #-}
