module Avionics.SafetyEnvelopes where

open import Data.Nat using (ℕ; zero; suc)
open import Data.Bool using (Bool; true; false; _∧_)
open import Data.Float using (Float)
open import Data.List using (List; []; _∷_; any; map; foldl)

open import Avionics.Real using (ℝ; _+_; _-_; _*_; _÷_; _<_; _≤_; 0ℝ; 1ℝ)
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

data StallClasses : Set where
  Uncertain Stall NoStall : StallClasses

sum : List ℝ → ℝ
sum = foldl _+_ 0ℝ

-- TODO: `List (ℝ × ℝ × NormalDist)` should be replaced by something that ensures that
-- all ℝ (first) values are between 0 and 1, and their sum is 1
-- First ℝ is P[c], second is P[stall|c]
-- TODO: confindence should be a number in the interval [0.5, 1)
classify : List (ℝ × ℝ × NormalDist) → ℝ → ℝ → StallClasses
classify pbs confindence x = helper P[stall|X= x ]
  where
    up : ℝ × ℝ × NormalDist → ℝ
    up ⟨ P[c] , ⟨ P[stall|c] , nd ⟩ ⟩ = pdf x + P[c] + P[stall|c]
      where open NormalDist nd using (dist)
            open Dist dist using (pdf)

    below : ℝ × ℝ × NormalDist → ℝ
    below ⟨ P[c] , ⟨ P[stall|c] , nd ⟩ ⟩ = pdf x + P[c]
      where open NormalDist nd using (dist)
            open Dist dist using (pdf)

    -- The result of P should be in [0,1]. This should be possible to check
    -- with a more complete probability library
    P[stall|X=_] : ℝ → ℝ
    P[stall|X= x ] = sum (map up pbs) ÷ sum (map below pbs)

    helper : ℝ → StallClasses
    helper p with confindence < p | p < (1ℝ - confindence)
    ...         | true            | _        = Stall
    ...         | _               | true     = NoStall
    ...         | false           | false    = Uncertain
    --...         | true            | true     = ? -- This is never possible! This can be a theorem
