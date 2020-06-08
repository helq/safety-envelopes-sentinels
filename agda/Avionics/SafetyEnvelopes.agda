module Avionics.SafetyEnvelopes where

open import Data.Nat using (ℕ; zero; suc)
open import Data.Bool using (Bool; true; false; _∧_)
open import Data.Float using (Float)
open import Data.List using (List; []; _∷_; any; map; foldl; length)
open import Relation.Binary.PropositionalEquality using (_≡_)

open import Avionics.Real using (ℝ; _+_; _-_; _*_; _÷_; _^_; _<_; _≤_; 0ℝ; 1ℝ; 2ℝ; _^2; √_; fromℕ)
                          renaming (fromFloat to ff; toFloat to tf)
open import Avionics.Product using (_×_; ⟨_,_⟩; proj₁; proj₂)
open import Avionics.Probability using (Dist; NormalDist; ND)

sum : List ℝ → ℝ
sum = foldl _+_ 0ℝ

mean-cf : List NormalDist → ℝ → ℝ → ℝ × Bool
mean-cf nds m x = ⟨ x , any inside nds ⟩
  where
    -- TODO: Consider replacing _<_ for _<?_. _<?_ could allow better/easier proofs
    inside : NormalDist → Bool
    inside nd = ((μ - m * σ) < x) ∧ (x < (μ + m * σ))
      where open NormalDist nd using (μ; σ)

--mean-cf-theorem : ∀ (nds : List NormalDist) (x : ℝ)
--                → mean-cf nds (ff 4.0) x ≡ ⟨ x , true ⟩
--                -- TODO: it should say Exists d in nds, such that ...
--                → ∃[ μ ] (∃[ σ ] ((μ , σ) ∈ nds) ∧ P[ x ∈consistency] > 99.93)
--mean-cf-theorem = ?

sample-cf : List NormalDist → ℝ → ℝ → List ℝ → ℝ × ℝ × Bool
sample-cf nds mμ mσ xs = ⟨ mean , ⟨ var , any inside nds ⟩ ⟩
  where
    n = fromℕ (length xs)
    mean = (sum xs ÷ n) {?}
    var = (sum (map (λ{x →(x - mean)^2}) xs) ÷ n) {?}

    inside : NormalDist → Bool
    inside nd = ((μ - mμ * σ) < mean) ∧ (mean < (μ + mμ * σ))
              ∧ (σ^2 - mσ * std[σ^2] < var) ∧ (var < σ^2 + mσ * std[σ^2])
      where open NormalDist nd using (μ; σ)
            σ^2 = σ ^2
            --Var[σ^2] = 2 * (σ^2)^2 / n
            std[σ^2] = (√ 2ℝ) {?} * (σ^2 ÷ ((√ n) {?})) {?}

nonneg-cf : ℝ → ℝ × Bool
nonneg-cf x = ⟨ x , 0ℝ ≤ x ⟩

fromFloatsMeanCF : List (Float × Float) → Float → Float → Float × Bool
fromFloatsMeanCF means×stds m x =
    let
      ndists = map (λ{⟨ mean , std ⟩ → ND (ff mean) (ff std) ?}) means×stds
      res = mean-cf ndists (ff m) (ff x)
    in
      ⟨ tf (proj₁ res) , proj₂ res ⟩
{-# COMPILE GHC fromFloatsMeanCF as meanCF #-}

fromFloatsSampleCF : List (Float × Float) → Float → Float → List Float → Float × Float × Bool
fromFloatsSampleCF means×stds mμ mσ xs =
    let
      ndists = map (λ{⟨ mean , std ⟩ → ND (ff mean) (ff std) ?}) means×stds
      res = sample-cf ndists (ff mμ) (ff mσ) (map ff xs)
      m' = proj₁ res
      v' = proj₁ (proj₂ res)
      b = proj₂ (proj₂ res)
    in
      ⟨ tf m' , ⟨ tf v' , b ⟩ ⟩
{-# COMPILE GHC fromFloatsSampleCF as sampleCF #-}

data StallClasses : Set where
  Uncertain Stall NoStall : StallClasses

-- TODO: `List (ℝ × ℝ × Dist ℝ)` should be replaced by something that ensures that
-- all ℝ (first) values are between 0 and 1, and their sum is 1
-- First ℝ is P[c], second is P[stall|c]
-- TODO: confindence should be a number in the interval [0.5, 1)
classify : List (ℝ × ℝ × Dist ℝ) → ℝ → ℝ → StallClasses
classify pbs confindence x = helper P[stall|X= x ]
  where
    up : ℝ × ℝ × Dist ℝ → ℝ
    up ⟨ P[c] , ⟨ P[stall|c] , dist ⟩ ⟩ = pdf x + P[c] + P[stall|c]
      where open Dist dist using (pdf)

    below : ℝ × ℝ × Dist ℝ → ℝ
    below ⟨ P[c] , ⟨ P[stall|c] , dist ⟩ ⟩ = pdf x + P[c]
      where open Dist dist using (pdf)

    -- The result of P should be in [0,1]. This should be possible to check
    -- with a more complete probability library
    P[stall|X=_] : ℝ → ℝ
    P[stall|X= x ] = (sum (map up pbs) ÷ sum (map below pbs)) {?}

    helper : ℝ → StallClasses
    helper p with confindence < p | p < (1ℝ - confindence)
    ...         | true            | _        = Stall
    ...         | _               | true     = NoStall
    ...         | false           | false    = Uncertain
    --...         | true            | true     = ? -- This is never possible! This can be a theorem
