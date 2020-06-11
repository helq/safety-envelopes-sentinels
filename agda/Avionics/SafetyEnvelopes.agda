{-# OPTIONS --allow-unsolved-metas #-}

module Avionics.SafetyEnvelopes where

open import Data.Bool using (Bool; true; false; _∧_; _∨_)
open import Data.Float using (Float)
open import Data.List using (List; []; _∷_; any; map; foldl; length)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc)
open import Relation.Binary.PropositionalEquality
    using (_≡_; refl; cong; subst; sym; trans)
open import Relation.Unary using (_∈_)

open import Avionics.Real
    using (ℝ; _+_; _-_; _*_; _÷_; _^_; _<ᵇ_; _≤ᵇ_; _≤_; _<_; _≢0;
           0ℝ; 1ℝ; 2ℝ; _^2; √_; fromℕ;
           ⟨0,∞⟩; [0,∞⟩;
           <-transˡ; 2>0; ⟨0,∞⟩→0<; 0<→⟨0,∞⟩; >0→≢0; >0→≥0; q>0→√q>0)
    renaming (fromFloat to ff; toFloat to tf)
open import Avionics.Product using (_×_; ⟨_,_⟩; proj₁; proj₂)
open import Avionics.Probability using (Dist; NormalDist; ND)

sum : List ℝ → ℝ
sum = foldl _+_ 0ℝ

-- TODO: Consider replacing _<_ for _<?_. _<?_ could allow better/easier proofs
inside : NormalDist → ℝ → ℝ → Bool
inside nd m x = ((μ - m * σ) <ᵇ x) ∧ (x <ᵇ (μ + m * σ))
  where open NormalDist nd using (μ; σ)

mean-cf : List NormalDist → ℝ → ℝ → ℝ × Bool
mean-cf nds m x = ⟨ x , any (λ nd → inside nd m x) nds ⟩

--mean-cf-theorem : ∀ (nds : List NormalDist) (x : ℝ)
--                → mean-cf nds (ff 4.0) x ≡ ⟨ x , true ⟩
--                -- TODO: it should say Exists d in nds, such that ...
--                → ∃[ μ ] (∃[ σ ] ((μ , σ) ∈ nds) ∧ P[ x ∈consistency] > 99.93)
--mean-cf-theorem = ?

sample-cf : List NormalDist → ℝ → ℝ → List ℝ → Maybe (ℝ × ℝ × Bool)
sample-cf nds mμ mσ [] = nothing
sample-cf nds mμ mσ (_ ∷ []) = nothing
sample-cf nds mμ mσ xs@(_ ∷ _ ∷ _) = just ⟨ mean , ⟨ var_est , any inside' nds ⟩ ⟩
  where
    n = fromℕ (length xs)
    -- Estimated mean from the data

    -- Proofs
    postulate
      2≤n : 2ℝ ≤ n  -- Because the list has at least two elements

    0<n : 0ℝ < n
    0<n = <-transˡ (⟨0,∞⟩→0< 2>0) 2≤n

    n≢0 : n ≢0
    n≢0 = >0→≢0 (0<→⟨0,∞⟩ 0<n)

    -- We can construct the rest of the proofs the same way. We are going to
    -- postulate them here now
    postulate
      n-1≢0 : (n - 1ℝ) ≢0

    mean = (sum xs ÷ n) {n≢0}
    -- Estimated Variance from the data (using the estimated mean)
    var_est = (sum (map (λ{x →(x - mean)^2}) xs) ÷ (n - 1ℝ)) {n-1≢0}

    inside' : NormalDist → Bool
    inside' nd = ((μ - mμ * σ) <ᵇ mean) ∧ (mean <ᵇ (μ + mμ * σ))
              ∧ (σ^2 - mσ * std[σ^2] <ᵇ var) ∧ (var <ᵇ σ^2 + mσ * std[σ^2])
      where open NormalDist nd using (μ; σ)
            -- Proofs
            2≥0 : 2ℝ ∈ [0,∞⟩
            2≥0 = >0→≥0 2>0

            n>0 : n ∈ ⟨0,∞⟩
            n>0 = 0<→⟨0,∞⟩ 0<n

            n≥0 : n ∈ [0,∞⟩
            n≥0 = >0→≥0 n>0

            √n≢0 : (√ n) ≢0
            √n≢0 = >0→≢0 (q>0→√q>0 n>0)

            -- Code
            σ^2 = σ ^2

            --Var[σ^2] = 2 * (σ^2)^2 / n
            std[σ^2] = (√ 2ℝ) {2≥0} * (σ^2 ÷ ((√ n) {n≥0})) {√n≢0}

            -- Notice that the estimated variance here is computed assuming `μ`
            -- it's the mean of the distribution. This is so that Cramer-Rao
            -- lower bound can be applied to it
            var = (sum (map (λ{x →(x - μ)^2}) xs) ÷ n) {n≢0}

nonneg-cf : ℝ → ℝ × Bool
nonneg-cf x = ⟨ x , 0ℝ ≤ᵇ x ⟩

fromFloatsMeanCF : List (Float × Float) → Float → Float → Maybe (Float × Bool)
fromFloatsMeanCF means×stds m x =
    let
      ndists = map (λ{⟨ mean , std ⟩ → ND (ff mean) (ff std) ?}) means×stds
      res = mean-cf ndists (ff m) (ff x)
    in
      just ⟨ tf (proj₁ res) , proj₂ res ⟩
{-# COMPILE GHC fromFloatsMeanCF as meanCF #-}

fromFloatsSampleCF : List (Float × Float) → Float → Float → List Float → Maybe (Float × Float × Bool)
fromFloatsSampleCF means×stds mμ mσ xs =
    let
      ndists = map (λ{⟨ mean , std ⟩ → ND (ff mean) (ff std) ?}) means×stds
    in
      return (sample-cf ndists (ff mμ) (ff mσ) (map ff xs))
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
    helper p with confindence <ᵇ p | p <ᵇ (1ℝ - confindence)
    ...         | true             | _        = Stall
    ...         | _                | true     = NoStall
    ...         | false            | false    = Uncertain
    --...         | true             | true     = ? -- This is never possible! This can be a theorem
