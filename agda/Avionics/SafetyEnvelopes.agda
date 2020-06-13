module Avionics.SafetyEnvelopes where

open import Data.Bool using (Bool; true; false; _∧_; _∨_)
open import Data.List using (List; []; _∷_; any; map; foldl; length)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc)
open import Relation.Binary.PropositionalEquality
    using (_≡_; refl; cong; subst; sym; trans)
open import Relation.Unary using (_∈_)
open import Relation.Nullary using (yes; no)
open import Relation.Nullary.Decidable using (fromWitnessFalse)

open import Avionics.Real
    using (ℝ; _+_; _-_; _*_; _÷_; _^_; _<ᵇ_; _≤ᵇ_; _≤_; _<_; _≢0; _≟_;
           0ℝ; 1ℝ; 2ℝ; _^2; √_; fromℕ;
           ⟨0,∞⟩; [0,∞⟩;
           <-transˡ; 2>0; ⟨0,∞⟩→0<; 0<→⟨0,∞⟩; >0→≢0; >0→≥0; q>0→√q>0)
open import Avionics.Product using (_×_; ⟨_,_⟩; proj₁; proj₂)
open import Avionics.Probability using (Dist; NormalDist; ND)

sum : List ℝ → ℝ
sum = foldl _+_ 0ℝ

-- TODO: Consider replacing _<_ for _<?_. _<?_ could allow better/easier proofs
inside : ℝ → ℝ → NormalDist → Bool
inside z x nd = ((μ - z * σ) <ᵇ x) ∧ (x <ᵇ (μ + z * σ))
  where open NormalDist nd using (μ; σ)

z-predictable : List NormalDist → ℝ → ℝ → ℝ × Bool
z-predictable nds z x = ⟨ x , any (inside z x) nds ⟩

sample-cf : List NormalDist → ℝ → ℝ → List ℝ → Maybe (ℝ × ℝ × Bool)
sample-cf nds zμ zσ [] = nothing
sample-cf nds zμ zσ (_ ∷ []) = nothing
sample-cf nds zμ zσ xs@(_ ∷ _ ∷ _) = just ⟨ mean , ⟨ var_est , any inside' nds ⟩ ⟩
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
    inside' nd = ((μ - zμ * σ) <ᵇ mean) ∧ (mean <ᵇ (μ + zμ * σ))
              ∧ (σ^2 - zσ * std[σ^2] <ᵇ var) ∧ (var <ᵇ σ^2 + zσ * std[σ^2])
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
    P[stall|X= x ] with sum (map below pbs) ≟ 0ℝ
    ... | yes _  = 0ℝ
    ... | no x≢0 = (sum (map up pbs) ÷ sum (map below pbs)) {fromWitnessFalse x≢0}

    helper : ℝ → StallClasses
    helper p with confindence <ᵇ p | p <ᵇ (1ℝ - confindence)
    ...         | true             | _        = Stall
    ...         | _                | true     = NoStall
    ...         | false            | false    = Uncertain
    --...         | true             | true     = ? -- This is never possible! This can be a theorem
