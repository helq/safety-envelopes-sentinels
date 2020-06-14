module Avionics.SafetyEnvelopes where

open import Data.Bool using (Bool; true; false; _∧_; _∨_)
open import Data.List using (List; []; _∷_; any; map; foldl; length)
open import Data.List.Relation.Unary.Any as Any using (Any)
open import Data.List.Relation.Unary.All as All using (All)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc)
open import Function using (_∘_)
open import Relation.Binary.PropositionalEquality
    using (_≡_; _≢_; refl; cong; subst; sym; trans)
open import Relation.Unary using (_∈_)
open import Relation.Nullary using (yes; no)
open import Relation.Nullary.Decidable using (fromWitnessFalse)

open import Avionics.Real
    using (ℝ; _+_; _-_; _*_; _÷_; _^_; _<ᵇ_; _≤ᵇ_; _≤_; _<_; _≢0; _≟_;
           1/_;
           0ℝ; 1ℝ; 2ℝ; _^2; √_; fromℕ;
           ⟨0,∞⟩; [0,∞⟩;
           <-transˡ; 2>0; ⟨0,∞⟩→0<; 0<→⟨0,∞⟩; >0→≢0; >0→≥0; q>0→√q>0)
open import Avionics.Product using (_×_; ⟨_,_⟩; proj₁; proj₂)
open import Avionics.Probability using (Dist; NormalDist; ND)

sum : List ℝ → ℝ
sum = foldl _+_ 0ℝ

inside : ℝ → ℝ → NormalDist → Bool
inside z x nd = ((μ - z * σ) <ᵇ x) ∧ (x <ᵇ (μ + z * σ))
  where open NormalDist nd using (μ; σ)

record Model : Set where
  field
    -- Angles of attack and airspeed available in the model
    SM : List (ℝ × ℝ)
    -- Map from angles of attack and airspeeds to Normal Distributions
    fM : List ((ℝ × ℝ) × (NormalDist × ℝ))
    -- Every pair of angle of attack and airspeed must be represented in the map fM
    .fMisMap₁ : All (λ ⟨α,v⟩ → Any (λ ⟨α,v⟩,ND → proj₁ ⟨α,v⟩,ND ≡ ⟨α,v⟩) fM ) SM
    .fMisMap₂ : All (λ ⟨α,v⟩,ND → Any (λ ⟨α,v⟩ → proj₁ ⟨α,v⟩,ND ≡ ⟨α,v⟩) SM ) fM
    .len>0 : length SM ≢ 0

z-predictable' : List NormalDist → ℝ → ℝ → ℝ × Bool
z-predictable' nds z x = ⟨ x , any (inside z x) nds ⟩

z-predictable : Model → ℝ → ℝ → ℝ × Bool
z-predictable M = z-predictable' (map (proj₁ ∘ proj₂) (Model.fM M))

--

sample-z-predictable : List NormalDist → ℝ → ℝ → List ℝ → Maybe (ℝ × ℝ × Bool)
sample-z-predictable nds zμ zσ [] = nothing
sample-z-predictable nds zμ zσ (_ ∷ []) = nothing
sample-z-predictable nds zμ zσ xs@(_ ∷ _ ∷ _) = just ⟨ mean , ⟨ var_est , any inside' nds ⟩ ⟩
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

--eqStall : StallClasses → StallClasses → Bool
--eqStall Stall     Stall = true
--eqStall NoStall   NoStall = true
--eqStall Uncertain Uncertain = true

-- TODO: `List (ℝ × ℝ × Dist ℝ)` should be replaced by something that ensures that
--       all ℝ (first) values are between 0 and 1, and their sum is 1
-- First ℝ is P[c], second is P[stall|c]
-- TODO: τ should be restricted to a number in the interval [0.5, 1)
-- TODO: Change `List (ℝ × ℝ × Dist ℝ)` for `Model`
classify' : List (ℝ × ℝ × Dist ℝ) → ℝ → ℝ → StallClasses
classify' pbs τ x = helper P[stall|X= x ]
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
    helper p with τ <ᵇ p | p <ᵇ (1ℝ - τ)
    ...         | true   | _        = Stall
    ...         | _      | true     = NoStall
    ...         | false  | false    = Uncertain
    --...         | true             | true     = ? -- This is never possible! This can be a theorem

classify : Model → ℝ → ℝ → StallClasses
classify M = classify' (map convert (Model.fM M))
  where 
    n = fromℕ (length (Model.fM M))
    postulate
      n≢0 : n ≢0

    1/n = (1/ n) {n≢0}

    convert : (ℝ × ℝ) × (NormalDist × ℝ) → ℝ × ℝ × Dist ℝ
    convert ⟨ _ , ⟨ nd , P[stall|c] ⟩ ⟩ = ⟨ 1/n , ⟨ P[stall|c] , dist ⟩ ⟩
      where open NormalDist nd using (dist)

not-uncertain : StallClasses → Bool
not-uncertain Uncertain = false
not-uncertain _ = true

τ-confident : Model → ℝ → ℝ → Bool
τ-confident M τ = not-uncertain ∘ classify M τ

safety-envelope : Model → ℝ → ℝ → ℝ → Bool
safety-envelope M z τ x = proj₂ (z-predictable M z x)
                          ∧ τ-confident M τ x
