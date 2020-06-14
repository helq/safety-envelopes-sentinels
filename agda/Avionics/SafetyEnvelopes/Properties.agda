module Avionics.SafetyEnvelopes.Properties where

open import Data.Bool using (true)
open import Data.List as List using (List; []; _∷_)
open import Data.List.Relation.Unary.Any as Any using (Any; here; there; satisfied)
open import Data.Product using (∃-syntax)
open import Function using (_∘_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; cong)
open import Relation.Nullary.Decidable using (toWitness)
open import Relation.Unary using (_∈_)

open import Avionics.Bool using (≡→T; T∧→×)
open import Avionics.List using (any-val)
open import Avionics.Real
    using (ℝ; _+_; _-_; _*_; _÷_; _^_; _<ᵇ_; _≤ᵇ_; _≤_; _<_; _<?_; _≢0;
           0ℝ; 1ℝ; 2ℝ; _^2; √_; fromℕ;
           ⟨0,∞⟩; [0,∞⟩;
           <-transˡ; 2>0; ⟨0,∞⟩→0<; 0<→⟨0,∞⟩; >0→≢0; >0→≥0; q>0→√q>0)
open import Avionics.Probability using (NormalDist; Dist)
open import Avionics.Product using (_×_; ⟨_,_⟩; proj₁; proj₂; map)
open import Avionics.SafetyEnvelopes
    using (inside; z-predictable'; classify; StallClasses; safety-envelope;
           z-predictable; Model; τ-confident)

open NormalDist using (σ; μ)

--<ᵇ→< : ∀ {x y} → T (x <ᵇ y) → x < y
--<ᵇ→< = toWitness

-- `pi` is the prediction interval for the z score, i.e.,
-- pi(N (μ, σ), z) = [μ − zσ, μ + zσ]
pi : NormalDist → ℝ → ℝ → Set
pi nd z x = (μ nd) - z * (σ nd) < x  ×  x < (μ nd) + z * (σ nd)

-- Proof that implementation follows from definition (Definition 2)
--
-- In words, the Property 1 says that:
--    The energy signal x is z-predictable iff there exist ⟨α, v⟩ s.t.
--    M(⟨α, v⟩)1 = di and x ∈ pi(di , z).
--
-- Notice that `Any (λ nd → x ∈ pi nd z) nds` translates to:
-- there exists nd such that `nd ∈ nds` and `x ∈ pi(nd, z)`
prop1 : ∀ (nds z x)
      → z-predictable' nds z x ≡ ⟨ x , true ⟩
      → Any (λ nd → x ∈ pi nd z) nds
prop1 nds z x res≡x,true = Any-x∈pi
  where
    res≡true = cong proj₂ res≡x,true

    -- the first `toWitness` takes a result `(μ nd - z * σ nd) <ᵇ x` (a
    -- boolean) and produces a proof of the type `(μ nd) - z * (σ nd) < x`
    -- assuming we have provided an operator `<?`
    toWitness' = λ nd → map (toWitness {Q = (μ nd - z * σ nd) <? x})
                            (toWitness {Q = x <? (μ nd + z * σ nd)})

    -- We find the value for which `inside z x` becomes true in the list `nds`
    Any-bool = any-val (inside z x) nds res≡true
    -- Converting the boolean proof into a proof at the type level
    Any-x∈pi = Any.map (λ {nd} → toWitness' nd ∘ T∧→×) Any-bool

-- From the prove above we can obtain the value `nd` and its prove `x ∈ pi nd z`
-- Note: An element of ∃[ nd ] (x ∈ pi nd z) is a tuple of the form ⟨ nd , proof ⟩
prop1' : ∀ (nds z x)
       → z-predictable' nds z x ≡ ⟨ x , true ⟩
       → ∃[ nd ] (x ∈ pi nd z)
prop1' nds z x res≡x,true = satisfied (prop1 nds z x res≡x,true)

extractDists : Model → List NormalDist
extractDists M = List.map (proj₁ ∘ proj₂) (Model.fM M)

prop1M : ∀ (M z x)
       → z-predictable M z x ≡ ⟨ x , true ⟩
       → Any (λ nd → x ∈ pi nd z) (extractDists M)
prop1M = ?


P[_❘_]_ : StallClasses → ℝ → Model → ℝ
P[_❘_]_ = ?

prop2 : ∀ (M τ x k)
      → classify M τ x ≡ k
      → τ < (P[ k ❘ x ] M)
      --→ τ < (P[ k ❘ x ] M) × τ-confident M τ x ≡ true
prop2 = ?

prop3 : ∀ (M z τ x)
      → safety-envelope M z τ x ≡ true
      → (Any (λ nd → x ∈ pi nd z) (extractDists M))
        × ∃[ k ] (classify M τ x ≡ k × τ < P[ k ❘ x ] M)
prop3 = ?
