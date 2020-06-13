module Avionics.SafetyEnvelopes.Properties where

open import Data.Bool using (true)
open import Data.List.Relation.Unary.Any as Any using (Any; satisfied)
open import Data.Product using (∃-syntax)
open import Function using (_∘_)
open import Relation.Binary.PropositionalEquality using (_≡_; cong)
open import Relation.Nullary.Decidable using (toWitness)
open import Relation.Unary using (_∈_)

open import Avionics.Bool using (≡→T; T∧→×)
open import Avionics.List using (any-val)
open import Avionics.Real
    using (ℝ; _+_; _-_; _*_; _÷_; _^_; _<ᵇ_; _≤ᵇ_; _≤_; _<_; _<?_; _≢0;
           0ℝ; 1ℝ; 2ℝ; _^2; √_; fromℕ;
           ⟨0,∞⟩; [0,∞⟩;
           <-transˡ; 2>0; ⟨0,∞⟩→0<; 0<→⟨0,∞⟩; >0→≢0; >0→≥0; q>0→√q>0)
open import Avionics.Probability using (NormalDist)
open import Avionics.Product using (_×_; ⟨_,_⟩; proj₂; map)
open import Avionics.SafetyEnvelopes using (inside; z-predictable)

open NormalDist

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
      → z-predictable nds z x ≡ ⟨ x , true ⟩
      → Any (λ nd → x ∈ pi nd z) nds
prop1 nds z x res≡x,true = Any-x∈pi
  where
    res≡true = cong proj₂ res≡x,true

    toWitnessLeft  = λ nd → toWitness {P = (μ nd) - z * (σ nd) < x} {Q = (μ nd - z * σ nd) <? x}
    toWitnessRight = λ nd → toWitness {P = x < (μ nd) + z * (σ nd)} {Q = x <? (μ nd + z * σ nd)}
    toWitness' = λ nd → map (toWitnessLeft nd) (toWitnessRight nd)

    Any-bool = any-val (inside z x) nds res≡true
    Any-x∈pi = Any.map (λ {nd} → toWitness' nd ∘ T∧→×) Any-bool

-- From the prove above we can obtain the value `nd` and its prove `x ∈ pi nd z`
-- Note: An element of ∃[ nd ] (x ∈ pi nd z) is a tuple of the form ⟨ nd , proof ⟩
prop1' : ∀ (nds z x)
       → z-predictable nds z x ≡ ⟨ x , true ⟩
       → ∃[ nd ] (x ∈ pi nd z)
prop1' nds z x res≡x,true = satisfied (prop1 nds z x res≡x,true)

--P⟨_⟩[_<X<_] = ?
--
--prop2 : ∀ (nds x)
--      → z-predictable nds 2 x ≡ ⟨ x , true ⟩
--      → ∃[ nd ] (dist ∈ nds
--                × ∃[ μ ] (∃[ σ ] (
--                P⟨ dist ⟩[ (μ - abs (μ - x)) <X< (μ + abs (μ - x)) ] < 95% )))
--prop2 = ?
