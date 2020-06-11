module Avionics.SafetyEnvelopes.Properties where

open import Data.Bool using (Bool; true; false; _∧_; _∨_; T)
open import Data.Bool.Properties using (∨-identityʳ; ∨-zeroˡ)
open import Data.List using (List; []; _∷_; any; map; foldl; length)
open import Data.List.Relation.Unary.Any using (satisfied)
open import Data.Product using (∃-syntax; _,_; map₂)
open import Relation.Binary.PropositionalEquality
    using (_≡_; refl; cong; subst; sym; trans)
open import Relation.Nullary.Decidable using (toWitness; fromWitness)

open import Avionics.Bool using (≡→T)
open import Avionics.List using (any-val)
open import Avionics.Real
    using (ℝ; _+_; _-_; _*_; _÷_; _^_; _<ᵇ_; _≤ᵇ_; _≤_; _<_; _≢0;
           0ℝ; 1ℝ; 2ℝ; _^2; √_; fromℕ;
           ⟨0,∞⟩; [0,∞⟩;
           <-transˡ; 2>0; ⟨0,∞⟩→0<; 0<→⟨0,∞⟩; >0→≢0; >0→≥0; q>0→√q>0)
    renaming (fromFloat to ff; toFloat to tf)
open import Avionics.Probability using (Dist; NormalDist; ND)
open import Avionics.Product using (_×_; ⟨_,_⟩; proj₁; proj₂)
--open import Avionics.SafetyEnvelopes using (inside; mean-cf)

-- TODO: Delete this section after the whole formula can be proved
-- use definition from Avionics.SafetyEnvelopes
inside : NormalDist → ℝ → ℝ → Bool
inside nd m x = (μ - m * σ) <ᵇ x
--inside nd m x = ((μ - m * σ) <ᵇ x) ∧ (x <ᵇ (μ + m * σ))
  where open NormalDist nd using (μ; σ)

mean-cf : List NormalDist → ℝ → ℝ → ℝ × Bool
mean-cf nds m x = ⟨ x , any (λ nd → inside nd m x) nds ⟩

open NormalDist

--<ᵇ→< : ∀ {x y} → T (x <ᵇ y) → x < y
--<ᵇ→< = toWitness

prop1' : ∀ (nds m y)
       → mean-cf nds m y ≡ ⟨ y , true ⟩
       → ∃[ nd ] ((μ nd) - m * (σ nd) < y)
prop1' nds m y res≡y,true = map₂ toWitness ∃[nd],T<ᵇ
  where
    op = (λ nd → (μ nd) - m * (σ nd) <ᵇ y)
    res≡true = cong proj₂ res≡y,true
    ∃[nd],T<ᵇ = satisfied (any-val op nds res≡true)

--prop1 : ∀ (nds m y)
--      → mean-cf nds m y ≡ ⟨ y , true ⟩
--      → ∃[ nd ] ((μ nd) - m * (σ nd) < y  ×  y < (μ nd) + m * (σ nd))

--P⟨_⟩[_<X<_] = ?
--
--prop2 : ∀ (nds m x)
--      → mean-cf nds m x ≡ ⟨ x , true ⟩
--      → ∃[ nd ] (dist ∈ nds × ∃[ μ ] (∃[ σ ] ( P⟨ dist ⟩[ (μ - abs (μ - x)) <X< (μ + abs (μ - x)) ] )))
--prop2 = ?
