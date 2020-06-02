module Avionics.Real where

open import Data.Bool using (Bool)
open import Data.Nat using (ℕ)
open import Data.Float using (Float)
open import Level using (0ℓ; _⊔_) renaming (suc to lsuc)
open import Relation.Unary using (Pred; _∈_)
open import Relation.Nullary.Decidable using (False)
open import Relation.Binary using (Decidable)
open import Relation.Binary.PropositionalEquality using (_≡_)

infix  4 _<_
infixl 6 _+_ _-_
infixl 7 _*_

infix 4 _≟_

postulate
  ℝ : Set
  -- TODO: `fromFloat` should return `Maybe ℝ`
  fromFloat : Float → ℝ
  toFloat : ℝ → Float
  fromℕ : ℕ → ℝ

  _<_ _≤_ : ℝ → ℝ → Bool
  _+_ _-_ _*_ _^_ : ℝ → ℝ → ℝ
  _^2 : ℝ → ℝ
  e π 0ℝ 1ℝ -1/2 2ℝ : ℝ

  _≟_ : Decidable {A = ℝ} _≡_

_≢0 : ℝ → Set
p ≢0 = False (p ≟ 0ℝ)

Subset : Set → Set _
Subset A = Pred A 0ℓ

postulate
  ⟨0,∞⟩ [0,∞⟩ [0,1] : Subset ℝ

  1/_ : (p : ℝ) → .{p≢0 : p ≢0} → ℝ
  √_ : (x : ℝ) → .{0≤x : x ∈ [0,∞⟩} → ℝ

_÷_ : (p q : ℝ) → .{q≢0 : q ≢0} → ℝ
(p ÷ q) {q≢0} = p * (1/ q) {q≢0}

postulate
  >0→≢0 : ∀ {x : ℝ} → x ∈ ⟨0,∞⟩ → x ≢0
  >0→≥0 : ∀ {x : ℝ} → x ∈ ⟨0,∞⟩ → x ∈ [0,∞⟩
  >0*>0→>0 : ∀ {p q : ℝ} → p ∈ ⟨0,∞⟩ → q ∈ ⟨0,∞⟩ → (p * q) ∈ ⟨0,∞⟩
  ≢0*≢0→≢0 : ∀ {p q : ℝ} → p ≢0 → q ≢0 → (p * q) ≢0

  2>0 : 2ℝ ∈ ⟨0,∞⟩
  π>0 : π ∈ ⟨0,∞⟩

  e^x>0 : (x : ℝ) → (e ^ x) ∈ ⟨0,∞⟩
  √q≥0 : (q : ℝ) → (0≤q : q ∈ [0,∞⟩) → (√ q) {0≤q} ∈ [0,∞⟩
  q>0→√q>0 : (q : ℝ) → (0<q : q ∈ ⟨0,∞⟩) → (√ q) {>0→≥0 0<q} ∈ ⟨0,∞⟩

-- One of the weakest points in the whole library architecture!!!
-- This is wrong, really wrong, but useful
{-# COMPILE GHC ℝ = type Double #-}
{-# COMPILE GHC fromFloat = \x -> x #-}
{-# COMPILE GHC toFloat = \x -> x #-}
{-# COMPILE GHC fromℕ = fromIntegral #-}

{-# COMPILE GHC _<_ = (<) #-}
{-# COMPILE GHC _≤_ = (<=) #-}

{-# COMPILE GHC _+_ = (+) #-}
{-# COMPILE GHC _-_ = (-) #-}
{-# COMPILE GHC _*_ = (*) #-}
{-# COMPILE GHC _^_ = (**) #-}

{-# COMPILE GHC _^2 = (**2) #-}

{-# COMPILE GHC e = 2.71828182845904523536 #-}
{-# COMPILE GHC π = 3.14159265358979323846 #-}
{-# COMPILE GHC 0ℝ = 0 #-}
{-# COMPILE GHC 1ℝ = 1 #-}
{-# COMPILE GHC 2ℝ = 2 #-}
{-# COMPILE GHC -1/2 = -1/2 #-}

-- REAAALY CAREFUL WITH THIS!
-- TODO: Add some runtime checking to this. Fail hard if divisor is zero
{-# COMPILE GHC 1/_ = (1/) #-}
{-# COMPILE GHC _÷_ = (/) #-}
{-# COMPILE GHC √_ = sqrt #-}
