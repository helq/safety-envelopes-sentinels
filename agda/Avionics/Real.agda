module Avionics.Real where

open import Algebra.Definitions using (LeftIdentity; RightIdentity; Commutative)
open import Data.Bool using (Bool; _∧_)
open import Data.Float using (Float)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ)
open import Level using (0ℓ; _⊔_) renaming (suc to lsuc)
open import Relation.Binary using (Decidable; _Preserves_⟶_)
open import Relation.Binary.Definitions using (Transitive; Trans)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)
open import Relation.Nullary using (Dec; yes; no)
open import Relation.Nullary.Decidable using (False; ⌊_⌋)
open import Relation.Unary using (Pred; _∈_)

infix  4 _<_ _≤_ _<ᵇ_ _≤ᵇ_
infixl 6 _+_ _-_
infixl 7 _*_

infix 4 _≟_

postulate
  ℝ : Set
  -- TODO: `fromFloat` should return `Maybe ℝ`
  fromFloat : Float → ℝ
  toFloat : ℝ → Float
  fromℕ : ℕ → ℝ

  _+_ _*_ _^_ : ℝ → ℝ → ℝ
  -_ abs _^2 : ℝ → ℝ
  e π 0ℝ 1ℝ -1/2 1/2 2ℝ : ℝ

  -- This was inspired on how the standard library handles things.
  -- See: https://plfa.github.io/Decidable/
  _<_ _≤_ : ℝ → ℝ → Set

  _≟_ : Decidable {A = ℝ} _≡_
  _≤?_ : (m n : ℝ) → Dec (m ≤ n)
  _<?_ : (m n : ℝ) → Dec (m < n)

_<ᵇ_ _≤ᵇ_ _≡ᵇ_ : ℝ → ℝ → Bool
p <ᵇ q = ⌊ p <? q ⌋
p ≤ᵇ q = ⌊ p ≤? q ⌋
p ≡ᵇ q = ⌊ p ≟ q ⌋

_≢0 : ℝ → Set
p ≢0 = False (p ≟ 0ℝ)

Subset : Set → Set _
Subset A = Pred A 0ℓ

-- Dangerous definitions, but necessary!
postulate
  ⟨0,∞⟩ [0,∞⟩ [0,1] : Subset ℝ

  1/_ : (p : ℝ) → ℝ
  √_ : (x : ℝ) → ℝ

_÷_ : (p q : ℝ) → ℝ
(p ÷ q) = p * (1/ q)

-- Dangerous definitions, but necessary!
postulate
  m÷n<o≡m<o*n : ∀ m n o → (m ÷ n <ᵇ o) ≡ (m <ᵇ o * n)
  m<o÷n≡m*n<o : ∀ m n o → (m <ᵇ o ÷ n) ≡ (m * n <ᵇ o)

_-_ : ℝ → ℝ → ℝ
p - q = p + (- q)

postulate
  double-neg : ∀ (x y : ℝ) → y - (y - x) ≡ x
  neg-involutive : ∀ x → -(- x) ≡ x
  neg-distrib-+ : ∀ m n → - (m + n) ≡ (- m) + (- n)
  neg-def : ∀ m → 0ℝ - m ≡ - m
  m-m≡0 : ∀ m → m - m ≡ 0ℝ
  neg-distribˡ-* : ∀ x y → - (x * y) ≡ (- x) * y
  √x^2≡absx : ∀ x → √ (x ^2) ≡ abs x

  >0→≢0 : ∀ {x : ℝ} → x ∈ ⟨0,∞⟩ → x ≢0
  >0→≥0 : ∀ {x : ℝ} → x ∈ ⟨0,∞⟩ → x ∈ [0,∞⟩
  >0*>0→>0 : ∀ {p q : ℝ} → p ∈ ⟨0,∞⟩ → q ∈ ⟨0,∞⟩ → (p * q) ∈ ⟨0,∞⟩
  ≢0*≢0→≢0 : ∀ {p q : ℝ} → p ≢0 → q ≢0 → (p * q) ≢0

  2>0 : 2ℝ ∈ ⟨0,∞⟩
  π>0 : π ∈ ⟨0,∞⟩

  e^x>0 : (x : ℝ) → (e ^ x) ∈ ⟨0,∞⟩
  x*x≡x^2 : ∀ x → x * x ≡ x ^2
  x^2*y^2≡⟨xy⟩^2 : ∀ x y → (x ^2) * (y ^2) ≡ (x * y)^2
  1/x^2≡⟨1/x⟩^2 : ∀ x → 1/ (x ^2) ≡ (1/ x)^2
  --√q≥0 : (q : ℝ) → (0≤q : q ∈ [0,∞⟩) → (√ q) {0≤q} ∈ [0,∞⟩
  --q>0→√q>0 : {q : ℝ} → (0<q : q ∈ ⟨0,∞⟩) → (√ q) {>0→≥0 0<q} ∈ ⟨0,∞⟩

  0≤→[0,∞⟩ : {n : ℝ} → 0ℝ ≤ n → n ∈ [0,∞⟩
  0<→⟨0,∞⟩ : {n : ℝ} → 0ℝ < n → n ∈ ⟨0,∞⟩
  [0,∞⟩→0≤ : {n : ℝ} → n ∈ [0,∞⟩ → 0ℝ ≤ n
  ⟨0,∞⟩→0< : {n : ℝ} → n ∈ ⟨0,∞⟩ → 0ℝ < n

  m≤n→m-p≤n-p : {m n p : ℝ} → m ≤ n → m - p ≤ n - p
  m<n→m-p<n-p : {m n p : ℝ} → m < n → m - p < n - p

  -- trans-≤ reduces to: {i j k : ℝ} → i ≤ j → j ≤ k → i ≤ k
  trans-≤ : Transitive _≤_
  <-transˡ : Trans _<_ _≤_ _<_

  --+-identityˡ : LeftIdentity 0ℝ _+_
  --+-identityʳ : RightIdentity 0ℝ _+_
  +-identityˡ : ∀ x → 0ℝ + x ≡ x
  +-identityʳ : ∀ x → x + 0ℝ ≡ x
  --+-comm : Commutative _+_
  +-comm : ∀ m n → m + n ≡ n + m
  --+-assoc : Associative _+_
  +-assoc : ∀ m n o → m + (n + o) ≡ (m + n) + o

  *-comm : ∀ m n → m * n ≡ n * m
  *-assoc : ∀ m n o → m * (n * o) ≡ (m * n) * o

  --0ℝ ≟_
  0≟0≡yes0≡0 : (0ℝ ≟ 0ℝ) ≡ yes refl

  -- TODO: These properties should be written for _<_ instead of _<ᵇ_
  abs<x→<x∧-x< : ∀ {u x} → (abs u <ᵇ x) ≡ ((u <ᵇ x) ∧ (- x <ᵇ u))
  --neg-mono-<-> : -_ Preserves _<_ ⟶ (λ a b -> b < a)
  neg-mono-<-> : ∀ m n → (m <ᵇ n) ≡ (- n <ᵇ - m)
  --+-monoˡ-< : ∀ n → (_+ n) Preserves _<_ ⟶ _<_
  +-monoˡ-< : ∀ n o p → (o <ᵇ p) ≡ (n + o <ᵇ n + p)

-- One of the weakest points in the whole library architecture!!!
-- This is wrong, really wrong, but useful
{-# COMPILE GHC ℝ = type Double #-}
{-# COMPILE GHC fromFloat = \x -> x #-}
{-# COMPILE GHC toFloat = \x -> x #-}
{-# COMPILE GHC fromℕ = fromIntegral #-}

{-# COMPILE GHC _<ᵇ_ = (<) #-}
{-# COMPILE GHC _≤ᵇ_ = (<=) #-}
{-# COMPILE GHC _≡ᵇ_ = (==) #-}

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
{-# COMPILE GHC 1/_ = \x -> (1/x) #-}
{-# COMPILE GHC _÷_ = \x y -> (x/y) #-}
{-# COMPILE GHC √_  = \x -> sqrt x #-}
