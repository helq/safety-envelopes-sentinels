module Avionics.Real where

open import Data.Bool using (Bool)
open import Data.Float using (Float)
open import Level using (0ℓ; _⊔_) renaming (suc to lsuc)
open import Relation.Unary using (Pred; _∈_)

infix  4 _<_
infixl 6 _+_ _-_
infixl 7 _*_

Subset : Set → Set _
Subset A = Pred A 0ℓ

postulate
  ℝ : Set
  fromFloat : Float → ℝ 
  toFloat : ℝ → Float

  _<_ _≤_ : ℝ → ℝ → Bool
  _+_ _-_ _*_ _^_ : ℝ → ℝ → ℝ
  e π 0ℝ 1ℝ -1/2 2ℝ : ℝ

  [0,∞⟩ [0,1] : Subset ℝ

  --WORST OF THE WORST. THIS IS WRONG; REALLY WRONG!!!
  --TODO: The following require a lot of care to work properly
  --Check Data/Rational/Base.agda for examples of how to care
  --They probably require dot pattern https://agda.readthedocs.io/en/v2.6.1/language/function-definitions.html#dot-patterns
  1/_ : ℝ → ℝ
  _÷_ : ℝ → ℝ → ℝ
  √_ : ℝ → ℝ
  -- Use one of the alternatives
  --√_ : ∀ (x : ℝ) → ∀ {x ∈ [0,∞⟩} → ℝ
  --√_ : ∀ (x : ℝ) → (x ∈ [0,∞⟩) → ℝ

-- One of the weakest points in the whole library architecture!!!
-- This is wrong, really wrong, but useful
{-# COMPILE GHC ℝ = type Double #-}
{-# COMPILE GHC fromFloat = \x -> x #-}
{-# COMPILE GHC toFloat = \x -> x #-}

{-# COMPILE GHC _<_ = (<) #-}
{-# COMPILE GHC _≤_ = (<=) #-}

{-# COMPILE GHC _+_ = (+) #-}
{-# COMPILE GHC _-_ = (-) #-}
{-# COMPILE GHC _*_ = (*) #-}
{-# COMPILE GHC _^_ = (**) #-}

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
