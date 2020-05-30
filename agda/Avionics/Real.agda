module Avionics.Real where

open import Data.Bool using (Bool)
open import Data.Float using (Float)
open import Level using (0ℓ; _⊔_) renaming (suc to lsuc)
open import Relation.Unary using (Pred; _∈_)

infix  4 _<_
infixl 6 _+_ _-_
infixl 7 _*_

postulate
  ℝ : Set
  fromFloat : Float → ℝ 
  toFloat : ℝ → Float

  _<_ : ℝ → ℝ → Bool
  _+_ _-_ _*_ _^_ : ℝ → ℝ → ℝ
  e π : ℝ

  --The following require a lot of care to work properly
  --Check Data/Rational/Base.agda for examples of how to care
  --They probably require dot pattern https://agda.readthedocs.io/en/v2.6.1/language/function-definitions.html#dot-patterns
  --1/_ : ℝ → ℝ
  --_÷_ : ℝ → ℝ
  --√_ : ℝ → ℝ

-- Weakest point in the whole real proof architecture!!!
-- This is wrong, really wrong, but useful
{-# COMPILE GHC ℝ = type Double #-}
{-# COMPILE GHC fromFloat = \x -> x #-}
{-# COMPILE GHC toFloat = \x -> x #-}

{-# COMPILE GHC _<_ = (<) #-}

{-# COMPILE GHC _+_ = (+) #-}
{-# COMPILE GHC _-_ = (-) #-}
{-# COMPILE GHC _*_ = (*) #-}
{-# COMPILE GHC _^_ = (**) #-}

{-# COMPILE GHC e = 2.71828182845904523536 #-}
{-# COMPILE GHC π = 3.14159265358979323846 #-}

Subset : Set → Set _
Subset A = Pred A 0ℓ

postulate
  [0,∞⟩ [0,1] : Subset ℝ
