{-# OPTIONS --allow-unsolved-metas #-}

module Avionics.Probability where

open import Data.Nat using (ℕ; zero; suc)
open import Relation.Unary using (_∈_)

open import Avionics.Real using (ℝ; _+_; [0,∞⟩; [0,1])

--postulate
--  Vec : Set → ℕ → Set
--  Mat : Set → ℕ → ℕ → Set

record Dist (Input : Set) : Set where
  field
    pdf : Input → ℝ
    cdf : Input → ℝ
    pdf→[0,∞⟩ : ∀ x → pdf x ∈ [0,∞⟩
    cdf→[0,1] : ∀ x → cdf x ∈ [0,1]
    --∫pdf≡cdf : ∫ pdf ≡ cdf
    --∫pdf[-∞,∞]≡1ℝ : ∫ pdf [ -∞ , ∞ ] ≡ 1ℝ

record NormalDist : Set where
  constructor ND
  field
    μ : ℝ
    σ : ℝ
    --σ>0 : {}
    -- TODO: Add proof that σ > 0
    --       Should use: https://agda.readthedocs.io/en/v2.6.1/language/irrelevance.html

  dist : Dist ℝ
  dist = record
    {
      pdf = ? -- λ x → 1/ (σ * √(2 * pi)) * e ^ (- 1/2 * ((x - μ) ÷ σ) ^ 2)
    ; cdf = ?
    ; pdf→[0,∞⟩ = ?
    ; cdf→[0,1] = ?
    }

--MultiNormal : ∀ {n : ℕ} → Vec ℝ n → Mat ℝ n n → Dist (Vec ℝ n)
--MultiNormal {n} means cov = record
--  {
--    pdf = ?
--  ; cdf = ?
--  }

--Things to prove using this approach
--_ : ∀ (mean1 std1 mean2 std2 x)
--  → Dist.pdf (Normal mean1 std1) x + Dist.pdf (Normal mean2 std2) x ≡ Dist.pdf (Normal (mean1 + mean2) (...))
