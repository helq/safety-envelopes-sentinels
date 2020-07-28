{-# OPTIONS --allow-unsolved-metas #-}

module Avionics.Probability where

open import Data.Nat using (ℕ; zero; suc)
open import Relation.Unary using (_∈_)

open import Avionics.Real using (
    ℝ; _+_; _-_; _*_; _÷_; _^_; √_; 1/_; _^2;
    -1/2; π; e; 2ℝ;
    ⟨0,∞⟩; [0,∞⟩; [0,1])

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

  dist : Dist ℝ
  dist = record
    {
      pdf = pdf
    ; cdf = ?
    ; pdf→[0,∞⟩ = ?
    ; cdf→[0,1] = ?
    }
    where
      √2π = (√ (2ℝ * π))
      1/⟨σ√2π⟩ = (1/ (σ * √2π))

      pdf : ℝ → ℝ
      pdf x = 1/⟨σ√2π⟩ * e ^ (-1/2 * (⟨x-μ⟩÷σ ^2))
        where
          ⟨x-μ⟩÷σ = ((x - μ) ÷ σ)

--MultiNormal : ∀ {n : ℕ} → Vec ℝ n → Mat ℝ n n → Dist (Vec ℝ n)
--MultiNormal {n} means cov = record
--  {
--    pdf = ?
--  ; cdf = ?
--  }

--Things to prove using this approach
--_ : ∀ (mean1 std1 mean2 std2 x)
--  → Dist.pdf (Normal mean1 std1) x + Dist.pdf (Normal mean2 std2) x ≡ Dist.pdf (Normal (mean1 + mean2) (...))
