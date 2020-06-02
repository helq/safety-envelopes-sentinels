{-# OPTIONS --allow-unsolved-metas #-}

-- Careful with this: https://github.com/agda/agda/issues/543
-- Not necessary since Agda: 2.6.1
{-# OPTIONS --irrelevant-projections #-}

module Avionics.Probability where

open import Data.Nat using (ℕ; zero; suc)
open import Relation.Unary using (_∈_)

open import Avionics.Real using (
    ℝ; _+_; _-_; _*_; _÷_; _^_; √_; 1/_; _^2;
    -1/2; π; e; 2ℝ;
    ⟨0,∞⟩; [0,∞⟩; [0,1];
    >0→≢0; >0→≥0; >0*>0→>0; ≢0*≢0→≢0; 2>0; π>0; e^x>0; q>0→√q>0)

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
    .σ>0 : σ ∈ ⟨0,∞⟩

  dist : Dist ℝ
  dist = record
    {
      pdf = pdf
    ; cdf = ?
    ; pdf→[0,∞⟩ = ?
    --; pdf→[0,∞⟩ = λ x → >0→≥0 (>0*>0→>0 (σ>0→1/⟨σ√2π⟩>0 {σ>0}) (e^x>0 ?))
    ; cdf→[0,1] = ?
    }
    where
      -- proofs
      2π>0 = >0*>0→>0 2>0 π>0
      2π≥0 = >0→≥0 2π>0

      -- computations
      2π = (2ℝ * π)
      √2π = (√ 2π) {2π≥0}

      -- proofs
      σ>0→σ√2π>0 : (σ>0 : σ ∈ ⟨0,∞⟩) → (σ * √2π) ∈ ⟨0,∞⟩
      σ>0→σ√2π>0 σ>0 = >0*>0→>0 σ>0 (q>0→√q>0 2π 2π>0)

      --σ>0→1/⟨σ√2π⟩>0 : .{σ>0 : σ ∈ ⟨0,∞⟩} → 1/ (σ * √2π) ∈ ⟨0,∞⟩
      --σ>0→1/⟨σ√2π⟩>0 {σ>0} = ? -- p>0→1/p>0 σ>0→σ√2π>0 -- TODO: implement p>0→1/p>0

      -- computation
      1/⟨σ√2π⟩ = (1/ (σ * √2π)) {>0→≢0 (σ>0→σ√2π>0 σ>0)}

      pdf : ℝ → ℝ
      pdf x = 1/⟨σ√2π⟩ * e ^ (-1/2 * (⟨x-μ⟩÷σ ^2))
        where
          ⟨x-μ⟩÷σ = ((x - μ) ÷ σ) {>0→≢0 σ>0}

--MultiNormal : ∀ {n : ℕ} → Vec ℝ n → Mat ℝ n n → Dist (Vec ℝ n)
--MultiNormal {n} means cov = record
--  {
--    pdf = ?
--  ; cdf = ?
--  }

--Things to prove using this approach
--_ : ∀ (mean1 std1 mean2 std2 x)
--  → Dist.pdf (Normal mean1 std1) x + Dist.pdf (Normal mean2 std2) x ≡ Dist.pdf (Normal (mean1 + mean2) (...))
