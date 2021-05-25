{-# OPTIONS --allow-unsolved-metas #-}

module Avionics.Probability where

--open import Data.Fin using (Fin; fromℕ<)
open import Data.Nat using (ℕ; zero; suc)
open import Data.Product using (_×_; _,_)
open import Data.Vec using (Vec; lookup)
open import Relation.Binary.PropositionalEquality using (refl)
open import Relation.Unary using (_∈_)

open import Avionics.Real using (
    ℝ; _+_; _-_; _*_; _÷_; _^_; √_; 1/_; _^2;
    -1/2; π; e; 2ℝ; 1ℝ; 0ℝ;
    ⟨0,∞⟩; [0,∞⟩; [0,1])

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
    --.0<σ : 0ℝ <? σ

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

-- Bivariate Normal Distribution
-- Representation taken from: https://upload.wikimedia.org/wikipedia/commons/a/a2/Cumulative_function_n_dimensional_Gaussians_12.2013.pdf
record BiNormalDist : Set where
  constructor ND
  field
    μ₁ : ℝ
    μ₂ : ℝ
    σ₁ : ℝ
    σ₂ : ℝ
    ρ : ℝ

  μ : ℝ × ℝ
  μ = μ₁ , μ₂

  Σ : ℝ × ℝ × ℝ × ℝ
  Σ = σ₁ , (ρ * σ₁ * σ₂) , (ρ * σ₁ * σ₂) , σ₂

  Σ-¹ : ℝ × ℝ × ℝ × ℝ
  Σ-¹ = ( 1/1-ρ² * (1ℝ ÷ σ₁)
        , 1/1-ρ² * -ρ/σ₁σ₂
        , 1/1-ρ² * -ρ/σ₁σ₂
        , 1/1-ρ² * (1ℝ ÷ σ₂))
    where 1/1-ρ² = 1ℝ ÷ (1ℝ - ρ * ρ)
          -ρ/σ₁σ₂ = (0ℝ - ρ) ÷ (σ₁ * σ₂)

  dist : Dist ℝ
  dist = record
    {
      pdf = ?
    ; cdf = ?
    ; pdf→[0,∞⟩ = ?
    ; cdf→[0,1] = ?
    }

Mat : ℕ → Set
Mat n = Vec (Vec ℝ n) n

-- Multivariate Normal Distribution
record MultiNormal : Set where
  constructor MultiND
  field
    n : ℕ
    μ : Vec ℝ n
    Σ : Mat n

  dist : Dist (Vec ℝ n)
  dist = record
    {
      pdf = ?
    ; cdf = ?
    ; pdf→[0,∞⟩ = ?
    ; cdf→[0,1] = ?
    }

--num : Vec ℕ 2 → ℕ
--num vec = lookup vec (fromℕ< {0} _)

--num2 : MultiNormal → ℝ
--num2 mn = lookup μ (fromℕ< {0} _)
--  where open MultiNormal mn using (n; μ; Σ) --; dist)
--        --open Dist dist using (pdf)
