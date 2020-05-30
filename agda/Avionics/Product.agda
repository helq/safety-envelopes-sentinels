module Avionics.Product where

-- TODO: Write to Agda team about the lack of compilability of Sigma.
-- I assumed that the builtin flag would allow to compile Σ into (,)
-- but it doesn't. That's why this microfile exists

infixr 4 ⟨_,_⟩
infixr 2 _×_

data _×_ (A B : Set) : Set where
  ⟨_,_⟩ : A → B → A × B

{-# COMPILE GHC _×_ = data (,) ((,)) #-} -- Yeah, kinda abstract

proj₁ : ∀ {A B : Set} → A × B → A
proj₁ ⟨ x , y ⟩ = x

proj₂ : ∀ {A B : Set} → A × B → B
proj₂ ⟨ x , y ⟩ = y
