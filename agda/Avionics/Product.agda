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

map : ∀ {A B C D : Set}
    → (A → C) → (B → D) → A × B → C × D
map f g ⟨ x , y ⟩ = ⟨ f x , g y ⟩

map₁ : ∀ {A B C : Set}
     → (A → C) → A × B → C × B
map₁ f = map f (λ x → x)

map₂ : ∀ {A B D : Set}
     → (B → D) → A × B → A × D
map₂ g = map (λ x → x) g
