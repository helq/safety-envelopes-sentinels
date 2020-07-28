module ExtInterface.Data.Maybe where

open import Function using (_∘_)
open import Level using (Level)

private
  variable
    a b c : Level
    A : Set a
    B : Set b
    C : Set c

data Maybe (A : Set) : Set where
  nothing : Maybe A
  just    : (x : A) → Maybe A

maybe : ∀ {A : Set} {B : Maybe A → Set} →
        ((x : A) → B (just x)) → B nothing → (x : Maybe A) → B x
maybe j n (just x) = j x
maybe j n nothing  = n

map : (A → B) → Maybe A → Maybe B
map f = maybe (just ∘ f) nothing

-- Monad: bind

infixl 1 _>>=_
_>>=_ : Maybe A → (A → Maybe B) → Maybe B
nothing >>= f = nothing
just a  >>= f = f a

{-# COMPILE GHC Maybe = data Maybe (Nothing | Just) #-}
