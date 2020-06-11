module Avionics.Bool where

open import Data.Bool using (Bool; true; T)
open import Data.Unit using (⊤; tt)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

≡→T : ∀ {b : Bool} → b ≡ true → T b
≡→T refl = tt
