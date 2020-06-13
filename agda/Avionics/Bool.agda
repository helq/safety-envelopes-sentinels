module Avionics.Bool where

open import Data.Bool using (Bool; true; _∧_; T)
open import Data.Unit using (⊤; tt)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

open import Avionics.Product using (_×_; ⟨_,_⟩)

--TODO: Replace with T⇔≡ from standard library
≡→T : ∀ {b : Bool} → b ≡ true → T b
≡→T refl = tt

T∧→× : ∀ {x y} → T (x ∧ y) → (T x) × (T y)
T∧→× {true} {true} tt = ⟨ tt , tt ⟩
--TODO: Find a way to extract the function below from `T-∧` (standard library)
--T∧→× : ∀ {x y} → T (x ∧ y) → (T x) P.× (T y)
--T∧→× {x} {y} = ? -- Equivalence.to (T-∧ {x} {y})
