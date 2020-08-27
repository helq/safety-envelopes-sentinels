module Avionics.Bool where

open import Data.Bool using (Bool; true; false; _∧_; T)
open import Data.Unit using (⊤; tt)
open import Data.Product using (_×_; _,_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; inspect; [_])

--open import Avionics.Product using (_×_; ⟨_,_⟩)

--TODO: Replace with T⇔≡ from standard library
≡→T : ∀ {b : Bool} → b ≡ true → T b
≡→T refl = tt

T→≡ : ∀ {b : Bool} → T b → b ≡ true
T→≡ {true} tt = refl

T∧→× : ∀ {x y} → T (x ∧ y) → (T x) × (T y)
T∧→× {true} {true} tt = tt , tt
--TODO: Find a way to extract the function below from `T-∧` (standard library)
--T∧→× {x} {y} = ? -- Equivalence.to (T-∧ {x} {y})

×→T∧ : ∀ {x y} → (T x) × (T y) → T (x ∧ y)
×→T∧ {true} {true} (tt , tt) = tt


lem∧ : {a b : Bool} → a ∧ b ≡ true → a ≡ true × b ≡ true
lem∧ {true} {true} refl = refl , refl

∧≡true→×≡ : ∀ {A B : Set} {f : A → Bool} {g : B → Bool}
            (n : A) (m : B)
          → f n ∧ g m ≡ true
          → f n ≡ true × g m ≡ true
∧≡true→×≡ {f = f} {g = g} n m fn∧gm≡true = lem∧ {f n} {g m} fn∧gm≡true
