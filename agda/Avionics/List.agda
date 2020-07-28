module Avionics.List where

open import Data.Bool using (Bool; true; false; T)
open import Data.List using (List; []; _∷_; any)
open import Data.List.Relation.Unary.Any using (Any; here; there)
open import Function using (_∘_)
open import Relation.Binary.PropositionalEquality using (_≡_; inspect; [_])

open import Avionics.Bool using (≡→T)

any-val : ∀ {a} {A : Set a} (f) (ns : List A)
        → any f ns ≡ true
        → Any (T ∘ f) ns
        --→ Any (λ x → T (f x)) ns
any-val f [] ()
any-val f (n ∷ ns) any-f-⟨n∷ns⟩≡true with f n | inspect f n
... | true  | [ fn≡t ] = here (≡→T fn≡t)
... | false | _        = there (any-val f ns any-f-⟨n∷ns⟩≡true)
