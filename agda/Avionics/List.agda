module Avionics.List where

open import Data.Bool using (Bool; true; false; T)
open import Data.List as List using (List; []; _∷_; any)
open import Data.List.Relation.Unary.Any using (Any; here; there)
open import Function using (_∘_)
open import Relation.Binary.PropositionalEquality using (_≡_; inspect; [_]; refl)

open import Avionics.Bool using (≡→T)

≡→any : ∀ {a} {A : Set a} (f) (ns : List A)
        → any f ns ≡ true
        → Any (T ∘ f) ns
        --→ Any (λ x → T (f x)) ns
≡→any f [] ()
≡→any f (n ∷ ns) any-f-⟨n∷ns⟩≡true with f n | inspect f n
... | true  | [ fn≡t ] = here (≡→T fn≡t)
... | false | _        = there (≡→any f ns any-f-⟨n∷ns⟩≡true)

any→≡ : ∀ {a} {A : Set a} (f) (ns : List A)
        → Any (T ∘ f) ns
        → any f ns ≡ true
any→≡ f (n ∷ _) (here _) with f n
... | true = refl -- or: T→≡ [*proof*from*here*]
any→≡ f (n ∷ ns) (there Any[T∘f]ns) with f n
... | true = refl
... | false = any→≡ f ns Any[T∘f]ns


any-map : ∀ {A B : Set} {p : B → Set} {ls : List A}
          (f : A → B)
        → Any p (List.map f ls)
        → Any (p ∘ f) ls
--any-map {ls = []} _ ()
any-map {ls = l ∷ ls} f (here pb) = here pb
any-map {ls = l ∷ ls} f (there pb) = there (any-map f pb)

any-map-rev : ∀ {A B : Set} {p : B → Set} {ls : List A}
          (f : A → B)
        → Any (p ∘ f) ls
        → Any p (List.map f ls)
any-map-rev {ls = l ∷ ls} f (here pb) = here pb
any-map-rev {ls = l ∷ ls} f (there pb) = there (any-map-rev f pb)
