module Avionics.SafetyEnvelopes.Properties where

open import Data.Bool using (Bool; true; false; _∧_; T)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List as List using (List; []; _∷_; any)
open import Data.List.Relation.Unary.Any as Any using (Any; here; there; satisfied)
open import Data.Maybe using (Maybe; just; nothing; is-just; _>>=_)
open import Data.Product as Prod using (∃-syntax; _×_; proj₁; proj₂) renaming (_,_ to ⟨_,_⟩)
open import Data.Sum using (_⊎_; inj₁; inj₂)
open import Function using (_∘_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; cong; cong₂; inspect; [_]; sym; trans)
open import Relation.Nullary using (Dec; yes; no; ¬_)
open import Relation.Nullary.Decidable using (toWitness; fromWitness)
open import Relation.Unary using (_∈_)

open import Avionics.Bool using (≡→T; T∧→×; ×→T∧; lem∧)
open import Avionics.List using (≡→any; any-map; any-map-rev; any→≡)
open import Avionics.Real
    using (ℝ; _+_; _-_; _*_; _÷_; _^_; _<ᵇ_; _≤_; _<_; _<?_; _≤?_; _≢0;
           0ℝ; 1ℝ; 2ℝ; _^2; √_; fromℕ;
           double-neg;
           ⟨0,∞⟩; [0,∞⟩;
           <-transˡ; 2>0; ⟨0,∞⟩→0<; 0<→⟨0,∞⟩; >0→≢0; >0→≥0;
           0≟0≡yes0≡0)
open import Avionics.Probability using (NormalDist; Dist)
open import Avionics.SafetyEnvelopes
    using (inside; z-predictable'; P[_|X=_]_; classify''; classify; M→pbs;
           StallClasses; Stall; NoStall; Uncertain;
           no-uncertain;
           safety-envelope; z-predictable; Model; τ-confident;
           Stall≡1-NoStall; NoStall≡1-Stall; ≤p→¬≤1-p; ≤1-p→¬≤p
           )

open NormalDist using (σ; μ)

--<ᵇ→< : ∀ {x y} → T (x <ᵇ y) → x < y
--<ᵇ→< = toWitness

-- Preliminary defitinions

-- `pi` is the prediction interval for the z score, i.e.,
-- pi(N (μ, σ), z) = [μ − zσ, μ + zσ]
pi : NormalDist → ℝ → ℝ → Set
pi nd z x =  (μ nd) - z * (σ nd) < x
          ×  x < (μ nd) + z * (σ nd)

extractDists : Model → List NormalDist
extractDists M = List.map (proj₁ ∘ proj₂) (Model.fM M)

------------------------------ Starting point - Theorem 1 ------------------------------
-- Proof of Theorem 1 (paper)
--
-- In words, the Property 1 says that:
--    The energy signal x is z-predictable iff there exist ⟨α, v⟩ s.t.
--    M(⟨α, v⟩)1 = di and x ∈ pi(di , z).
--
-- Notice that `Any (λ nd → x ∈ pi nd z) nds` translates to:
-- there exists nd such that `nd ∈ nds` and `x ∈ pi(nd, z)`
follows-def←' : ∀ (nds z x)
              → z-predictable' nds z x ≡ ⟨ x , true ⟩
              → Any (λ nd → x ∈ pi nd z) nds
follows-def←' nds z x res≡x,true = Any-x∈pi
  where
    res≡true = cong proj₂ res≡x,true

    -- the first `toWitness` takes a result `(μ nd - z * σ nd) <ᵇ x` (a
    -- boolean) and produces a proof of the type `(μ nd) - z * (σ nd) < x`
    -- assuming we have provided an operator `<?`
    toWitness' = λ nd → Prod.map (toWitness {Q = (μ nd - z * σ nd) <? x})
                                 (toWitness {Q = x <? (μ nd + z * σ nd)})

    -- We find the value for which `inside z x` becomes true in the list `nds`
    Any-bool = ≡→any (λ nd → inside nd z x) nds res≡true
    -- Converting the boolean proof into a proof at the type level
    Any-x∈pi = Any.map (λ {nd} → toWitness' nd ∘ T∧→×) Any-bool

-- forward proof
follows-def→' : ∀ (nds z x)
              → Any (λ nd → x ∈ pi nd z) nds
              → z-predictable' nds z x ≡ ⟨ x , true ⟩
follows-def→' nds z x any[x∈pi-z]nds = let
    -- converts a tuple of `(μ nd) - z * (σ nd) < x , x < (μ nd + z * σ nd)`
    -- (a proof) into a boolean
    fromWitness' nd = λ{⟨ μ-zσ<x , x<μ+zσ ⟩ →
                ×→T∧ ⟨ (fromWitness {Q = (μ nd - z * σ nd) <? x} μ-zσ<x)
                     , (fromWitness {Q = x <? (μ nd + z * σ nd)} x<μ+zσ)
                     ⟩}

    -- Converting from a proof on `_<_` to a proof on `_<ᵇ_`
    any[inside]nds = (Any.map (λ {nd} → fromWitness' nd) any[x∈pi-z]nds)

    -- Transforming the prove from Any into equality (_≡_)
    z-pred-x≡⟨x,true⟩ = any→≡ (λ nd → inside nd z x) nds any[inside]nds

     -- Extending the result from a single Bool value to a pair `ℝ × Bool`
  in cong (⟨ x ,_⟩) z-pred-x≡⟨x,true⟩

-- From the prove above we can obtain the value `nd` and its prove `x ∈ pi nd z`
-- Note: An element of ∃[ nd ] (x ∈ pi nd z) is a tuple of the form ⟨ nd , proof ⟩
--prop1← : ∀ (nds z x)
--       → z-predictable' nds z x ≡ ⟨ x , true ⟩
--       → ∃[ nd ] (x ∈ pi nd z)
--prop1← nds z x res≡x,true = satisfied (follows-def←' nds z x res≡x,true)

-- This proofs is telling us that `z-predictable` follows from the definition
follows-def← : ∀ (M z x)
             → z-predictable M z x ≡ ⟨ x , true ⟩
             → Any (λ nd → x ∈ pi nd z) (extractDists M)
follows-def← M z x res≡x,true = follows-def←' (extractDists M) z x res≡x,true

follows-def→ : ∀ (M z x)
             → Any (λ nd → x ∈ pi nd z) (extractDists M)
             → z-predictable M z x ≡ ⟨ x , true ⟩
follows-def→ M z x Any[x∈pi-nd-z]M = follows-def→' (extractDists M) z x Any[x∈pi-nd-z]M

-- ############ FINAL RESULT - Theorem 1 ############

-- In words: Given a Model `M` and parameter `z`, if `x` is z-predictable, then
-- there exists a pair ⟨α,v⟩ (angle of attack and velocity) such that they are
-- associated to a `nd` (Normal Distribution) and `x` falls withing the
-- Predictable Interval
theorem1← : ∀ (M z x)
          → z-predictable M z x ≡ ⟨ x , true ⟩
          → Any (λ{⟨ ⟨α,v⟩ , ⟨ nd , p ⟩ ⟩ → x ∈ pi nd z}) (Model.fM M)
theorem1← M z x res≡x,true = any-map (proj₁ ∘ proj₂) (follows-def← M z x res≡x,true)

-- The reverse of theorem1←
theorem1→ : ∀ (M z x)
          → Any (λ{⟨ ⟨α,v⟩ , ⟨ nd , p ⟩ ⟩ → x ∈ pi nd z}) (Model.fM M)
          → z-predictable M z x ≡ ⟨ x , true ⟩
theorem1→ M z x Any[⟨α,v⟩→x∈pi-nd-z]M = follows-def→ M z x (any-map-rev (proj₁ ∘ proj₂) Any[⟨α,v⟩→x∈pi-nd-z]M)

-- ################# Theorem 1 END ##################

------------------------------ Starting point - Theorem 2 ------------------------------
lem← : ∀ (pbs τ x k)
     → classify'' pbs τ x ≡ k
     → k ≡ Uncertain ⊎ ∃[ p ] ((P[ k |X= x ] pbs ≡ just p) × (τ ≤ p))
lem← pbs τ x k _ with P[ Stall |X= x ] pbs | inspect (P[ Stall |X=_] pbs) x
lem← _ _ _ Uncertain _ | nothing | [ P[k|X=x]≡nothing ] = inj₁ refl
lem← _ τ _ _       _   | just p  | [ _ ] with τ ≤? p | τ ≤? (1ℝ - p)
lem← _ _ _ Stall   _   | just p  | [ P[k|X=x]≡justp ] | yes τ≤p | no ¬τ≤1-p = inj₂ ⟨ p , ⟨ P[k|X=x]≡justp , τ≤p ⟩ ⟩
lem← _ _ _ NoStall _   | just p  | [ P[k|X=x]≡justp ] | no ¬τ≤p | yes τ≤1-p =
  let P[NoStall|X=x]≡just1-p = Stall≡1-NoStall P[k|X=x]≡justp
  in inj₂ ⟨ 1ℝ - p , ⟨ P[NoStall|X=x]≡just1-p , τ≤1-p ⟩ ⟩
lem← _ _ _ Uncertain _ | _       | _ | _ | _ = inj₁ refl

lem→' : ∀ (pbs τ x p)
      → (P[ Stall |X= x ] pbs) ≡ just p
      → τ ≤ (1ℝ - p)
      → classify'' pbs τ x ≡ NoStall
lem→' pbs _ x _ _ _ with P[ Stall |X= x ] pbs
lem→' _ τ _ _ _ _ | just p with τ ≤? p | τ ≤? (1ℝ - p)
lem→' _ _ _ _ _    _     | just p | no _    | yes _     = refl
lem→' _ _ _ _ refl τ≤1-p | just p | _       | no ¬τ≤1-p = ⊥-elim (¬τ≤1-p τ≤1-p)
lem→' _ _ _ _ refl τ≤1-p | just p | yes τ≤p | yes _     = ⊥-elim (¬τ≤p τ≤p)
  where ¬τ≤p = ≤1-p→¬≤p τ≤1-p

τ≤p→τ≤1-⟨1-p⟩ : ∀ τ p → τ ≤ p → τ ≤ 1ℝ - (1ℝ - p)
τ≤p→τ≤1-⟨1-p⟩ τ p τ≤p rewrite double-neg p 1ℝ = τ≤p

lem→ : ∀ (pbs τ x k)
     → ∃[ p ] (((P[ k |X= x ] pbs) ≡ just p) × (τ ≤ p))
     → classify'' pbs τ x ≡ k
lem→ pbs _ x Stall _ with P[ Stall |X= x ] pbs
lem→ _ τ _ _ _                      | just p with τ ≤? p | τ ≤? (1ℝ - p)
lem→ _ _ _ _ _                      | just p | yes _   | no  _     = refl
lem→ _ _ _ _ ⟨ _ , ⟨ refl , τ≤p ⟩ ⟩ | just p | no ¬τ≤p | _         = ⊥-elim (¬τ≤p τ≤p)
lem→ _ _ _ _ ⟨ _ , ⟨ refl , τ≤p ⟩ ⟩ | just p | yes _   | yes τ≤1-p = ⊥-elim (¬τ≤1-p τ≤1-p)
  where ¬τ≤1-p = ≤p→¬≤1-p τ≤p
lem→ pbs τ x NoStall ⟨ p , ⟨ P[k|X=x]≡justp , τ≤p ⟩ ⟩ = let
    P[S|X=x]≡just1-p = NoStall≡1-Stall P[k|X=x]≡justp
    τ≤1-⟨1-p⟩ = τ≤p→τ≤1-⟨1-p⟩ τ p τ≤p
  in lem→' pbs τ x (1ℝ - p) P[S|X=x]≡just1-p τ≤1-⟨1-p⟩

prop2M-prior← : ∀ (M τ x k)
              → classify M τ x ≡ k
              → k ≡ Uncertain ⊎ ∃[ p ] (((P[ k |X= x ] (M→pbs M)) ≡ just p) × (τ ≤ p))
prop2M-prior← M = lem← (M→pbs M)

prop2M-prior←' : ∀ (M τ x k)
               → k ≡ Stall ⊎ k ≡ NoStall
               → classify M τ x ≡ k
               → ∃[ p ] (((P[ k |X= x ] (M→pbs M)) ≡ just p) × (τ ≤ p))
prop2M-prior←' M τ x k _ cMτx≡k with prop2M-prior← M τ x k cMτx≡k
prop2M-prior←' _ _ _ Stall   (inj₁ _) _ | inj₂ P[k|X=x]≥τ = P[k|X=x]≥τ
prop2M-prior←' _ _ _ NoStall _        _ | inj₂ P[k|X=x]≥τ = P[k|X=x]≥τ

prop2M-prior→ : ∀ (M τ x k)
              → ∃[ p ] (((P[ k |X= x ] (M→pbs M)) ≡ just p) × (τ ≤ p))
              → classify M τ x ≡ k
prop2M-prior→ M = lem→ (M→pbs M)

prop2M← : ∀ (M τ x)
       → τ-confident M τ x ≡ true
       → ∃[ k ] ((classify M τ x ≡ k) × (k ≡ Stall ⊎ k ≡ NoStall))
prop2M← M τ x τconf≡true with classify M τ x
... | Stall   = ⟨ Stall ,   ⟨ refl , inj₁ refl ⟩ ⟩
... | NoStall = ⟨ NoStall , ⟨ refl , inj₂ refl ⟩ ⟩

prop2M→ : ∀ (M τ x k)
        → k ≡ Stall ⊎ k ≡ NoStall
        → classify M τ x ≡ k
        → τ-confident M τ x ≡ true
prop2M→ M τ x Stall   (inj₁ k≡Stall)   cMτx≡k = cong no-uncertain cMτx≡k
prop2M→ M τ x NoStall (inj₂ k≡NoStall) cMτx≡k = cong no-uncertain cMτx≡k

---- ############ FINAL RESULT - Theorem 2 ############
-- Theorem 2 says:
-- a classification k is τ-confident iff τ ≤ P[ k | X = x ]
theorem2← : ∀ (M τ x)
          → τ-confident M τ x ≡ true
          → ∃[ k ] (∃[ p ] (((P[ k |X= x ] (M→pbs M)) ≡ just p) × (τ ≤ p))) -- which means: τ ≤ P[ k | X = x ]
theorem2← M τ x τconf≡true = let -- prop2M-prior← M τ x k (prop2M← M τ x k τconf≡true)
  ⟨ k , ⟨ cMτx≡k , k≢Uncertain ⟩ ⟩ = prop2M← M τ x τconf≡true
  in ⟨ k , prop2M-prior←' M τ x k k≢Uncertain cMτx≡k ⟩

theorem2→ : ∀ (M τ x k)
          → k ≡ Stall ⊎ k ≡ NoStall
          → ∃[ p ] (((P[ k |X= x ] (M→pbs M)) ≡ just p) × (τ ≤ p)) -- which means: τ ≤ P[ k | X = x ]
          → τ-confident M τ x ≡ true
theorem2→ M τ x k k≢Uncertain ⟨p,⟩ = prop2M→ M τ x k k≢Uncertain (prop2M-prior→ M τ x k ⟨p,⟩)
---- ############ Theorem 2 END ############

------------------------------ Starting point - Theorem 3 ------------------------------
---- ############ FINAL RESULT - Theorem 3 ############
-- The final theorem is more a corolary. It follows from Theorem 1 and 2
prop3M← : ∀ (M z τ x)
        → safety-envelope M z τ x ≡ true
        → (Any (λ{⟨ ⟨α,v⟩ , ⟨ nd , p ⟩ ⟩ → x ∈ pi nd z}) (Model.fM M))
          × ∃[ k ] (∃[ p ] (((P[ k |X= x ] (M→pbs M)) ≡ just p) × (τ ≤ p)))
prop3M← M z τ x seM≡true = let
    -- Taking from the safety-envelope definition its components
    ⟨ left , τ-conf ⟩ =
              lem∧ {a = proj₂ (z-predictable M z x)}
                   {b = τ-confident M τ x}
                   seM≡true
    z-pred-x≡⟨x,true⟩ = cong (⟨ x ,_⟩) left
  in ⟨ theorem1← M z x z-pred-x≡⟨x,true⟩ , theorem2← M τ x τ-conf ⟩

prop3M→ : ∀ (M z τ x k)
        → k ≡ Stall ⊎ k ≡ NoStall
        → (Any (λ{⟨ ⟨α,v⟩ , ⟨ nd , p ⟩ ⟩ → x ∈ pi nd z}) (Model.fM M))
          × ∃[ p ] (((P[ k |X= x ] (M→pbs M)) ≡ just p) × (τ ≤ p))
        → safety-envelope M z τ x ≡ true
prop3M→ M z τ x k k≢Uncertain ⟨ Any[⟨α,v⟩→x∈pi-nd-z]M , ⟨p,⟩ ⟩ = let
  z-pred≡⟨x,true⟩ = theorem1→ M z x Any[⟨α,v⟩→x∈pi-nd-z]M
  τ-conf          = theorem2→ M τ x k k≢Uncertain ⟨p,⟩
  in cong₂ (_∧_) (cong proj₂ z-pred≡⟨x,true⟩) τ-conf
---- ############ Theorem 3 END ############
