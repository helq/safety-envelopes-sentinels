module Avionics.SafetyEnvelopes.Properties where

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List as List using (List; []; _∷_)
open import Data.List.Relation.Unary.Any as Any using (Any; here; there; satisfied)
open import Data.Maybe using (Maybe; just; nothing; is-just; _>>=_)
open import Data.Product using (∃-syntax; _×_; proj₁; proj₂; map) renaming (_,_ to ⟨_,_⟩)
open import Function using (_∘_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; cong; inspect; [_]; sym; trans)
open import Relation.Nullary using (Dec; yes; no)
open import Relation.Nullary.Decidable using (toWitness)
open import Relation.Unary using (_∈_)

open import Avionics.Bool using (≡→T; T∧→×)
open import Avionics.List using (any-val)
open import Avionics.Real
    using (ℝ; _+_; _-_; _*_; _÷_; _^_; _<ᵇ_; _≤ᵇ_; _≤_; _<_; _<?_; _≢0;
           0ℝ; 1ℝ; 2ℝ; _^2; √_; fromℕ;
           ⟨0,∞⟩; [0,∞⟩;
           <-transˡ; 2>0; ⟨0,∞⟩→0<; 0<→⟨0,∞⟩; >0→≢0; >0→≥0; q>0→√q>0;
           0≟0≡yes0≡0)
open import Avionics.Probability using (NormalDist; Dist)
open import Avionics.SafetyEnvelopes
    using (inside; z-predictable'; P[_|X=_]_; classify''; classify; M→pbs;
           StallClasses; Stall; NoStall;
           safety-envelope; z-predictable; Model; τ-confident;
           Stall≡1-NoStall
           )

open NormalDist using (σ; μ)

--<ᵇ→< : ∀ {x y} → T (x <ᵇ y) → x < y
--<ᵇ→< = toWitness

-- `pi` is the prediction interval for the z score, i.e.,
-- pi(N (μ, σ), z) = [μ − zσ, μ + zσ]
pi : NormalDist → ℝ → ℝ → Set
pi nd z x = (μ nd) - z * (σ nd) < x  ×  x < (μ nd) + z * (σ nd)

-- Proof that implementation follows from definition (Definition 2)
--
-- In words, the Property 1 says that:
--    The energy signal x is z-predictable iff there exist ⟨α, v⟩ s.t.
--    M(⟨α, v⟩)1 = di and x ∈ pi(di , z).
--
-- Notice that `Any (λ nd → x ∈ pi nd z) nds` translates to:
-- there exists nd such that `nd ∈ nds` and `x ∈ pi(nd, z)`
prop1 : ∀ (nds z x)
      → z-predictable' nds z x ≡ ⟨ x , true ⟩
      → Any (λ nd → x ∈ pi nd z) nds
prop1 nds z x res≡x,true = Any-x∈pi
  where
    res≡true = cong proj₂ res≡x,true

    -- the first `toWitness` takes a result `(μ nd - z * σ nd) <ᵇ x` (a
    -- boolean) and produces a proof of the type `(μ nd) - z * (σ nd) < x`
    -- assuming we have provided an operator `<?`
    toWitness' = λ nd → map (toWitness {Q = (μ nd - z * σ nd) <? x})
                            (toWitness {Q = x <? (μ nd + z * σ nd)})

    -- We find the value for which `inside z x` becomes true in the list `nds`
    Any-bool = any-val (inside z x) nds res≡true
    -- Converting the boolean proof into a proof at the type level
    Any-x∈pi = Any.map (λ {nd} → toWitness' nd ∘ T∧→×) Any-bool

-- From the prove above we can obtain the value `nd` and its prove `x ∈ pi nd z`
-- Note: An element of ∃[ nd ] (x ∈ pi nd z) is a tuple of the form ⟨ nd , proof ⟩
prop1' : ∀ (nds z x)
       → z-predictable' nds z x ≡ ⟨ x , true ⟩
       → ∃[ nd ] (x ∈ pi nd z)
prop1' nds z x res≡x,true = satisfied (prop1 nds z x res≡x,true)

extractDists : Model → List NormalDist
extractDists M = List.map (proj₁ ∘ proj₂) (Model.fM M)

-- ############ PROP 1 ############
prop1M : ∀ (M z x)
       → z-predictable M z x ≡ ⟨ x , true ⟩
       → Any (λ nd → x ∈ pi nd z) (extractDists M)
prop1M M z x res≡x,true = prop1 (extractDists M) z x res≡x,true
-- ############ PROP 1 END ############

--postulate
--  P'[_❘X=_]_ : StallClasses → ℝ → Model → ℝ

lem← : ∀ (pbs τ x k)
     → classify'' pbs τ x ≡ just k
     → ∃[ p ] (((P[ k |X= x ] pbs) ≡ just p) × (τ < p))
lem← pbs τ x k _ with P[ Stall |X= x ] pbs | inspect (P[ Stall |X=_] pbs) x
lem← _ τ _ _       _ | just p | [ _ ] with τ <? p | τ <? (1ℝ - p)
lem← _ _ _ Stall   _ | just p | [ P[k|X=x]≡justp ] | yes τ<p | no ¬τ<1-p = ⟨ p , ⟨ P[k|X=x]≡justp , τ<p ⟩ ⟩
lem← _ _ _ NoStall _ | just p | [ P[k|X=x]≡justp ] | no ¬τ<p | yes τ<1-p =
  let P[NoStall|X=x]≡just1-p = Stall≡1-NoStall P[k|X=x]≡justp
  in ⟨ 1ℝ - p , ⟨ P[NoStall|X=x]≡just1-p , τ<1-p ⟩ ⟩

lem→ : ∀ (pbs τ x k)
     → ∃[ p ] (((P[ k |X= x ] pbs) ≡ just p) × (τ < p))
     → classify'' pbs τ x ≡ just k
lem→ M τ x k ⟨ p , ⟨ P[k|X=x]M , τ<p ⟩ ⟩ = ?
--lem→ pbs _ x Stall _ with P[ Stall |X= x ] pbs
--lem→ _ τ _ _ _ | just p with τ <? p | τ <? (1ℝ - p)
--lem→ _ _ _ _ _ | just p | yes _ | no  _ = ?
--lem→ _ _ _ _ _ | just p | no  _ | yes _ = ?
--lem→ _ _ _ _ _ | just p | _     | _ = ?
--lem→ _ _ _ _ _ | nothing = ?

prop2M-prior→ : ∀ (M τ x k)
              → ∃[ p ] (((P[ k |X= x ] (M→pbs M)) ≡ just p) × (τ < p))
              → classify M τ x ≡ just k
prop2M-prior→ M = lem→ (M→pbs M)

prop2M-prior← : ∀ (M τ x k)
              → classify M τ x ≡ just k
              → ∃[ p ] (((P[ k |X= x ] (M→pbs M)) ≡ just p) × (τ < p))
prop2M-prior← M = lem← (M→pbs M)

prop2M→ : ∀ (M τ x k)
        → classify M τ x ≡ just k
        → τ-confident M τ x ≡ true
prop2M→ M τ x k cMτx≡k = cong is-just cMτx≡k

-- Not extrictly true, it requires an existential quantification
prop2M← : ∀ (M τ x k)
       → τ-confident M τ x ≡ true
       → classify M τ x ≡ just k
prop2M← M τ x k τconf≡true = ?

-- ############ PROP 2 ############
prop2M'→ : ∀ (M τ x k)
         → ∃[ p ] (((P[ k |X= x ] (M→pbs M)) ≡ just p) × (τ < p))
         → τ-confident M τ x ≡ true
prop2M'→ M τ x k ⟨p,⟩ = prop2M→ M τ x k (prop2M-prior→ M τ x k ⟨p,⟩)

prop2M'← : ∀ (M τ x k)
         → τ-confident M τ x ≡ true
         → ∃[ p ] (((P[ k |X= x ] (M→pbs M)) ≡ just p) × (τ < p))
prop2M'← M τ x k τconf≡true = prop2M-prior← M τ x k (prop2M← M τ x k τconf≡true)
-- ############ PROP 2 END ############

---- ############ PROP 3 ############
prop3M← : ∀ (M z τ x)
        → safety-envelope M z τ x ≡ true
        → (Any (λ nd → x ∈ pi nd z) (extractDists M))
          × ∃[ k ] (classify M τ x ≡ just k  ×  ∃[ p ] (((P[ k |X= x ] (M→pbs M)) ≡ just p) × (τ < p)))
prop3M← M z τ x seM≡true = ? -- ⟨ prop1M M z x ? , ⟨ ? , ⟨ ? , prop2M'→ M τ x ? ? ⟩ ⟩ ⟩

--prop3M'← : ∀ (M z τ x)
--        → safety-envelope M z τ x ≡ true
--        → z-predictable M z x ≡ ⟨ x , true ⟩  ×  τ-confident M τ x ≡ true
--prop3M'← = ?
---- ############ PROP 3 END ############
