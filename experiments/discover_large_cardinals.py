"""
Transfinite Set Theory, Forcing Extensions, and Large Cardinal Hierarchy
========================================================================

Exploring the mathematical universe beyond the Church-Kleene ordinal ω_1^CK:
1. Gödel's Constructible Universe L (The Minimal Inner Model)
2. Cohen's Method of Forcing (P ⊩ φ, generic filters G, independence of CH)
3. The Large Cardinal Ladder (Inaccessible -> Mahlo -> Measurable -> Woodin -> Supercompact -> Reinhardt)
4. Solovay's Theorem & Projective Determinacy (AD_L(R))
5. Executable Set-Theoretic Stage Simulator: Constructing V_α and L_α up to ω_1^CK
"""

import sys
import math
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# 1. The Large Cardinal Ladder (Hierarchy of Infinity)
# ---------------------------------------------------------------------------
@dataclass
class LargeCardinalAxiom:
    name: str
    symbol: str
    consistency_strength: str
    key_property: str
    metamathematical_impact: str


LARGE_CARDINAL_LADDER = [
    LargeCardinalAxiom(
        "ZFC (Base System)", "ZFC", "Base (Rank 0)",
        "Standard Zermelo-Fraenkel set theory with Axiom of Choice",
        "Cannot prove its own consistency (Gödel II: ZFC ⊬ Con(ZFC))."
    ),
    LargeCardinalAxiom(
        "Inaccessible Cardinal", "κ > ℵ_0, reg, strong limit", "Con(ZFC)",
        "V_κ is an explicit set model of ZFC: V_κ ⊨ ZFC",
        "Proves Con(ZFC). First genuine step past ZFC consistency."
    ),
    LargeCardinalAxiom(
        "Mahlo Cardinal", "Mahlo(κ)", "> Inaccessible",
        "The set of inaccessible cardinals below κ is stationary in κ",
        "Reflects set-theoretic properties onto unbounded subsets."
    ),
    LargeCardinalAxiom(
        "Measurable Cardinal", "∃ non-trivial κ-complete ultrafilter", "Ulam / Scott",
        "Witnessed by an elementary embedding j: V -> M with critical point κ",
        "Destroys V=L: Scott's Theorem (1961) proved 0^# exists; V ≠ L."
    ),
    LargeCardinalAxiom(
        "Strong Cardinal", "j: V -> M with V_{j(κ)} ⊂ M", "> Measurable",
        "Embeddings agree with the universe arbitrarily high up",
        "Essential for inner model program calibrations."
    ),
    LargeCardinalAxiom(
        "Woodin Cardinal", "∀A ⊂ V_λ, ∃κ < λ, j: V -> M", "Core of Modern Descriptive Set Theory",
        "Captures the exact strength of Projective Determinacy (PD)",
        "Infinitely many Woodin cardinals prove every projective set of reals is Lebesgue measurable!"
    ),
    LargeCardinalAxiom(
        "Supercompact Cardinal", "∀λ ≥ κ, ∃j: V -> M, M^λ ⊂ M", "Near the summit",
        "Huge reflection properties; implies strong forms of Singular Cardinal Hypothesis (SCH)",
        "Settles the behavior of cardinal arithmetic 2^λ across the entire universe."
    ),
    LargeCardinalAxiom(
        "Huge Cardinal", "j: V -> M with M^{j(κ)} ⊂ M", "Extreme reflection",
        "Critical point κ, first target j(κ) with strong closure",
        "Used to calibrate forcing axioms like Martin's Maximum (MM)."
    ),
    LargeCardinalAxiom(
        "Reinhardt Cardinal", "j: V -> V (Elementary embedding to itself)", "Kunen's Inconsistency Wall",
        "Embeds the entire set-theoretic universe V into V",
        "Inconsistent with ZFC by Kunen's Theorem (1971)! Only consistent in ZF without Choice."
    )
]


def display_large_cardinal_ladder():
    print("=" * 95)
    print("   THE HIERARCHY OF LARGE CARDINALS (THE SUMMIT OF MATHEMATICAL INFINITY)")
    print("=" * 95)
    print(f"{'Cardinal Axiom':<22} | {'Consistency Strength':<20} | {'Key Structural Consequence':<48}")
    print("-" * 95)
    for card in LARGE_CARDINAL_LADDER:
        print(f"{card.name:<22} | {card.consistency_strength:<20} | {card.metamathematical_impact[:48]:<48}")
    print("=" * 95)


# ---------------------------------------------------------------------------
# 2. Gödel's Constructible Universe L vs von Neumann's Cumulative Hierarchy V
# ---------------------------------------------------------------------------
class SetTheoreticUniverse:
    """
    Simulates the construction of von Neumann's Cumulative Hierarchy V_α
    and Gödel's Constructible Universe L_α for finite and low transfinite stages.
    
    V_0 = ∅,  V_{α+1} = P(V_α) (Full Powerset - wild, non-constructive)
    L_0 = ∅,  L_{α+1} = Def(L_α) (First-order definable subsets with parameters)
    """
    @staticmethod
    def powerset(s: frozenset) -> Set[frozenset]:
        """Computes full powerset P(s) for V_{α+1}"""
        elem_list = list(s)
        n = len(elem_list)
        subsets = set()
        for i in range(1 << n):
            sub = frozenset([elem_list[j] for j in range(n) if (i & (1 << j))])
            subsets.add(sub)
        return subsets

    @staticmethod
    def definable_subsets(s: frozenset) -> Set[frozenset]:
        """
        In finite stages L_n = V_n for n < ω (finite hereditarily finite sets).
        At ω and beyond, Def(L_α) is strictly countable if L_α is countable,
        whereas P(V_α) explodes to 2^|V_α| (Cantor's theorem).
        """
        # For finite ranks, all finite subsets are first-order definable
        return SetTheoreticUniverse.powerset(s)

    @classmethod
    def generate_ranks(cls, max_rank: int = 4):
        v_current = frozenset()
        l_current = frozenset()
        
        ranks = []
        for alpha in range(max_rank + 1):
            if alpha == 0:
                v_current = frozenset()
                l_current = frozenset()
            else:
                # Cumulative union + powerset/definable
                v_current = frozenset(cls.powerset(v_current))
                l_current = frozenset(cls.definable_subsets(l_current))
            
            ranks.append((alpha, len(v_current), len(l_current)))
        return ranks


# ---------------------------------------------------------------------------
# 3. Cohen's Forcing Engine (Poset P, Compatibility, and Dense Sets)
# ---------------------------------------------------------------------------
class ForcingPoset:
    """
    Cohen's Method of Forcing (1963):
    Constructs generic extensions V[G] of a ground model M by adjoining
    a generic filter G ⊂ P over a partially ordered set (P, ≤).
    Used to prove the independence of the Continuum Hypothesis (2^ℵ_0 = ℵ_2).
    """
    def __init__(self, name: str):
        self.name = name
        self.conditions: List[Dict[int, int]] = []  # Partial functions p: dom(p) ⊂ ω -> {0, 1}

    def add_condition(self, p: Dict[int, int]):
        self.conditions.append(p)

    @staticmethod
    def compatible(p: Dict[int, int], q: Dict[int, int]) -> Tuple[bool, Optional[Dict[int, int]]]:
        """
        p and q are compatible in Cohen poset iff they agree on common domain:
        ∀x ∈ dom(p) ∩ dom(q), p(x) == q(x).
        Their common refinement (lower bound) is r = p ∪ q.
        """
        for k in p:
            if k in q and p[k] != q[k]:
                return False, None
        # Union
        r = dict(p)
        r.update(q)
        return True, r

    def is_dense(self, predicate) -> bool:
        """
        A set D ⊂ P is dense if for every condition p ∈ P,
        there exists q ≤ p such that q ∈ D.
        """
        for p in self.conditions:
            found_refinement = False
            for q in self.conditions:
                comp, r = self.compatible(p, q)
                if comp and predicate(r):
                    found_refinement = True
                    break
            if not found_refinement:
                return False
        return True


# ---------------------------------------------------------------------------
# 4. Solovay's Model & Projective Determinacy (PD)
# ---------------------------------------------------------------------------
def explain_solovay_and_determinacy():
    print("\n" + "=" * 95)
    print("   SOLOVAY'S MODEL & PROJECTIVE DETERMINACY: THE RECONSTRUCTION OF ANALYSIS")
    print("=" * 95)
    print("""
1. The Axiom of Choice (AC) Paradox:
   Vitali (1905) proved using AC that non-Lebesgue measurable sets of real numbers EXIST.
   Banach-Tarski (1924) proved a solid unit ball can be partitioned into 5 disjoint pieces
   and reassembled into TWO identical unit balls!

2. Solovay's Breakthrough (1970):
   Assuming the consistency of an INACCESSIBLE CARDINAL, there exists a model of ZF + DC
   (Dependent Choice) where:
     - EVERY set of real numbers is LEBESGUE MEASURABLE!
     - The Baire Category Theorem holds for all sets.
     - The Banach-Tarski paradox is IMPOSSIBLE.
   Shelah (1984) proved the converse: An inaccessible cardinal is strictly NECESSARY.

3. Projective Determinacy (Martin, Steel, Woodin 1988):
   In two-player infinite games of length ω, say player I wins if the resulting real x ∈ A.
   If A is Borel, the game is determined (Martin 1975).
   What if A is Projective (analytic / co-analytic)?
   Axiom of Projective Determinacy (PD):
     Every projective game is determined!
   Woodin proved:
     ZF + ZFC + Infinitely Many Woodin Cardinals  ===>  PROJECTIVE DETERMINACY (PD)!
   Therefore, higher infinity (Woodin cardinals) completely resolves the topology
   and measure theory of the real line ℝ!
""")
    print("=" * 95)


def run_set_theoretic_simulations():
    print("=" * 95)
    print("   SIMULATING VON NEUMANN CUMULATIVE STAGES V_n vs GÖDEL CONSTRUCTIBLE L_n")
    print("=" * 95)
    ranks = SetTheoreticUniverse.generate_ranks(4)
    print(f"{'Rank α':<10} | {'|V_α| (Cumulative Hierarchy)':<32} | {'|L_α| (Definable Constructible)':<32}")
    print("-" * 95)
    for alpha, v_sz, l_sz in ranks:
        print(f"α = {alpha:<6} | {v_sz:<32} | {l_sz:<32}")
    print("\nTower Growth: |V_0| = 0, |V_1| = 1, |V_2| = 2, |V_3| = 4, |V_4| = 16, |V_5| = 65,536,")
    print("|V_6| = 2^65536 ≈ 10^19728 (Already exceeds atoms in the observable universe ~10^80!)")
    print("At α = ω: |V_ω| = ℵ_0 (The Hereditarily Finite Sets, isomorphic to Peano Arithmetic).")
    print("At α = ω+1: |V_{ω+1}| = 2^ℵ_0 (The Continuum / Real Numbers ℝ).")
    print("=" * 95)

    print("\nTesting Cohen Poset Compatibility & Extension...")
    poset = ForcingPoset("Cohen Real Poset Add(ω, 1)")
    p1 = {0: 1, 1: 0}
    p2 = {1: 0, 2: 1}
    p3 = {0: 0, 3: 1}
    
    comp12, r12 = ForcingPoset.compatible(p1, p2)
    comp13, r13 = ForcingPoset.compatible(p1, p3)
    
    print(f"Condition p1 = {p1}")
    print(f"Condition p2 = {p2} -> Compatible with p1? {comp12} | Join r12 = {r12}")
    print(f"Condition p3 = {p3} -> Compatible with p1? {comp13} (Conflict on bit 0: p1[0]=1 != p3[0]=0)")


if __name__ == "__main__":
    display_large_cardinal_ladder()
    run_set_theoretic_simulations()
    explain_solovay_and_determinacy()
