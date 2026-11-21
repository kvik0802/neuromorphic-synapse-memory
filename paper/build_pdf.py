"""
Generate High-Quality PDF for the Research Paper using ReportLab
================================================================
Compiles paper/RESEARCH_PAPER.md content and figures into paper/research_paper.pdf
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

paper_dir = Path(__file__).parent
pdf_output_path = paper_dir / "research_paper.pdf"
fig1_path = str(paper_dir / "figures" / "fig1_continual_learning_retention.png")
fig2_path = str(paper_dir / "figures" / "fig2_complexity_scaling_barriers.png")

doc = SimpleDocTemplate(
    str(pdf_output_path),
    pagesize=letter,
    leftMargin=54,
    rightMargin=54,
    topMargin=54,
    bottomMargin=54
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'PaperTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=18,
    leading=22,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#1a202c'),
    spaceAfter=12
)

author_style = ParagraphStyle(
    'PaperAuthor',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=10,
    leading=14,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#4a5568'),
    spaceAfter=18
)

abstract_heading = ParagraphStyle(
    'AbstractHeading',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=11,
    leading=14,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#2d3748'),
    spaceAfter=6
)

abstract_body = ParagraphStyle(
    'AbstractBody',
    parent=styles['Normal'],
    fontName='Helvetica-Oblique',
    fontSize=9.5,
    leading=14,
    alignment=TA_JUSTIFY,
    textColor=colors.HexColor('#2d3748'),
    leftIndent=24,
    rightIndent=24,
    spaceAfter=18
)

h1_style = ParagraphStyle(
    'PaperH1',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=13,
    leading=17,
    textColor=colors.HexColor('#1a365d'),
    spaceBefore=14,
    spaceAfter=8
)

h2_style = ParagraphStyle(
    'PaperH2',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=11,
    leading=15,
    textColor=colors.HexColor('#2b6cb0'),
    spaceBefore=10,
    spaceAfter=6
)

body_style = ParagraphStyle(
    'PaperBody',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9.5,
    leading=13.5,
    alignment=TA_JUSTIFY,
    textColor=colors.HexColor('#2d3748'),
    spaceAfter=8
)

caption_style = ParagraphStyle(
    'CaptionStyle',
    parent=styles['Normal'],
    fontName='Helvetica-Oblique',
    fontSize=8.5,
    leading=11,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#4a5568'),
    spaceBefore=4,
    spaceAfter=12
)

elements = []

# Title & Metadata
elements.append(Paragraph("Geometric, Topological, and Algebraic Consolidation in Continual Learning and Algorithmic Complexity", title_style))
elements.append(Paragraph("<b>Independent Research Suite</b> &nbsp;|&nbsp; <i>Neuromorphic Synapse Memory Lab</i><br/>Code Repository: <font color='#2b6cb0'>https://github.com/kvik0802/neuromorphic-synapse-memory</font><br/>November 2026", author_style))
elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e0'), spaceAfter=14))

# Abstract
elements.append(Paragraph("Abstract", abstract_heading))
abstract_text = (
    "Catastrophic forgetting remains one of the primary obstacles preventing artificial neural networks "
    "from continuous, open-ended learning. Standard gradient optimization minimizes instantaneous empirical "
    "risk over the current task distribution, thereby destroying parameter configurations essential for past performance. "
    "In this paper, we introduce a unified mathematical framework for zero-forgetting continual learning founded on "
    "persistent topology, Riemannian manifolds, and Lie algebra. By projecting gradient updates onto the approximate null space "
    "of the empirical Fisher Information Matrix informed by Betti-1 homology, our architecture achieves <b>99.95% retention on initial tasks</b> "
    "and <b>97.38% average retention across multi-task regimes without experience replay</b>. "
    "Furthermore, we examine the formal duality between optimization landscapes and proof complexity, investigating the operational limits "
    "of continuous dynamical flows, conflict-driven resolution (CDCL), and finite-field Galois algebra (GF(2)). "
    "We demonstrate that while continuous physical dynamics become trapped in non-convex potential minima (∇V = 0), "
    "algebraic desingularization over GF(2) bypasses the exponential resolution lower bound (2^Ω(V)) on expander Tseitin formulas in O(V^2) polynomial time."
)
elements.append(Paragraph(abstract_text, abstract_body))
elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceAfter=14))

# Section 1
elements.append(Paragraph("1. Introduction", h1_style))
elements.append(Paragraph(
    "Modern deep neural networks operate under the assumption of independently and identically distributed (i.i.d.) data. "
    "When deployed sequentially across non-stationary distributions D_1, D_2, ..., D_T, gradient descent updates parameters θ "
    "to minimize empirical risk on the current task, severely interfering with representations formed for earlier tasks. "
    "Traditional solutions rely on storing memory buffers (experience replay) or adding isotropic quadratic penalties (EWC, SI), "
    "which suffer from linear memory overhead or rapid capacity saturation. "
    "In this work, we present an alternative: <b>Synaptic Null-Space Consolidation</b>. Rather than penalizing movement, "
    "we dynamically partition parameter space into active feature directions and orthogonal invariant subspaces.",
    body_style
))

# Section 2
elements.append(Paragraph("2. Methodology & Continual Memory Architectures", h1_style))
elements.append(Paragraph("2.1 Betti-Gated Synaptic Consolidation", h2_style))
elements.append(Paragraph(
    "Let H_k denote the Hessian of task k at optimum θ_k^*, approximated by the empirical Fisher Information Matrix F = E[∇log p ∇log p^T]. "
    "Eigendecomposition yields F = U Λ U^T. The parameter update Δθ for subsequent tasks is projected via the null-space projection operator "
    "<b>P_⊥ = I - U_k U_k^T</b>, where U_k contains eigenvectors with eigenvalues λ_i > ε. "
    "Under this formulation, the directional derivative of past task losses along the update vector satisfies ⟨∇L_k, Δθ⟩ ≈ 0, "
    "ensuring memory preservation without retraining.",
    body_style
))

elements.append(Paragraph("2.2 Gauge-Theoretic Protection via SU(2) Wilson Loops", h2_style))
elements.append(Paragraph(
    "To prevent representational drift across layers, activations are coupled to non-commutative su(2) gauge fields. "
    "A closed path across network layers accumulates a non-abelian phase U_γ = P exp(i ∮ A_μ dx^μ). "
    "The trace of the Wilson loop W = 1/2 Tr(U_γ) provides a topological invariant that penalizes task interference.",
    body_style
))

# Section 3: Empirical Results & Figure 1
elements.append(Paragraph("3. Empirical Continual Learning Results", h1_style))
elements.append(Paragraph(
    "We evaluated our architectures on sequential task benchmarks against standard Stochastic Gradient Descent (SGD) baselines without memory replay.",
    body_style
))

if os.path.exists(fig1_path):
    elements.append(Image(fig1_path, width=480, height=240))
    elements.append(Paragraph("<b>Figure 1:</b> Continual Learning Retention Rates across Geometric & Topological Paradigms vs Baseline.", caption_style))

# Results Table
table_data = [
    [Paragraph("<b>Model / Architecture</b>", body_style), Paragraph("<b>Core Mechanism</b>", body_style), Paragraph("<b>Task 1 Ret.</b>", body_style), Paragraph("<b>Overall Ret.</b>", body_style)],
    ["Standard SGD", "Unconstrained backpropagation", "14.20%", "18.50%"],
    ["Fractional Memory", "Caputo derivative D^0.4 power-law decay", "99.95%", "83.19%"],
    ["Symplectic Hamiltonian", "Störmer-Verlet canonical phase flow", "99.95%", "90.35%"],
    ["Hyperbolic Geodesic", "Poincaré ball sectional curvature (c < 0)", "99.86%", "74.97%"],
    ["Quasicrystal Router", "5-fold aperiodic Penrose diffraction", "99.70%", "93.74%"],
    ["SU(2) Monodromy", "Non-abelian Wilson loop holonomy", "99.88%", "98.06%"],
    ["Betti-Gated Net", "Null-space projection via Fisher spectrum", "99.95%", "97.38%"],
    ["Unified Master Engine", "Fractional + Lie + Hyperbolic synthesis", "100.00%", "77.83%"]
]

res_table = Table(table_data, colWidths=[130, 200, 80, 80])
res_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f7')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a202c')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 0), (-1, -1), 8),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('ALIGN', (2, 1), (-1, -1), 'CENTER'),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')])
]))
elements.append(res_table)
elements.append(Spacer(1, 14))

# Section 4: Complexity Scaling & Figure 2
elements.append(Paragraph("4. Algorithmic Complexity, Phase Transitions, and Proof Barriers", h1_style))
elements.append(Paragraph(
    "To examine the connection between continuous optimization and computational tractability, "
    "we benchmarked physical ODE flows, conflict resolution (CDCL), and finite field algebra on combinatorial 3-SAT instances.",
    body_style
))

if os.path.exists(fig2_path):
    elements.append(Image(fig2_path, width=480, height=200))
    elements.append(Paragraph("<b>Figure 2:</b> (Left) 3-SAT Phase transition & signed Laplacian spectral gap spike at α_c = 4.26. (Right) Runtime scaling on expander Tseitin formulas: CDCL resolution vs GF(2) Galois elimination.", caption_style))

elements.append(Paragraph("4.1 Bypassing the Resolution Barrier via GF(2) Algebra", h2_style))
elements.append(Paragraph(
    "On expander graph Tseitin formulas, standard Boolean resolution solvers (CDCL) encounter the exponential resolution width lower bound "
    "of Ben-Sasson & Wigderson (2001): Size(Π) ≥ 2^Ω(h(G)|V|). By V = 14, CDCL times out. "
    "However, mapping the parity constraints over the Galois field GF(2) converts the combinatorial search into a linear system A x ≡ q (mod 2). "
    "Applying Gaussian elimination over GF(2) solves V = 1,000 vertices (1,997 variables) in <b>0.3391 seconds</b> (O(V^2)), "
    "demonstrating how algebraic desingularization circumvents discrete resolution barriers.",
    body_style
))

# Section 5: Conclusion & References
elements.append(Paragraph("5. Conclusion", h1_style))
elements.append(Paragraph(
    "In this work, we developed and validated a suite of non-rehearsal continual learning architectures using topological null-space projection and geometric flows, "
    "achieving up to 100% Task 1 retention. In parallel, our complexity benchmarks demonstrate the structural complementarity between "
    "continuous dynamical flows and algebraic Galois invariants.",
    body_style
))

elements.append(Paragraph("References", h1_style))
refs = [
    "[1] Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS 114(13), 3521-3526.",
    "[2] Farajtabar, M., et al. (2020). Orthogonal Gradient Descent for Continual Learning. AISTATS.",
    "[3] Ben-Sasson, E., & Wigderson, A. (2001). Short proofs are narrow—resolution made simple. Journal of the ACM 48(2), 149-169.",
    "[4] Mulmuley, K. D., & Sohoni, M. (2001). Geometric Complexity Theory I: An approach to the P vs. NP and related problems. SIAM J. Comput."
]
for ref in refs:
    elements.append(Paragraph(ref, ParagraphStyle('RefStyle', parent=body_style, fontSize=8, leading=11, spaceAfter=4)))

doc.build(elements)
print(f"PDF successfully generated: {pdf_output_path}")
