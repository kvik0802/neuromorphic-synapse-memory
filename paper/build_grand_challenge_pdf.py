"""
Generate The Grand Millennium Challenge Document as a Formal Academic PDF
==========================================================================
Compiles the complete, detailed specification of:
'The Holographic Cascade Problem: Spectral-Geometric Singularity & Complexity Barrier'
into a publication-grade PDF directly on the Desktop:
C:\\Users\\vicky\\Desktop\\The_Grand_Millennium_Challenge.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.pdfgen import canvas

desktop_pdf_path = Path("C:/Users/vicky/Desktop/The_Grand_Millennium_Challenge.pdf")


class ChallengeCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 8.5)
        self.setFillColor(colors.HexColor("#4a5568"))

        if self._pageNumber > 1:
            self.drawString(54, 750, "The Holographic Cascade Problem — An Open Mathematical Challenge")
            self.drawRightString(612 - 54, 750, "Open Problem Formulation")
            self.setStrokeColor(colors.HexColor("#cbd5e0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)

        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 36, page_str)
        self.drawString(54, 36, "Formulated by N. Vivekanand | Independent Foundational Study")
        self.setStrokeColor(colors.HexColor("#cbd5e0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 612 - 54, 46)

        self.restoreState()


doc = SimpleDocTemplate(
    str(desktop_pdf_path),
    pagesize=letter,
    leftMargin=54,
    rightMargin=54,
    topMargin=54,
    bottomMargin=54
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'ChallengeTitle',
    parent=styles['Normal'],
    fontName='Times-Bold',
    fontSize=18,
    leading=22,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#111827'),
    spaceAfter=8
)

subtitle_style = ParagraphStyle(
    'ChallengeSubtitle',
    parent=styles['Normal'],
    fontName='Times-BoldItalic',
    fontSize=11,
    leading=15,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#1e3a8a'),
    spaceAfter=12
)

author_style = ParagraphStyle(
    'ChallengeAuthor',
    parent=styles['Normal'],
    fontName='Times-Roman',
    fontSize=10,
    leading=14,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#374151'),
    spaceAfter=14
)

notice_style = ParagraphStyle(
    'ChallengeNotice',
    parent=styles['Normal'],
    fontName='Times-Bold',
    fontSize=9,
    leading=13,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#991b1b'),
    spaceAfter=14
)

h1_style = ParagraphStyle(
    'ChallengeH1',
    parent=styles['Normal'],
    fontName='Times-Bold',
    fontSize=12,
    leading=16,
    textColor=colors.HexColor('#111827'),
    spaceBefore=12,
    spaceAfter=6,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    'ChallengeH2',
    parent=styles['Normal'],
    fontName='Times-BoldItalic',
    fontSize=10.5,
    leading=14,
    textColor=colors.HexColor('#1e40af'),
    spaceBefore=8,
    spaceAfter=4,
    keepWithNext=True
)

body_style = ParagraphStyle(
    'ChallengeBody',
    parent=styles['Normal'],
    fontName='Times-Roman',
    fontSize=9.5,
    leading=13.5,
    alignment=TA_JUSTIFY,
    textColor=colors.HexColor('#1f2937'),
    spaceAfter=6
)

math_style = ParagraphStyle(
    'ChallengeMath',
    parent=styles['Normal'],
    fontName='Times-Italic',
    fontSize=9.5,
    leading=13,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#111827'),
    spaceBefore=4,
    spaceAfter=6
)

story = []

# Title Header
story.append(Paragraph("THE HOLOGRAPHIC CASCADE PROBLEM", title_style))
story.append(Paragraph("Spectral-Geometric Singularity, Gauge Mass Gap, and Computational Decidability", subtitle_style))
story.append(Paragraph("<b>Author & Architect: N. Vivekanand</b><br/>Independent Research Suite &nbsp;|&nbsp; November 2026", author_style))
story.append(Paragraph("<b>OFFICIAL CHALLENGE FORMULATION — STRICTLY UNSOLVED — NO KNOWN PROOF EXISTS</b>", notice_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1e3a8a'), spaceAfter=12))

# 1. Problem Overview
story.append(Paragraph("1. Fundamental Objective & Background", h1_style))
story.append(Paragraph(
    "The grand challenge of modern mathematical physics is the unification of discrete computation, "
    "algebraic arithmetic, continuous partial differential equations, and non-abelian quantum fields. "
    "Historically, major unsolved problems (the Millennium Prize problems) have been analyzed in isolation: "
    "fluid analysts examine the Sobolev scaling of Navier-Stokes, complexity theorists analyze circuit lower bounds for P vs NP, "
    "and quantum field theorists investigate the Yang-Mills mass gap. "
    "<b>The Holographic Cascade Problem</b> binds these disjoint domains into a single, closed, self-interacting mathematical system.",
    body_style
))

# 2. Formal System Setup
story.append(Paragraph("2. Mathematical Formulation of the System", h1_style))
story.append(Paragraph(
    "Let M be a compact, smooth 4-dimensional Riemannian spin manifold without boundary, endowed with metric g. "
    "On M, we introduce three coupled fields:",
    body_style
))
story.append(Paragraph(
    "1. <b>Non-Abelian Gauge Connection A:</b> An SU(N) Yang-Mills gauge field with field strength tensor "
    "F_mu,nu = d_mu A_nu - d_nu A_mu + g_YM [A_mu, A_nu], governed by the Yang-Mills action S_YM = (1/4) int_M Tr(F /\\ *F).<br/>"
    "2. <b>Incompressible Navier-Stokes Flow u:</b> A tangential velocity field u in T(M) satisfying divergence-free incompressibility "
    "div(u) = 0 and Navier-Stokes momentum: du/dt + (u . grad)u = -grad p + nu Delta u + F_ext.<br/>"
    "3. <b>Arithmetic L-Function Manifold Moduli:</b> The metric g is algebraically parameterized by the central values "
    "and order of vanishing of Hasse-Weil L-functions L(s, V) originating from an algebraic variety V over Q, "
    "incorporating the Birch and Swinnerton-Dyer / Hodge cycle invariants.",
    body_style
))
story.append(Paragraph("du/dt + (u . grad)u = -grad p + nu Delta u + g_YM * Tr(F /\\ *F) * u &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1)", math_style))
story.append(Paragraph("d *F + [A, *F] = J(u), &nbsp;&nbsp;&nbsp;&nbsp; div(u) = 0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2)", math_style))

# 3. The Core Challenge Questions
story.append(Paragraph("3. The Three Interlocked Challenge Questions", h1_style))

story.append(Paragraph("Question A: The Singularity vs. Mass Gap Resolution (Fluid Dynamics & QFT)", h2_style))
story.append(Paragraph(
    "Does there exist a smooth initial datum (M_0, A_0, u_0) with finite kinetic energy E(0) &lt; infty "
    "such that non-linear 3D vortex stretching (u . grad)u drives the local enstrophy Omega(t) = (1/2)||grad u||_{L^2}^2 "
    "to blow up to infinity in finite time T* &lt; infty (Navier-Stokes singularity), "
    "<b>OR</b> does the non-abelian Yang-Mills vacuum dynamically generate a strictly positive spectral mass gap Delta &gt; 0 "
    "that universally bounds the enstrophy growth, forcing all solutions to remain infinitely smooth for all t in [0, infty)?",
    body_style
))

story.append(Paragraph("Question B: The Holographic Computational Barrier (Complexity & Gödel Incompleteness)", h2_style))
story.append(Paragraph(
    "Let an observer measure the asymptotic radiation spectrum of M to decode the termination state of a universal Turing machine "
    "whose tape is initialized by the parity structure of expander graphs. "
    "Can the asymptotic ground-state energy E_0 of this coupled quantum-fluid system be determined by any deterministic "
    "or randomized algorithm in polynomial time O(n^k) (P = BPP)? "
    "Or is deciding the existence of a ground-state gap Delta &gt; 0 <b>strictly undecidable and independent of the ZFC axioms</b>, "
    "proving that physical quantum geometry computes beyond the Arithmetical Hierarchy?",
    body_style
))

story.append(Paragraph("Question C: The Arithmetic-Topological Correspondence (Number Theory & Geometry)", h2_style))
story.append(Paragraph(
    "Prove or disprove: The spectral zeros of the coupled fluid-gauge operator on M coincide identically with the critical zeros "
    "of the Riemann zeta function and Dirichlet L-functions, such that a finite-time blowup occurs if and only if "
    "there exists a non-trivial counterexample to Goldbach's Conjecture or an odd perfect number.",
    body_style
))

# 4. Summary Table of Embedded Barriers
story.append(Paragraph("4. Embedded Mathematical & Physical Barriers", h1_style))
story.append(Paragraph(
    "Any attempted proof or disproof must simultaneously circumvent four established impossibility barriers:",
    body_style
))

barrier_data = [
    [Paragraph("<b>Domain</b>", body_style), Paragraph("<b>Embedded Problem</b>", body_style), Paragraph("<b>Theoretical Impossibility Barrier</b>", body_style)],
    ["Fluid Dynamics", "Navier-Stokes Smoothness", "Supercritical Sobolev Gap (2/p + 3/q = 1.5 > 1.0)"],
    ["Quantum Fields", "Yang-Mills Mass Gap", "Wightman Axiom Non-Perturbative Confinement"],
    ["Complexity", "P vs NP / Derandomization", "Relativization, Natural Proofs, & Algebrization"],
    ["Mathematical Logic", "Universal Ground State", "Gödel Incompleteness & Halting Problem (0')"]
]

barr_table = Table(barrier_data, colWidths=[110, 160, 234])
barr_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
    ('FONTSIZE', (0, 0), (-1, -1), 8.5),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
]))
story.append(KeepTogether([barr_table]))
story.append(Spacer(1, 10))

# 5. Formal Challenge Verdict
story.append(Paragraph("5. Challenge Verdict & Rules of Engagement", h1_style))
story.append(Paragraph(
    "• <b>No Approximate / Numerical Solutions:</b> Discrete simulations (grid approximations, PINNs, neural operators) "
    "cannot settle this challenge due to discretization cutoff and spectral bias.<br/>"
    "• <b>Formal Machine Verification:</b> A complete resolution must be formalized in an interactive theorem prover "
    "(such as Lean 4, Coq, or Isabelle) verified down to foundational type-theoretic axioms.<br/>"
    "• <b>Current Global Status:</b> STRICTLY UNSOLVED. No proof, partial solution, or counterexample exists.",
    body_style
))

doc.build(story, canvasmaker=ChallengeCanvas)
print(f"Grand Millennium Challenge PDF generated: {desktop_pdf_path}")
