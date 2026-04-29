// ═══════════════════════════════════════════════════════════════════════════
// Task 2: 3D Brain Tumour Segmentation — Typst Report
// CN6021 — Advanced Topics in AI and Data Science
// ═══════════════════════════════════════════════════════════════════════════

// ── Design System ────────────────────────────────────────────────────────
#let accent     = rgb("#6C63FF")    // vibrant indigo
#let accent2    = rgb("#00D2FF")    // cyan highlight
#let dark       = rgb("#1a1a2e")    // near-black
#let mid        = rgb("#16213e")    // dark navy
#let light-bg   = rgb("#f8f9fc")    // off-white
#let text-gray  = rgb("#555566")
#let success    = rgb("#10b981")
#let warning    = rgb("#f59e0b")
#let danger     = rgb("#ef4444")

// ── Page Setup ───────────────────────────────────────────────────────────
#set page(
  paper: "a4",
  margin: (top: 28mm, bottom: 28mm, left: 25mm, right: 25mm),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(8pt, fill: text-gray)
      #grid(
        columns: (1fr, 1fr),
        align: (left, right),
        [CN6021 — Advanced Topics in AI & Data Science],
        [Task 2: 3D Brain Tumour Segmentation],
      )
      #line(length: 100%, stroke: 0.4pt + luma(200))
    ]
  },
  footer: context {
    if counter(page).get().first() > 1 [
      #line(length: 100%, stroke: 0.4pt + luma(200))
      #v(2pt)
      #set text(8pt, fill: text-gray)
      #grid(
        columns: (1fr, 1fr),
        align: (left, right),
        [Group Coursework],
        [Page #counter(page).display("1 / 1", both: true)],
      )
    ]
  },
)

#set text(font: "New Computer Modern", size: 10.5pt, fill: dark)
#set par(justify: true, leading: 0.7em)
#set heading(numbering: "1.1.")

#show heading.where(level: 1): it => {
  v(1em)
  block(width: 100%)[
    #rect(
      width: 100%,
      fill: gradient.linear(accent, accent2, angle: 0deg),
      radius: 4pt,
      inset: (x: 14pt, y: 10pt),
    )[
      #text(font: "New Computer Modern", weight: "bold", size: 14pt, fill: white)[
        #counter(heading).display() #it.body
      ]
    ]
  ]
  v(0.5em)
}

#show heading.where(level: 2): it => {
  v(0.8em)
  block[
    #text(weight: "bold", size: 12pt, fill: accent)[
      #box(width: 4pt, height: 12pt, fill: accent, radius: 1pt)
      #h(6pt)
      #counter(heading).display() #it.body
    ]
  ]
  v(0.3em)
}

#show heading.where(level: 3): it => {
  v(0.5em)
  text(weight: "bold", size: 10.5pt, fill: mid)[
    #counter(heading).display() #it.body
  ]
  v(0.2em)
}

// ── Styled components ────────────────────────────────────────────────────
#let info-box(title: "", body) = {
  rect(
    width: 100%,
    fill: rgb("#eef2ff"),
    stroke: (left: 3pt + accent),
    radius: (right: 4pt),
    inset: 12pt,
  )[
    #if title != "" [
      #text(weight: "bold", fill: accent, size: 9.5pt)[#title]
      #v(4pt)
    ]
    #set text(size: 9.5pt)
    #body
  ]
}

#let metric-card(label, value, unit: "", color: accent) = {
  rect(
    width: 100%,
    fill: white,
    stroke: 0.8pt + luma(220),
    radius: 6pt,
    inset: 10pt,
  )[
    #align(center)[
      #text(size: 8pt, fill: text-gray, weight: "bold")[#upper(label)]
      #v(2pt)
      #text(size: 20pt, fill: color, weight: "bold")[#value]
      #if unit != "" [
        #text(size: 8pt, fill: text-gray)[ #unit]
      ]
    ]
  ]
}


// ═══════════════════════════════════════════════════════════════════════════
//  TITLE PAGE
// ═══════════════════════════════════════════════════════════════════════════
#page(margin: 0pt, header: none, footer: none)[
  // Gradient background
  #rect(width: 100%, height: 100%, fill: gradient.linear(dark, mid, angle: 180deg))[
    #v(1fr)
    #align(center)[
      // Accent line
      #rect(width: 60%, height: 3pt, fill: gradient.linear(accent, accent2), radius: 2pt)
      #v(20pt)

      #text(font: "New Computer Modern", size: 11pt, fill: accent2, weight: "bold", tracking: 3pt)[
        CN6021 — ADVANCED TOPICS IN AI & DATA SCIENCE
      ]
      #v(14pt)

      #text(font: "New Computer Modern", size: 28pt, fill: white, weight: "bold")[
        3D Brain Tumour Segmentation
      ]
      #v(6pt)
      #text(font: "New Computer Modern", size: 16pt, fill: luma(180))[
        Using Deep Learning with PyTorch
      ]

      #v(20pt)
      #rect(width: 40%, height: 1pt, fill: luma(80))
      #v(16pt)

      #text(size: 11pt, fill: luma(160))[
        Group Coursework Submission
      ]
      #v(6pt)
      #text(size: 10pt, fill: luma(120))[
        // Replace with your group name
        Group \[Your Group Name\] #h(10pt) | #h(10pt) May 2026
      ]

      #v(30pt)

      // Key metrics preview
      #grid(
        columns: (1fr, 1fr, 1fr),
        column-gutter: 12pt,
        rect(fill: rgb("#ffffff10"), radius: 8pt, inset: 14pt, stroke: 0.5pt + rgb("#ffffff20"))[
          #align(center)[
            #text(size: 8pt, fill: accent2, weight: "bold")[MEAN DICE SCORE]
            #v(4pt)
            #text(size: 26pt, fill: white, weight: "bold")[0.775]
          ]
        ],
        rect(fill: rgb("#ffffff10"), radius: 8pt, inset: 14pt, stroke: 0.5pt + rgb("#ffffff20"))[
          #align(center)[
            #text(size: 8pt, fill: accent2, weight: "bold")[PARAMETERS]
            #v(4pt)
            #text(size: 26pt, fill: white, weight: "bold")[6.05M]
          ]
        ],
        rect(fill: rgb("#ffffff10"), radius: 8pt, inset: 14pt, stroke: 0.5pt + rgb("#ffffff20"))[
          #align(center)[
            #text(size: 8pt, fill: accent2, weight: "bold")[PATIENTS]
            #v(4pt)
            #text(size: 26pt, fill: white, weight: "bold")[600]
          ]
        ],
      )
    ]
    #v(1fr)

    // Bottom accent
    #rect(width: 100%, height: 4pt, fill: gradient.linear(accent, accent2))
  ]
]

// ── Table of Contents ────────────────────────────────────────────────────
#page[
  #v(10pt)
  #text(size: 18pt, weight: "bold", fill: accent)[Table of Contents]
  #v(6pt)
  #line(length: 100%, stroke: 1pt + accent)
  #v(10pt)
  #outline(indent: 1.5em, depth: 3)
]


// ═══════════════════════════════════════════════════════════════════════════
= Introduction

Brain tumour segmentation from MRI scans is a critical task in clinical neuro-oncology, enabling precise treatment planning, surgical guidance, and longitudinal monitoring of disease progression. Manual segmentation by radiologists is time-consuming, subjective, and prone to inter-observer variability, motivating the development of automated deep learning approaches.

This report presents a *3D convolutional neural network (CNN)* pipeline for multi-class semantic segmentation of brain tumours from volumetric MRI data. The system is built entirely in *PyTorch* and addresses several real-world challenges:

+ *3D Volumetric Data*: MRI scans are 3D volumes (typically 240×240×155 voxels across four modalities), requiring specialised architectures and memory-efficient training strategies.
+ *Varied Tumour Morphology*: Gliomas exhibit extreme variability in size, shape, and spatial location.
+ *Severe Class Imbalance*: Tumour regions constitute less than 2% of the total brain volume.
+ *Limited Computational Resources*: Training on a laptop GPU (6 GB VRAM) demands patch-based sampling and mixed-precision training.

#info-box(title: "Dataset")[
  We utilise the *BraTS 2024 Adult Glioma (GLI)* dataset (Synapse ID: `syn59059776`), the gold-standard benchmark for brain tumour segmentation. It provides multi-parametric MRI scans with expert-annotated segmentation masks delineating three tumour sub-regions: *Necrotic Core (NCR)*, *Peritumoral Oedema (ED)*, and *Enhancing Tumour (ET)*.
]


// ═══════════════════════════════════════════════════════════════════════════
= Methodology

== Dataset and Exploratory Data Analysis

=== Dataset Description

The BraTS 2024 GLI dataset contains *1,809 patient cases*, each comprising four co-registered MRI modalities. We selected the two most clinically informative:

- *T2-FLAIR (t2f)*: Highlights peritumoral oedema and non-enhancing tumour regions.
- *Post-contrast T1 (t1c)*: Delineates enhancing tumour boundary via gadolinium contrast uptake.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (center, left, left),
    stroke: none,
    table.hline(stroke: 1.5pt + accent),
    table.header(
      [*Label*], [*Region*], [*Clinical Significance*],
    ),
    table.hline(stroke: 0.5pt + luma(200)),
    [0], [Background], [Healthy brain tissue],
    [1], [Necrotic Core (NCR)], [Dead tissue at tumour centre],
    [2], [Peritumoral Oedema (ED)], [Swelling surrounding the tumour],
    [3/4], [Enhancing Tumour (ET)], [Actively growing tumour],
    table.hline(stroke: 1.5pt + accent),
  ),
  caption: [BraTS 2024 segmentation labels. Label 4 is remapped to class 3 for contiguous indexing.],
) <tbl-labels>

#info-box(title: "Critical Fix: Label Remapping")[
  The original BraTS labels use value 4 for Enhancing Tumour. Without explicit remapping (`4 → 3`), the ET class would be silently dropped during training, producing 0.0 Dice for this clinically critical region. This was identified and fixed during our pipeline development.
]

=== Exploratory Analysis

#grid(
  columns: (1fr, 1fr),
  column-gutter: 10pt,
  figure(
    image("outputs/results/eda_class_distribution.png", width: 100%),
    caption: [Class distribution — background dominates at >98%.],
  ),
  figure(
    image("outputs/results/eda_sample_slice.png", width: 100%),
    caption: [Sample axial slice with ground-truth overlay.],
  ),
)

The class distribution confirms *extreme imbalance*: background voxels constitute over 98% of each volume. This motivates foreground-biased sampling and class-weighted loss functions.

=== Data Splits

A *stratified subset of 600 patients* was selected based on tumour volume distribution:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (center, center, center),
    stroke: none,
    table.hline(stroke: 1.5pt + accent),
    table.header([*Split*], [*Patients*], [*Percentage*]),
    table.hline(stroke: 0.5pt + luma(200)),
    [Training], [420], [70%],
    [Validation], [90], [15%],
    [Test], [90], [15%],
    table.hline(stroke: 1.5pt + accent),
  ),
  caption: [Data split configuration with stratified sampling by tumour volume.],
)

== Data Loading and Preprocessing

=== Patch-Based Sampling

Loading entire 3D volumes into GPU memory is infeasible on a 6 GB card. We use *patch-based training*:

- *Patch Size*: 96 × 96 × 96 voxels — balancing spatial context with VRAM constraints.
- *Patches per Volume*: 2 random patches per patient per epoch.
- *Foreground-Biased Sampling*: With probability 0.75, the patch centre is on a tumour voxel.

=== Intensity Normalisation

Each modality undergoes *Z-score normalisation* on non-zero brain voxels, with intensity clipping at the 0.5th and 99.5th percentiles to remove scanner-specific outliers.

== 3D Data Augmentation

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    stroke: none,
    table.hline(stroke: 1.5pt + accent),
    table.header([*Augmentation*], [*Probability*], [*Purpose*]),
    table.hline(stroke: 0.5pt + luma(200)),
    [Random axis flips], [50%], [Spatial invariance],
    [Random 90° rotations], [30%], [Rotational invariance],
    [Elastic deformation], [10%], [Tissue deformation simulation],
    [Intensity scaling/shifting], [30%], [Inter-scanner variability],
    table.hline(stroke: 1.5pt + accent),
  ),
  caption: [Data augmentation strategy applied on-the-fly to image and label tensors.],
)

== Model Architecture <sec-arch>

=== 3D U-Net with Squeeze-and-Excitation Attention

We employ a *3D U-Net* — the dominant paradigm for volumetric medical image segmentation — with an encoder–decoder structure and skip connections.

#rect(
  width: 100%,
  fill: rgb("#f8f8fc"),
  radius: 6pt,
  inset: 14pt,
  stroke: 0.5pt + luma(220),
)[
  #set text(size: 8.5pt, font: "DejaVu Sans Mono")
  ```
  Input (2 × 96³)
    ├─ Enc1: ConvBlock(2→16)   ─────────────────┐ Skip
    │  └─ MaxPool3D                               │
    ├─ Enc2: ConvBlock(16→32)  ────────────┐      │
    │  └─ MaxPool3D                         │     │
    ├─ Enc3: ConvBlock(32→64)  ───────┐     │     │
    │  └─ MaxPool3D                    │    │     │
    ├─ Enc4: ConvBlock(64→128) ──┐    │    │     │
    │  └─ MaxPool3D               │   │    │     │
    │                              │   │    │     │
    ├─ Bottleneck (128→256)       │   │    │     │
    │                              │   │    │     │
    ├─ Dec4: Up + Cat ←───────────┘   │    │     │
    ├─ Dec3: Up + Cat ←───────────────┘    │     │
    ├─ Dec2: Up + Cat ←────────────────────┘     │
    ├─ Dec1: Up + Cat ←──────────────────────────┘
    │
    └─ Output: Conv3D(16→4, k=1) → Softmax
  ```
]

*Key architectural innovations:*

+ *Squeeze-and-Excitation (SE) Blocks*: Channel attention that adaptively recalibrates feature responses via global pooling → bottleneck MLP → per-channel scaling.

+ *LeakyReLU*: Prevents "dying neuron" problem common in deep 3D networks (negative slope = 0.01).

+ *Residual Connections*: 1×1×1 shortcuts in each block for gradient flow.

+ *Deep Supervision*: Auxiliary heads at 1/2 and 1/4 resolution for multi-scale gradient signals.

The model contains *6,049,348 trainable parameters*.

== Loss Functions <sec-loss>

The extreme background dominance (>98%) makes standard Cross-Entropy inadequate. We use a *Combined Loss*:

$ cal(L)_"combined" = lambda_"dice" dot cal(L)_"dice" + lambda_"focal" dot cal(L)_"focal" $

where $lambda_"dice" = 0.6$ and $lambda_"focal" = 0.4$.

*Soft Dice Loss* optimises overlap directly, treating each class equally:

$ cal(L)_"dice" = 1 - 1/C sum_(c=1)^C (2 sum_i p_(i c) dot g_(i c) + epsilon) / (sum_i p_(i c) + sum_i g_(i c) + epsilon) $

*Focal Loss* down-weights easy voxels and focuses on hard boundary regions:

$ cal(L)_"focal" = - sum_c alpha_c (1 - p_(t c))^gamma log(p_(t c)) $

where $gamma = 2.0$ and $alpha_c$ are per-class weights inversely proportional to class frequency.

== Training Configuration

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    stroke: none,
    table.hline(stroke: 1.5pt + accent),
    table.header([*Parameter*], [*Value*], [*Rationale*]),
    table.hline(stroke: 0.5pt + luma(200)),
    [Optimiser], [AdamW], [Decoupled weight decay],
    [Initial LR], [1×10⁻³], [Standard for 3D segmentation],
    [Weight Decay], [1×10⁻⁵], [Mild L2 regularisation],
    [Batch Size], [2], [Maximises GPU within 6 GB VRAM],
    [Grad Accumulation], [2 steps], [Effective batch = 4],
    [Gradient Clipping], [1.0], [Prevents gradient explosion],
    [Mixed Precision], [Enabled], [Halves VRAM; ~30% speedup],
    [Early Stopping], [Patience 10], [Val Dice monitor],
    table.hline(stroke: 1.5pt + accent),
  ),
  caption: [Training hyperparameters.],
)

=== Learning Rate Schedule

Two-phase schedule: *Linear Warm-up* (epochs 1–5, LR ramps from 1% to 100%) followed by *Cosine Annealing* (epochs 6–50, smooth decay to 1×10⁻⁶).

=== Hardware

Training on an *NVIDIA GeForce RTX 3060 Laptop GPU* (6 GB VRAM) under WSL2 with cuDNN auto-tuning, 8 data workers with prefetch factor 4, and persistent workers.


// ═══════════════════════════════════════════════════════════════════════════
= Results

== Training Progression

The model trained for the full *50 epochs* without early stopping being triggered.

#figure(
  image("outputs/results/training_curves.png", width: 95%),
  caption: [Training curves: loss, Dice score, and learning rate over 50 epochs.],
) <fig-curves>

Key observations:
- *Rapid convergence*: Val Dice 0.12 → 0.59 in 5 epochs.
- *Steady refinement*: Continued climbing through epochs 10–30.
- *Best performance*: Peak validation Dice of *0.7639* at epoch 48.
- *No overfitting*: Training and validation curves track closely.

== Test Set Evaluation

#v(6pt)
// Metric cards
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  column-gutter: 8pt,
  metric-card("Mean Dice", "0.775", color: accent),
  metric-card("Mean IoU", "0.704", color: accent2),
  metric-card("Best Epoch", "48", color: success),
  metric-card("Parameters", "6.05M", color: mid),
)
#v(8pt)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    table.hline(stroke: 1.5pt + accent),
    table.header([*Tumour Region*], [*Dice Score*], [*IoU*], [*Hausdorff (mm)*]),
    table.hline(stroke: 0.5pt + luma(200)),
    [Necrotic Core], [*0.769*], [0.728], [12.07],
    [Peritumoral Oedema], [*0.822*], [0.744], [23.76],
    [Enhancing Tumour], [*0.735*], [0.641], [21.19],
    table.hline(stroke: 0.8pt + luma(180)),
    [*Mean (Foreground)*], [*0.775*], [*0.704*], [—],
    table.hline(stroke: 1.5pt + accent),
  ),
  caption: [Test set segmentation metrics on 90 held-out patients.],
) <tbl-results>

#figure(
  image("outputs/results/dice_per_class.png", width: 65%),
  caption: [Per-class Dice scores on the test set.],
)

== Per-Class Analysis

*Peritumoral Oedema (Dice = 0.822)* achieves the highest score, consistent with its larger spatial extent and high contrast on T2-FLAIR. The U-Net's multi-scale features capture its diffuse boundaries well.

*Necrotic Core (Dice = 0.769)* performs strongly despite being smaller and more heterogeneous. SE attention helps distinguish necrotic tissue by learning modality-specific feature weightings.

*Enhancing Tumour (Dice = 0.735)* is the most challenging due to its thin, ring-like morphology. Focal Loss with $gamma = 2.0$ up-weights hard boundary voxels to address this.

== Qualitative Results

#grid(
  columns: (1fr, 1fr),
  column-gutter: 8pt,
  row-gutter: 8pt,
  figure(image("outputs/results/sample_predictions/sample_01.png", width: 100%), caption: [Patient 1]),
  figure(image("outputs/results/sample_predictions/sample_02.png", width: 100%), caption: [Patient 2]),
  figure(image("outputs/results/sample_predictions/sample_03.png", width: 100%), caption: [Patient 3]),
  figure(image("outputs/results/sample_predictions/sample_04.png", width: 100%), caption: [Patient 4]),
)

== Fine-Tuning Experiment

Following initial training, a *fine-tuning phase* used the best model (epoch 48) with 10× lower LR (1×10⁻⁴) and cosine annealing. Early stopping triggered after 9 epochs — no epoch exceeded the baseline Dice of 0.7639, confirming the initial training had converged to a strong solution.

#info-box(title: "Fine-Tuning Insight")[
  Fine-tuning improved *Necrotic Core* (0.769 → 0.802 Dice) at the expense of Oedema, suggesting a trade-off in the loss landscape between sub-region specialisation. The initial model remains the best overall.
]


// ═══════════════════════════════════════════════════════════════════════════
= Analysis and Discussion

== Addressing Coursework Challenges

=== 3D Data Handling
Patch-based sampling (96³ voxels) effectively manages memory, enabling training on a 6 GB laptop GPU. Foreground-biased sampling (75% tumour-centred) ensures sufficient positive examples.

=== Varied Tumour Sizes and Shapes
The multi-scale U-Net (four encoder levels, 96³ → 6³ resolution) handles size variability. Skip connections preserve fine boundaries. SE attention adapts to tumour heterogeneity by dynamically re-weighting channels.

=== Class Imbalance
Three complementary strategies: (1) foreground-biased sampling, (2) Focal Loss ($gamma = 2.0$), and (3) Soft Dice Loss. This yields balanced per-class scores all above 0.73.

=== Limited Annotated Data
Four augmentation strategies (flips, rotations, elastic deformation, intensity perturbation) expand the effective training set. SE attention and deep supervision act as implicit regularisers.

== Comparison with Literature

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    stroke: none,
    table.hline(stroke: 1.5pt + accent),
    table.header([*Method*], [*Mean Dice*], [*Notes*]),
    table.hline(stroke: 0.5pt + luma(200)),
    [Standard 3D U-Net], [0.68–0.72], [Baseline without attention],
    [nnU-Net (MICCAI 2020)], [0.80–0.84], [Full dataset, auto-configured],
    [*Our approach*], [*0.775*], [600 patients, laptop GPU, 50 epochs],
    table.hline(stroke: 1.5pt + accent),
  ),
  caption: [Comparison with published brain tumour segmentation methods.],
)

Achieving 0.775 with 33% of available data on a laptop GPU demonstrates the effectiveness of SE attention, deep supervision, and combined Focal + Dice loss.

== Limitations and Future Work

+ *Subset Training*: Full dataset (1,809 patients) would likely improve results by 3–5%.
+ *Two Modalities*: Incorporating T1n and T2w could provide additional discriminative information.
+ *Post-Processing*: Connected component analysis could reduce false positives and improve Hausdorff distance.
+ *Test-Time Augmentation*: Averaging across flipped/rotated inputs typically improves Dice by 1–2%.


// ═══════════════════════════════════════════════════════════════════════════
= Conclusions

#rect(
  width: 100%,
  fill: gradient.linear(rgb("#eef2ff"), rgb("#f0fdfa"), angle: 0deg),
  radius: 6pt,
  inset: 16pt,
  stroke: 0.5pt + luma(220),
)[
  We developed a complete, production-grade pipeline for *3D brain tumour segmentation* achieving:

  #grid(
    columns: (1fr, 1fr),
    column-gutter: 12pt,
    row-gutter: 8pt,
    [✓ *Mean Dice Score: 0.775* on 90 test patients],
    [✓ *All sub-regions above 0.73 Dice*],
    [✓ *Class imbalance addressed* via triple strategy],
    [✓ *6 GB laptop GPU training* via AMP + patches],
    [✓ *Resume-capable checkpointing* for robustness],
    [✓ *Fully automated* single-command execution],
  )
]

The pipeline is fully automated — from dataset verification through training, evaluation, and visualisation — and can be executed with a single command. The code, trained model weights, and all evaluation outputs are provided as supplementary materials.
