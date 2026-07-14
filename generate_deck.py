import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_eurosat_presentation():
    prs = Presentation()
    
    # Set slide dimensions to 16:9 Widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    # Theme Colors
    DARK_BG = RGBColor(30, 41, 59)      # Deep Slate Blue/Gray
    LIGHT_BG = RGBColor(248, 250, 252)  # Cool Off-White
    TEXT_DARK = RGBColor(15, 23, 42)    # Dark Charcoal
    TEXT_LIGHT = RGBColor(255, 255, 255)# Pure White
    ACCENT_GREEN = RGBColor(16, 185, 129)# Emerald Green
    MUTED_GRAY = RGBColor(100, 116, 139)# Slate Gray

    blank_layout = prs.slide_layouts[6]

    # -------------------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE (Dark Theme)
    # -------------------------------------------------------------------------
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = DARK_BG
    bg1.line.fill.background()

    # Title & Subtitle text box
    title_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(2.5))
    tf1 = title_box.text_frame
    tf1.word_wrap = True
    
    p1 = tf1.paragraphs[0]
    p1.text = "Fine-Tuning Vision Transformers"
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_LIGHT
    p1.alignment = PP_ALIGN.LEFT
    
    p2 = tf1.add_paragraph()
    p2.text = "13-Band Multispectral Satellite Land-Cover Classification on EuroSAT"
    p2.font.size = Pt(28)
    p2.font.bold = True
    p2.font.color.rgb = ACCENT_GREEN
    p2.space_before = Pt(10)
    
    p3 = tf1.add_paragraph()
    p3.text = "Adapting ViT-B/16 for full-spectrum Sentinel-2 imagery under a 6 GB VRAM constraint"
    p3.font.size = Pt(16)
    p3.font.color.rgb = MUTED_GRAY
    p3.space_before = Pt(15)

    # Dashboard Metrics row at the bottom
    metrics = [
        ("97.58%", "Final Test Acc"),
        ("0.9747", "Macro F1-Score"),
        ("13 Channels", "Full Spectral Depth"),
        ("6 GB GPU", "RTX 3050 Native Execution")
    ]
    for i, (val, label) in enumerate(metrics):
        box = slide1.shapes.add_textbox(Inches(1.0 + i * 2.8), Inches(5.2), Inches(2.6), Inches(1.5))
        tf = box.text_frame
        tf.word_wrap = True
        
        pv = tf.paragraphs[0]
        pv.text = val
        pv.font.size = Pt(32)
        pv.font.bold = True
        pv.font.color.rgb = ACCENT_GREEN
        
        pl = tf.add_paragraph()
        pl.text = label
        pl.font.size = Pt(12)
        pl.font.color.rgb = TEXT_LIGHT
        pl.space_before = Pt(5)

    # -------------------------------------------------------------------------
    # SLIDE 2: DATASET & LIBRARY SHIFT (Light Theme)
    # -------------------------------------------------------------------------
    slide2 = prs.slides.add_slide(blank_layout)
    bg2 = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg2.fill.solid()
    bg2.fill.fore_color.rgb = LIGHT_BG
    bg2.line.fill.background()

    # Section Title
    title_box = slide2.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.5), Inches(0.8))
    tf2 = title_box.text_frame
    p = tf2.paragraphs[0]
    p.text = "Dataset & Ingestion Infrastructure Shift"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK

    # Content Left: The Library Transition
    left_box = slide2.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.5), Inches(4.8))
    tf_left = left_box.text_frame
    tf_left.word_wrap = True
    
    p = tf_left.paragraphs[0]
    p.text = "Core API & Framework Evolution"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    
    bullets_left = [
        "Legacy Framework: Built upon torchvision.datasets.EuroSAT, restricted entirely to 3-channel visual RGB imagery arrays.",
        "Multispectral Ingestion: Migrated to Hugging Face datasets interface to fetch the raw giswqs/EuroSAT_MS data pipeline (~2.25 GB compressed cache).",
        "Data Integrity Fix: The multispectral array naturally contains 26,998 structural image patches, cleanly bypassing the 2 empty corrupted artifacts present in the 27,000 RGB package."
    ]
    for b in bullets_left:
        p = tf_left.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(12)

    # Content Right: Distribution Metrics Table
    rows, cols = 4, 2
    left, top, width, height = Inches(6.8), Inches(2.2), Inches(5.5), Inches(3.5)
    table_shape = slide2.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    table.columns[0].width = Inches(3.0)
    table.columns[1].width = Inches(2.5)

    data = [
        ["Dataset Split Partition", "Patch Cardinality"],
        ["Training Split (70%)", "18,898 Images"],
        ["Validation Split (15%)", "4,049 Images"],
        ["Testing Split (15%)", "4,051 Images"]
    ]
    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.text = data[r][c]
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(14)
            if r == 0:
                p.font.bold = True
                p.font.color.rgb = TEXT_LIGHT
                cell.fill.solid()
                cell.fill.fore_color.rgb = DARK_BG
            else:
                p.font.color.rgb = TEXT_DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(235, 240, 245) if r % 2 == 0 else RGBColor(255, 255, 255)

    # -------------------------------------------------------------------------
    # SLIDE 3: CORE ARCHITECTURAL ADAPTATIONS (Light Theme)
    # -------------------------------------------------------------------------
    slide3 = prs.slides.add_slide(blank_layout)
    bg3 = slide3.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg3.fill.solid()
    bg3.fill.fore_color.rgb = LIGHT_BG
    bg3.line.fill.background()

    title_box = slide3.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.5), Inches(0.8))
    tf3 = title_box.text_frame
    p = tf3.paragraphs[0]
    p.text = "Core Architectural Adaptations (RGB to MS)"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK

    # Column 1
    c1_box = slide3.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.5), Inches(5.0))
    tf_c1 = c1_box.text_frame
    tf_c1.word_wrap = True
    
    p = tf_c1.paragraphs[0]
    p.text = "Patch Embedding Mutation"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    
    b_c1 = [
        "Dimension Overhaul: Standard timm.create_model initializes the entry-level patch_embed.proj layer specifically for a Conv2d(3, 768) ImageNet array.",
        "In-Place Expansion: Dynamically re-profiled the convolutional layer structural configuration to process Conv2d(13, 768) directly before parsing state dict weight checkpoints.",
        "Seeding Non-Visual Bands: While visible Bands 3, 2, 1 read ImageNet spatial parameters explicitly, the 10 remaining invisible bands (Red-Edge, NIR, SWIR) are mathematically initialized via the average mean of RGB parameters."
    ]
    for b in b_c1:
        p = tf_c1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(12)

    # Column 2
    c2_box = slide3.shapes.add_textbox(Inches(6.8), Inches(1.8), Inches(5.5), Inches(5.0))
    tf_c2 = c2_box.text_frame
    tf_c2.word_wrap = True
    
    p = tf_c2.paragraphs[0]
    p.text = "Tensor-Driven Transforms"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = DARK_BG
    
    b_c2 = [
        "PIL Engine Obsolescence: Standard Python Imaging Library (PIL) engines physically break or throw exceptions when holding a multi-channel 13-band image matrix.",
        "Functional Vectoring: Rewrote the entire data augmentation framework using functional torchvision.transforms.v2 primitives.",
        "Zero-Copy Tensors: Images are cast into underlying raw floating-point PyTorch C×H×W tensors immediately upon extraction, preserving hyperspectral data precision throughout the fine-tuning operations."
    ]
    for b in b_c2:
        p = tf_c2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(12)

    # -------------------------------------------------------------------------
    # SLIDE 4: PROGRESSIVE UNFREEZING PERFORMANCE (Light Theme)
    # -------------------------------------------------------------------------
    slide4 = prs.slides.add_slide(blank_layout)
    bg4 = slide4.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg4.fill.solid()
    bg4.fill.fore_color.rgb = LIGHT_BG
    bg4.line.fill.background()

    title_box = slide4.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.5), Inches(0.8))
    tf4 = title_box.text_frame
    p = tf4.paragraphs[0]
    p.text = "Multi-Stage Progressive Unfreezing Performance"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK

    # Progression Table
    t_rows, t_cols = 4, 4
    t_left, t_top, t_width, t_height = Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.5)
    t_shape = slide4.shapes.add_table(t_rows, t_cols, t_left, t_top, t_width, t_height)
    tbl = t_shape.table
    tbl.columns[0].width = Inches(2.2)
    tbl.columns[1].width = Inches(3.2)
    tbl.columns[2].width = Inches(2.0)
    tbl.columns[3].width = Inches(4.333)

    t_data = [
        ["Fine-Tuning Phase", "Trainable / Total Parameters", "Best Val Acc", "Execution Strategy Notes"],
        ["Stage 1: Head Only", "2,564,362 / 87,772,426 (2.92%)", "96.25%", "Backbone frozen. High baseline accuracy driven by raw multi-spectral signature resolution."],
        ["Stage 2: Last 4 Blocks", "30,534,922 / 87,772,426 (34.78%)", "97.09%", "Unfreezes blocks 8-11. Moderate learning rate (1e-5) lets deep features adjust to land patterns."],
        ["Stage 3: Last 8 Blocks", "59,268,874 / 87,772,426 (67.53%)", "97.38%", "Unfreezes blocks 4-11. Tight micro-learning rate (5e-6) prevents catastrophic forgetting."]
    ]
    for r in range(t_rows):
        for c in range(t_cols):
            cell = tbl.cell(r, c)
            cell.text = t_data[r][c]
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(13)
            cell.text_frame.word_wrap = True
            if r == 0:
                p.font.bold = True
                p.font.color.rgb = TEXT_LIGHT
                cell.fill.solid()
                cell.fill.fore_color.rgb = DARK_BG
            else:
                p.font.color.rgb = TEXT_DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(240, 244, 248) if r % 2 == 0 else RGBColor(255, 255, 255)
                if c == 2 and r == 3: # Highlight final stage validation metric
                    p.font.bold = True
                    p.font.color.rgb = ACCENT_GREEN

    # -------------------------------------------------------------------------
    # SLIDE 5: FINAL EVALUATION & CLASS DIAGNOSTICS (Light Theme)
    # -------------------------------------------------------------------------
    slide5 = prs.slides.add_slide(blank_layout)
    bg5 = slide5.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg5.fill.solid()
    bg5.fill.fore_color.rgb = LIGHT_BG
    bg5.line.fill.background()

    title_box = slide5.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.5), Inches(0.8))
    tf5 = title_box.text_frame
    p = tf5.paragraphs[0]
    p.text = "Final Stage-3 Test Matrix & Target Performance"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK

    # Left Box - Top Metrics & Super performers
    l_box = slide5.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tf_l = l_box.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "Operational Test Strengths"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    
    b_l = [
        "Test Loss Minimum: Achieved an optimal 0.1650 scalar cross-entropy loss barrier.",
        "Flawless Classification: Forest reached an near-perfect 0.9979 F1-score.",
        "Absolute Water Capture: SeaLake hit a perfect 1.0000 recall curve, successfully mapping every single one of the 448 target image coordinates without a single false negative footprint."
    ]
    for b in b_l:
        p = tf_l.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

    # Right Box - Class F1 Table
    c_rows, c_cols = 11, 2
    c_left, c_top, c_width, c_height = Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2)
    c_shape = slide5.shapes.add_table(c_rows, c_cols, c_left, c_top, c_width, c_height)
    c_tbl = c_shape.table
    c_tbl.columns[0].width = Inches(3.2)
    c_tbl.columns[1].width = Inches(2.5)

    c_data = [
        ["EuroSAT Land Class Category", "Stage-3 Test F1-Score"],
        ["Forest", "0.9979"],
        ["SeaLake", "0.9967"],
        ["Residential", "0.9809"],
        ["River", "0.9775"],
        ["Industrial", "0.9769"],
        ["AnnualCrop", "0.9750"],
        ["HerbaceousVegetation", "0.9728"],
        ["Pasture", "0.9713"],
        ["PermanentCrop (Target for TTA)", "0.9562"],
        ["Highway (Target for TTA)", "0.9422"]
    ]
    for r in range(c_rows):
        for c in range(c_cols):
            cell = c_tbl.cell(r, c)
            cell.text = c_data[r][c]
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(11)
            if r == 0:
                p.font.bold = True
                p.font.color.rgb = TEXT_LIGHT
                cell.fill.solid()
                cell.fill.fore_color.rgb = DARK_BG
            else:
                p.font.color.rgb = TEXT_DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(245, 247, 250) if r % 2 == 0 else RGBColor(255, 255, 255)
                # Highlight targets for TTA optimization
                if r in [9, 10]:
                    p.font.bold = True
                    if c == 1:
                        p.font.color.rgb = RGBColor(220, 38, 38) # Soft Red highlight for lower F1

    # -------------------------------------------------------------------------
    # SLIDE 6: DATA GOVERNANCE & SOURCE OF TRUTH (Dark Theme Conclusion)
    # -------------------------------------------------------------------------
    slide6 = prs.slides.add_slide(blank_layout)
    bg6 = slide6.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg6.fill.solid()
    bg6.fill.fore_color.rgb = DARK_BG
    bg6.line.fill.background()

    title_box = slide6.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.5), Inches(0.8))
    tf6 = title_box.text_frame
    p = tf6.paragraphs[0]
    p.text = "Data Governance: 'One Source of Truth' Architecture"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = TEXT_LIGHT

    c_box = slide6.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.733), Inches(4.5))
    tf_c = c_box.text_frame
    tf_c.word_wrap = True

    b_c = [
        "Preventing Hardcoded Drift: Manually hardcoding 13-channel floating-point normalization mean and standard deviation matrices across distinct training and evaluation source repositories risks running script variations.",
        "Automated Sidecar Telemetry: The train_v3_multispectral.py compilation logic explicitly captures calculated global band statistics and exports them directly into checkpoints/ms_band_stats.json upon dataset partition initialization.",
        "Decoupled Post-Processing: The standalone Test-Time Augmentation (TTA_ms.py) environment reads directly from this JSON telemetry output, ensuring strict parity without hardcoding constants.",
        "Next Operational Step: Deploying TTA_ms.py across 5 geometrical orientations (Horizontal Flip, Vertical Flip, Rot90, Rot270, and Baseline) to boost weak boundary lines on structurally narrow channels like Highways and Permanent Crops."
    ]
    for b in b_c:
        p = tf_c.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(16)
        p.font.color.rgb = RGBColor(226, 232, 240)
        p.space_before = Pt(18)
        # Highlight key file terms in accent green
        if "ms_band_stats.json" in b or "TTA_ms.py" in b:
            p.font.color.rgb = ACCENT_GREEN

    # Save presentation
    filename = "EuroSAT_13Band_MS_Presentation.pptx"
    prs.save(filename)
    print(f"[SUCCESS] Presentation successfully written to: {os.path.abspath(filename)}")

if __name__ == "__main__":
    create_eurosat_presentation()