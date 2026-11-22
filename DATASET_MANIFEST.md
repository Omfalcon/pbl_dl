# DATASET_MANIFEST.md — Dataset Strategy, Manifest Schemas & Provenance

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Status:** Approved for Baseline Architecture (Phase 0)  
**Last Updated:** 2026-11-22  

---

## 1. Visual Intelligence Dataset Architecture & Engineering Targets

Visual intelligence represents the primary empirical research pillar of PhishVision AI. The dataset strategy establishes four specialized visual datasets, a unified normalized URL corpus, and one curated brand reference gallery:

| Dataset Modality | Target Volume | Acquired / Prepared Status | Annotations & Formats | Primary Model Consumer |
| :--- | :--- | :--- | :--- | :--- |
| **1. General Webpage Screenshots** | 10,000 images (5k Legit / 5k Phish) | **1,008 valid screenshots on disk** (804 Phish / 204 Legit; 1,000 from `data/raw/external_dataset/` + 8 scanner baseline) | Binary label (`0`/`1`), Domain, Timestamp, pHash | Module 1: ResNet-18/50 Classifier |
| **2. Brand Visual Comparison Pairs** | $\ge 1,000$ image pairs (18 Brands) | 18 Brand workspaces scaffolded; 0 pairs curated (pending brand image assets) | Pair tuples `(img1, img2, label)`, Triplet tuples | Module 2: Siamese CNN (Triplet Loss) |
| **3. UI Component & Logo Detection** | ~1,500 annotated screenshots | Annotation engine & independent validator operational; 0 annotations on disk | YOLO format `.txt` (5 classes, normalized bboxes) | Module 3: Ultralytics YOLOv8/v11 |
| **4. Webpage Region Segmentation** | ~1,000 annotated screenshots | `SegmentationMaskManager` operational; 0 masks on disk | 8-bit PNG masks (U-Net) & COCO JSON (YOLO-Seg) | Module 4: U-Net & YOLO-Seg |
| **5. Curated Brand Reference Gallery**| 18 high-value target brands | **18 brand folders scaffolded**; 0 reference images on disk | Canonical domains, brand metadata, SVG/PNG assets | Module 2 Gallery & Domain Association |
| **6. Redirect & Navigation Sequences** | ~2,000 multi-hop traces | Scanner redirect capture active | Chronological hops, HTTP status, domain switches | LSTM & Transformer Sequence Models |
| **7. Lexical URL Feature Vectors** | 20,000+ tabular records | **44,129 verified records** + **50,000 Tranco candidate domains** | PSL domains, tracking stripped, binary labels | MLP Baseline Model |

> [!NOTE]
> Phase 2 has successfully prepared **94,125 unique URLs** partitioned across **55,364 unique domains** with **zero domain leakage**. The raw corpus combines PhishTank (43k), Tranco (50k), PhreshPhish (1k HTML pairs), and OpenPhish (300).

---

## 2. Curated Visual Brand Reference Dataset (`data/brands/reference_gallery/`)

To support conditional brand impersonation verification without maintaining an exhaustive database of every website on the internet, a dedicated, high-fidelity brand reference dataset is maintained for 15–20 selected brands:

### 2.1 Monitored Target Brands (18 Selected Institutions)
1. **State Bank of India (`sbi`):** Official domains: `sbi.co.in`, `onlinesbi.sbi`, `onlinesbi.com`.
2. **HDFC Bank (`hdfc`):** Official domains: `hdfcbank.com`, `netbanking.hdfcbank.com`.
3. **PayPal (`paypal`):** Official domains: `paypal.com`, `paypal-community.com`.
4. **Microsoft (`microsoft`):** Official domains: `microsoft.com`, `live.com`, `office.com`, `login.microsoftonline.com`.
5. **Google (`google`):** Official domains: `google.com`, `accounts.google.com`, `youtube.com`.
6. **Apple (`apple`):** Official domains: `apple.com`, `icloud.com`, `appleid.apple.com`.
7. **Amazon (`amazon`):** Official domains: `amazon.com`, `amazon.in`, `amazon.co.uk`.
8. **Chase Bank (`chase`):** Official domains: `chase.com`, `chaseonline.chase.com`.
9. **Wells Fargo (`wellsfargo`):** Official domains: `wellsfargo.com`.
10. **Netflix (`netflix`):** Official domains: `netflix.com`.
11. **Meta / Facebook (`facebook`):** Official domains: `facebook.com`, `fb.com`.
12. **Instagram (`instagram`):** Official domains: `instagram.com`.
13. **WhatsApp (`whatsapp`):** Official domains: `whatsapp.com`, `web.whatsapp.com`.
14. **Bank of America (`bofa`):** Official domains: `bankofamerica.com`.
15. **DHL Express (`dhl`):** Official domains: `dhl.com`, `express.dhl`.
16. **FedEx (`fedex`):** Official domains: `fedex.com`.
17. **LinkedIn (`linkedin`):** Official domains: `linkedin.com`.
18. **Adobe (`adobe`):** Official domains: `adobe.com`, `account.adobe.com`.

### 2.2 Brand Gallery Filesystem Structure
```
data/brands/reference_gallery/
├── manifest_brands.json                 # Master registry of all reference brands
├── sbi/
│   ├── metadata.json                    # Canonical domains, synonyms, category
│   ├── logos/
│   │   ├── logo_primary.png             # Transparent RGBA 512x512
│   │   ├── logo_monochrome.png          # High-contrast version
│   │   ├── logo_icon.png                # Favicon / circular emblem
│   │   └── vector_master.svg            # Lossless vector ground truth
│   ├── canonical_screenshots/
│   │   ├── login_desktop.png            # Official login page screenshot
│   │   └── home_desktop.png             # Official portal homepage
│   └── embeddings/
│       └── reference_embeddings.npy     # 256-d unit vectors from Siamese network
└── paypal/
    └── ...
```

### 2.3 Brand Manifest Schema (`manifest_brands.json`)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "brands": [
    {
      "brand_id": "paypal",
      "display_name": "PayPal Inc.",
      "category": "Banking & Payments",
      "official_registrable_domains": ["paypal.com", "paypal-community.com"],
      "authorized_subdomains": ["www.paypal.com", "login.paypal.com"],
      "assets": {
        "primary_logo": "data/brands/reference_gallery/paypal/logos/logo_primary.png",
        "icon": "data/brands/reference_gallery/paypal/logos/logo_icon.png",
        "canonical_login": "data/brands/reference_gallery/paypal/canonical_screenshots/login_desktop.png",
        "embedding_file": "data/brands/reference_gallery/paypal/embeddings/reference_embeddings.npy"
      }
    }
  ]
}
```

---

## 3. Detailed Specifications for Visual Datasets

### 3.1 Dataset 1: General Webpage Screenshots (ResNet Classifier)
- **Target Count:** 5,000 legitimate and 5,000 phishing screenshots.
- **Diversity Coverage:** Legitimate samples stratified across 8 industry verticals (Banking, Retail, Tech, Education, Government, Media, Health, Travel).
- **Format:** Lossless PNG, $1280 \times 720$ resolution, resized to $224 \times 224$ during preprocessing.
- **Manifest:** `data/manifests/manifest_screenshots_general.csv` with fields: `sample_id`, `url`, `registrable_domain`, `label` (0/1), `phash`, `split` (train/val/test).

### 3.2 Dataset 2: Brand Comparison Pairs (Siamese CNN)
- **Target Count:** $\ge 1,000$ paired crops ($128 \times 128 \times 3$).
- **Composition:**
  - **Positive Authentic Pairs (30%):** Official logo vs. alternate official resolutions, light/dark modes, and minor rendering variations.
  - **Negative Hard-Lookalike Pairs (40%):** Official logo vs. phishing attacker spoofed variants (re-colored, perturbed, partially occluded, homoglyphic typography).
  - **Negative Cross-Brand Pairs (30%):** Official brand logo vs. unrelated authentic logos (e.g., SBI vs. PayPal).
- **Manifest:** `data/manifests/manifest_siamese_pairs.csv` with fields: `pair_id`, `img1_path`, `img2_path`, `brand_id`, `is_match` (1 for positive, 0 for negative).

### 3.3 Dataset 3: UI Component Detection (YOLO)
- **Target Count:** 1,500 screenshots with dense bounding box annotations.
- **Classes:**
  - `0`: `brand_logo`
  - `1`: `username_input`
  - `2`: `password_input`
  - `3`: `submit_button`
  - `4`: `qr_code`
- **Annotation Format:** YOLO TXT format: `<class_id> <x_center> <y_center> <width> <height>` normalized $[0, 1]$.
- **Storage:** `data/detection/images/{train,val,test}/` and `data/detection/labels/{train,val,test}/`.

### 3.4 Dataset 4: Webpage Region Segmentation (U-Net & YOLO-Seg)
- **Target Count:** 1,000 annotated screenshots.
- **Classes:** `[0: background, 1: header, 2: credential_form, 3: body_content, 4: footer]`.
- **Formats:**
  - Semantic: 8-bit PNG single-channel mask images ($256 \times 256$) where pixel values correspond to class indices.
  - Instance: COCO JSON segmentation format with polygon coordinates for individual form containers.

---

## 4. Evaluation of Candidate Data Sources

| Source | Category | Modality | Licensing | Quality / Reliability | Utilization Plan |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Zenodo Phishing Dataset** | Phishing & Legit | Screenshots & HTML | CC-BY 4.0 | High academic quality; requires pHash deduplication. | Primary source for ResNet and YOLO baseline images. |
| **PhreshPhish** | Phishing | URLs & Brand Labels | Open Access | Brand labels attached; excellent for Siamese pair curation. | Primary source for brand lookalike crops. |
| **Tranco Top Sites** | Legitimate | Domain CSV | Open Data | Hardened popularity list; active crawling needed. | Candidate domains for legitimate screenshot sampling. |
| **PhishTank** | Active Phishing | JSON/CSV API | Developer Agreement | Community-verified with submission timestamps. | Source for fresh phishing URLs and redirect traces. |
| **OpenPhish** | Active Phishing | Plaintext URL feed | Free Community Tier | Fast zero-day updates. | Sourcing temporal holdout test samples. |
| **URLhaus** | Malware/Phishing | CSV / JSON | CC0 | Highly verified malicious hosting infrastructure. | High-risk redirect and domain transition patterns. |

---

## 5. Leakage-Safe Partitioning Protocols

1. **Strict Registrable Domain Split:** All samples sharing the same PSL registrable domain reside strictly within the same partition (`train`, `val`, or `test`). Zero domain overlap is enforced.
2. **Temporal Holdout:** Samples from the latest 15% chronological window are reserved exclusively for the test set to measure zero-day generalization.
3. **Perceptual Deduplication:** Image pHash Hamming distance threshold $\le 5$ removes duplicate layouts prior to splitting.
4. **Reproducibility:** Seed 42 strictly enforced across all partitioning scripts.
