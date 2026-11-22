# DATASET_SOURCES.md — PhishVision AI Dataset Source Research & Verification

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Phase:** Phase 2 — Dataset Acquisition, Preparation, Annotation & Validation  
**Date:** 2026-11-22  
**Verification Method:** Empirical HTTP probing, API inspection, and official documentation analysis.

---

## 1. Executive Summary & Source Matrix

This document provides a rigorous, verified audit of candidate data sources for training and evaluating PhishVision AI's deep learning and multi-evidence detection models.

| Source Name | Modality | Primary Modality Details | Access Method | Auth / Key Required | Verified Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Zenodo (Record 8041387)** | Multimodal (Screenshots, HTML, Metadata) | 10,395 websites (5,151 phish / 5,244 legit) across 86 brands; visual screenshots & DOM. | REST API / Direct HTTP | None (Open Access) | **ACTIVE & VERIFIED** |
| **Zenodo (Record 23477065)** | Multimodal (Screenshots + URLs) | 60,000 URLs, 21,311 rendered screenshots (Phish-Iris benchmark). | REST API / Direct HTTP | None (Open Access) | **ACTIVE & VERIFIED** |
| **PhreshPhish** | Webpage HTML + URLs | 650,000+ verified URL + raw HTML pairs with brand targets. | Hugging Face Dataset API / Git LFS | None (Hugging Face Open) | **ACTIVE & VERIFIED** |
| **PhishTank** | URL Feed + Target Metadata | Active, verified phishing URLs with submission & verification timestamps, target brand names. | Direct HTTP / REST API | None for rate-limited public access; App Key recommended for high volume. | **ACTIVE & VERIFIED** |
| **OpenPhish** | URL Feed | Zero-day / active phishing URLs updated every 6 hours (Community Feed). | Direct HTTP download | None for Community Feed; Commercial key for premium feed. | **ACTIVE & VERIFIED** |
| **URLhaus (abuse.ch)** | Malware & Phishing URLs | Threat intelligence feed of active malware/phishing distribution sites. | REST API v2 export | **Auth-Key Required** (Retired keyless downloads; returns HTTP 410 Gone without Auth-Key). | **AUTH REQUIRED** |
| **Tranco List** | Top Legitimate Domains | Top 1M domains aggregated across Chrome FLoC, Cloudflare, Umbrella, and Majestic. | Direct ZIP/CSV download | None (Open Access) | **ACTIVE & VERIFIED** |

---

## 2. Source-by-Source Empirical Audit

### 2.1 Zenodo Phishing & Brand Datasets

- **Official Source URLs:**
  - Record 8041387: `https://zenodo.org/records/8041387` (Brand Impersonation & Visuals)
  - Record 23477065: `https://zenodo.org/records/23477065` (Multimodal Phishing & Legitimate Websites)
- **API Availability:** REST API at `https://zenodo.org/api/records/<record_id>`. Provides file metadata, direct download links, and checksums.
- **Registration Requirements:** None for public records. Optional personal access token for higher API rate limits.
- **License & Permitted Usage:** Creative Commons Attribution 4.0 International (CC BY 4.0). Full academic and research reuse permitted.
- **Available Labels:** Binary (`phishing` vs `not-phishing`), and brand targets (`brands.csv` mapping 86 global brands).
- **Available Visuals:**
  - Record 8041387: Modular chunked archives (e.g. `phishing_0001-0500.zip` [808 MB], `not-phishing_0001-0500.zip` [3.6 GB]) containing full PNG screenshots and HTML DOM snapshots.
  - Record 23477065: `Phish-irish.zip` (798.9 MB) containing 21,311 curated screenshot images.
- **Dataset Limitations:** Large archive sizes require modular/selective chunked downloading to avoid local storage exhaustion. Historical captures reflect temporal distribution of collection dates (2021–2023).

---

### 2.2 PhreshPhish Benchmark

- **Official Source URL:** `https://huggingface.co/datasets/phreshphish/phreshphish`
- **Download Method:** Hugging Face `datasets` Python library or direct Parquet/JSONL stream.
- **Registration Requirements:** None for public data streaming.
- **License:** Open Research / Creative Commons.
- **Available Labels:** Binary (`label: phish | benign`), target brand (`target: str`), and language confidence (`lang`, `lang_score`).
- **Available Content:** URL, full raw page HTML, SHA-256 hash, and timestamps.
- **Critical Limitation for Computer Vision:** **PhreshPhish contains full HTML DOM and URLs, but does NOT contain pre-rendered screenshot images.** Screenshots must be rendered using our Phase 1 scanner if visual features are required from this feed.

---

### 2.3 PhishTank (OpenDNS / Cisco)

- **Official Source URL:** `https://data.phishtank.com/data/online-valid.csv` (also available as `.json` and `.xml`).
- **API Availability:** Developer REST API (`https://checkurl.phishtank.com/checkurl/`).
- **Registration Requirements:** Public download accessible with proper `User-Agent`. An API key (`PHISHTANK_API_KEY`) is recommended to avoid aggressive IP-based rate limiting (HTTP 429).
- **License & Permitted Usage:** Free for non-commercial academic and research use with attribution.
- **Available Metadata:** `phish_id`, `url`, `phish_detail_url`, `submission_time`, `verified`, `verification_time`, `online`, `target`.
- **Visual Evidence:** **No screenshots provided directly.** Contains live verified phishing URLs that can be selectively scanned or used for lexical analysis.
- **Limitations:** Feeds expire rapidly as phishing sites are taken down by web hosts (typically 24–48 hours lifespan).

---

### 2.4 OpenPhish

- **Official Source URL:** `https://openphish.com/feed.txt`
- **Download Method:** Direct HTTP `GET` returning plain text list of active phishing URLs.
- **Registration Requirements:** None for the public Community Feed (updated every 6 hours, capped at ~1,000 active URLs). Premium feed requires enterprise subscription.
- **License:** Open for security research and academic evaluation.
- **Available Metadata:** URL string only in community feed. (No target brand or timestamp in free tier).
- **Visual Evidence:** No screenshots provided.
- **Limitations:** Only provides raw URLs without labels (all entries are assumed phishing) or target brand annotations.

---

### 2.5 URLhaus (abuse.ch)

- **Official Source URL:** `https://urlhaus.abuse.ch/`
- **API Endpoint:** `https://urlhaus-api.abuse.ch/v2/files/exports/<AUTH-KEY>/recent.csv`
- **Authentication Requirement (IMPORTANT):**  
  Keyless public exports (`https://urlhaus.abuse.ch/downloads/csv_recent/`) have been permanently retired by abuse.ch and return **`HTTP 410 Gone`**. Downloading automated dumps now strictly requires a free personal Auth-Key from the [abuse.ch Authentication Portal](https://auth.abuse.ch/).
- **Environment Variable:** `URLHAUS_AUTH_KEY`
- **Available Metadata:** `id`, `dateadded`, `url`, `url_status`, `last_online`, `threat`, `tags`, `urlhaus_link`, `reporter`.
- **Visual Evidence:** No screenshots provided.
- **Limitations:** Primarily focuses on malware distribution payloads, credential harvesting campaigns, and drop sites.

---

### 2.6 Tranco List

- **Official Source URL:** `https://tranco-list.eu/top-1m.csv.zip`
- **Download Method:** Direct HTTP ZIP download.
- **Registration Requirements:** None.
- **License:** Open Access (academic research standard for benign website domain sampling).
- **Available Data:** Ranked list of domains: `rank, domain` (e.g. `1, google.com`, `2, a-msedge.net`).
- **Critical Caution:** **Tranco domains are popular domains, NOT verified legitimate ground-truth URLs.** Many high-ranking domains represent CDNs, parking pages, ad servers, or may host user-generated content and compromised subpaths. Tranco must be strictly used for top-level benign domain discovery and filtered before labeling.

---

## 3. Dataset Mapping to PhishVision AI Model Tasks

| Model Task | Required Dataset | Primary Sources | Secondary / Verification Source |
| :--- | :--- | :--- | :--- |
| **ResNet Screenshot Classifier** | 10k screenshots (5k legit / 5k phish) | Zenodo Records 8041387 & 23477065 (Pre-rendered) | Phase 1 Scanner on validated Tranco/OpenPhish URLs |
| **Siamese CNN Brand Matcher** | 1,000+ brand comparison pairs | Curated Brand Reference Gallery (`data/brands/`) + Zenodo 8041387 | Target brand matches in PhreshPhish/PhishTank |
| **YOLO UI & Logo Detector** | ~1,500 annotated screenshots | Curated screenshots + PhishDecloaker (Zenodo) + Synthetic UI | Manual / semi-automated YOLO `.txt` labeling |
| **U-Net Webpage Segmentation** | ~1,000 semantic masks | Screenshot captures with DOM bounding boxes converted to masks | Phase 1 DOM layout extraction |
| **LSTM / Transformer Redirects** | ~2,000 multi-hop chains | Phase 1 Scanner redirect traces on live feeds (PhishTank/OpenPhish) | Forensic navigation logs |
| **MLP URL Lexical Baseline** | 20,000+ labeled URL records | PhishTank + OpenPhish + URLhaus + Tranco benign samples | PhreshPhish URL records |

---

## 4. Storage & Git Policy Compliance

- Large raw datasets (ZIP archives, raw PNG screenshots, large Parquet files) are stored exclusively under `data/raw/` and `data/processed/`, which are ignored by `.gitignore`.
- Only schemas, manifests (`DATASET_MANIFEST.md`, `manifest_brands.json`), and acquisition scripts are tracked in Git.
- Reproducible MD5 and SHA-256 provenance hashes are generated for every downloaded artifact and logged in `data/metadata/provenance.json`.
