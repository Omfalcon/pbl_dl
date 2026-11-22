# TECH_STACK.md — Technology Stack Specification

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Status:** Approved for Baseline Architecture (Phase 0)  
**Last Updated:** 2026-11-22  

---

## 1. Primary Technology Selections

| Layer / Subsystem | Technology | Version | Purpose & Rationale |
| :--- | :--- | :--- | :--- |
| **Language Runtime** | Python | 3.13.3 | Primary backend language; broad ecosystem support for deep learning, async APIs, and security tools. |
| **Deep Learning Framework** | PyTorch | 2.6.0+cu124 | GPU-accelerated tensor computations, neural network training, autograd, and inference pipeline. |
| **Computer Vision Engine** | Torchvision | 0.21.0+cu124 | Pretrained CNN backbones (ResNet-18/50), image transformations, and dataset loaders. |
| **Object Detection & Seg** | Ultralytics YOLO | 8.4.60 | High-performance real-time detection of brand logos, login fields, and instance segmentation. |
| **Machine Learning Toolkit** | scikit-learn | 1.8.0 | Classical ML baselines, evaluation metrics (ROC-AUC, PR-AUC), and probability calibration (Platt/Isotonic). |
| **Image Processing** | OpenCV / Pillow | opencv-python / PIL | Screenshot cropping, bounding-box rendering, perceptual hashing (pHash), and image normalization. |
| **Browser Automation** | Playwright | latest | Headless, isolated dynamic web scanning with anti-SSRF network interception and screenshot rendering. |
| **HTML / DOM Parsing** | BeautifulSoup4 | 4.14.3 | DOM tree inspection, form action extraction, input field discovery, and script source mapping. |
| **Domain Intelligence** | tldextract | 5.3.1 | Accurate registrable domain and effective TLD extraction based on the Public Suffix List (PSL). |
| **Data Manipulation** | Pandas & NumPy | 2.x+ / 2.x+ | Structured tabular data processing, dataset manifests, and matrix operations. |
| **Sequence Models** | Transformers | 4.57.6 | Self-attention / Transformer redirect sequence analysis and comparative modeling. |
| **API Framework** | FastAPI | 0.115+ | High-performance asynchronous REST microservice framework with automatic OpenAPI/Swagger schemas. |
| **ASGI Web Server** | Uvicorn | 0.32+ | Asynchronous event-loop server for FastAPI backend. |
| **Frontend Framework** | React | 18.x / 19.x | Declarative, component-driven user interface for scan submissions and forensic reviews. |
| **Build Tooling & Bundler** | Vite | 5.x+ | Ultra-fast frontend development server and optimized production bundler. |
| **CSS & Design System** | Tailwind CSS | 3.x+ | Utility-first CSS framework for clean, responsive, and modern cybersecurity dashboard aesthetics. |
| **Data Visualization** | Recharts | 2.x+ | Composable charting library for rendering risk radar charts, probability breakdowns, and timelines. |
| **HTTP Client (Client)** | Axios | 1.7+ | Promise-based HTTP client for browser-to-backend communication. |
| **Containerization** | Docker | Engine 26+ | Ephemeral sandbox isolation and production packaging. |

---

## 2. Active Python Environment Details

The project leverages a pre-configured Python virtual environment containing the complete deep learning and computer vision stack:

- **Filesystem Path:** `C:\Users\hp\PycharmProjects\blansole_footscan\.venv`
- **Python Binary:** `C:\Users\hp\PycharmProjects\blansole_footscan\.venv\Scripts\python.exe`
- **Hardware Acceleration:** CUDA 12.4 acceleration verified (`torch.cuda.is_available() == True`).
- **Installed Key Packages:** `torch`, `torchvision`, `ultralytics`, `transformers`, `scikit-learn`, `fastapi`, `uvicorn`, `tldextract`, `beautifulsoup4`, `scipy`, `pandas`, `numpy`.

### Execution Instructions for Terminal / Scripts:
When running Python commands from PowerShell within this workspace, reference the interpreter directly:
```powershell
& "C:\Users\hp\PycharmProjects\blansole_footscan\.venv\Scripts\python.exe" <script.py>
```
*Note: In automated agent tool calls, if executing outside the sandbox boundary is necessary to access this path, `BypassSandbox: true` should be specified.*

---

## 3. Approved Technology Mandate (ADR-001)

- **Official Project Technology:** Python 3.13 (Backend, ML, Math Sandbox, Testing) and React / Vite / Tailwind CSS / JavaScript (Frontend).
- **Excluded Languages & Build Systems:** C, C++, Java, CMake, Make, Maven, and Gradle are permanently excluded from the project under this approved technology decision.
- **Academic Pedagogy Implementation:** The educational goals of the academic curriculum (demonstrating tensor operations, analytical backpropagation, numerical gradient checking, optimization algorithms, and biological learning paradigms) will be strictly satisfied in Phase 3 & 4 through first-principles implementations in pure Python and NumPy. No C/C++/Java will be introduced or maintained.
