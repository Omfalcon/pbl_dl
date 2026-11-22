# ANNOTATION_GUIDELINES.md — PhishVision AI YOLO Object Detection Guidelines

**Module:** M-04 UI Component & Logo Detector  
**Architecture:** Ultralytics YOLO (Phase 7.1)  
**Target Format:** YOLO Normalized Bounding Boxes (`<class_id> <x_center> <y_center> <width> <height>`)  
**Standardized Coordinate Space:** All coordinates normalized to $[0.0, 1.0]$ relative to original image dimensions.  
**Version:** 1.0 (Phase 7.1 Approved)  
**Date:** 2026-11-22  

---

## 1. Class Taxonomy & ID Mapping

| Class ID | Class Name | Category | Primary Function in PhishVision AI |
| :---: | :--- | :--- | :--- |
| **0** | `brand_logo` | Brand Identifier | Localizes brand emblem/wordmark for cropping and forwarding into **M-03 Siamese CNN Brand Matcher**. |
| **1** | `login_form` | Layout Container | Identifies the spatial perimeter / card enclosing authentication inputs for spatial co-occurrence analysis. |
| **2** | `username_field` | Credential Input | Detects account identifier input fields (email, username, phone, account number). |
| **3** | `password_field` | Credential Input | Detects sensitive secret input fields (password, PIN, security code). |
| **4** | `submit_button` | Primary Action | Detects primary form submission triggers ("Sign In", "Log In", "Next", "Submit", "Continue"). |
| **5** | `payment_element` | Transaction Element | Detects financial credential inputs (credit card number, CVV, expiry), payment gateways, and payment QR codes. |

---

## 2. Comprehensive Class Definitions & Annotation Rules

### Class 0: `brand_logo`
- **Definition:** The visual trademark, emblem, monogram, or official wordmark representing an organization, company, or service.
- **In-Scope Examples:**
  - Official brand logos in the top header or upper-left corner (e.g., Google multi-color "G" or wordmark, Microsoft 4-square logo, PayPal monogram, Amazon smile logo, SBI blue circle).
  - Centered logos on standalone authentication modals and splash dialogs.
  - Footer copyright logo marks when visually distinct.
- **Out-of-Scope (DO NOT ANNOTATE):**
  - General decorative photography, hero banners, stock illustration vectors.
  - Social media share icons (small circular Facebook/Twitter/Instagram icons in footer navigation).
  - Browser UI elements, URL address bar text, or favicon icons.
- **Bounding Box Rule:**
  - Draw a tight bounding box around the logo symbol and wordmark together if they form a unified lockup.
  - If the symbol and text are widely separated, annotate the primary visual mark.
  - Include zero or minimal surrounding whitespace (tight crop is essential for M-03 Siamese embedding fidelity).

---

### Class 1: `login_form`
- **Definition:** The visual bounding container, modal card, or panel that groups credential input fields and submission controls together into a coherent authentication interface.
- **In-Scope Examples:**
  - The white or styled card enclosing the login inputs against a darkened/blurred page background.
  - Dedicated login/registration panels in split-screen layouts.
  - Pop-up authentication dialogs / modals.
- **Out-of-Scope (DO NOT ANNOTATE):**
  - The entire webpage canvas when there is no distinct card or grouped container.
  - Generic newsletter sign-up boxes in page footers.
  - Search filter sidebars on e-commerce sites.
- **Bounding Box Rule:**
  - Enclose the entire boundary of the card or container holding the inputs, labels, and submit button.
  - **Hierarchical Nesting:** The `login_form` box will spatially enclose `username_field`, `password_field`, and `submit_button` boxes. This overlap is intentional and required for computing topological co-occurrence.

---

### Class 2: `username_field`
- **Definition:** An interactive single-line text input field designated for the user's identity, email address, phone number, customer ID, or account name.
- **In-Scope Examples:**
  - Text input boxes with placeholder or label text such as: "Email or phone", "Username", "User ID", "Mobile Number", "Apple ID".
  - First-step input box in two-step authentication flows (e.g., Google or Microsoft "Enter your email" screen).
  - Account number fields on banking login portals.
- **Out-of-Scope (DO NOT ANNOTATE):**
  - Site-wide search bars (e.g. Amazon search input, Google search homepage bar).
  - Newsletter / email subscription inputs ("Subscribe to our newsletter").
  - Form fields for shipping address, comments, or promotional discount codes.
  - The text label outside the input box (only annotate the interactive input box itself).
- **Bounding Box Rule:**
  - Tight box around the rectangular border of the input field.
  - Include any internal leading icon (e.g. user silhouette icon) or trailing clear icon inside the input boundary.
  - Exclude external labels positioned above or to the left of the field.

---

### Class 3: `password_field`
- **Definition:** An interactive input field specifically designed for entering confidential authentication credentials (passwords, passcodes, PIN numbers).
- **In-Scope Examples:**
  - Input boxes with placeholder or label text: "Password", "Enter PIN", "Passcode", "Security Key".
  - Input fields displaying masked characters (dots, asterisks) or containing a reveal/hide "eye" toggle icon.
  - OTP (One-Time Password) entry boxes or grouped multi-digit PIN input cells.
- **Out-of-Scope (DO NOT ANNOTATE):**
  - "Forgot password?" hyperlink text buttons (these are links, not input fields).
  - Captcha solution text inputs (annotate as generic input or ignore, not password).
  - Account number or username fields.
- **Bounding Box Rule:**
  - Tight bounding box enclosing the input element border and any integrated password-reveal toggle icon.

---

### Class 4: `submit_button`
- **Definition:** The primary actionable visual button that submits credentials, advances authentication steps, or authorizes a transaction.
- **In-Scope Examples:**
  - Action buttons labeled: "Sign In", "Log In", "Next", "Continue", "Submit", "Verify Account", "Access Account", "Proceed".
  - Full-width call-to-action buttons directly below credential inputs.
  - Primary OAuth federated login buttons (e.g., "Sign in with Google", "Continue with Apple").
- **Out-of-Scope (DO NOT ANNOTATE):**
  - Generic navigation menu links ("Home", "Products", "About Us").
  - Secondary auxiliary links ("Forgot password?", "Help", "Privacy Policy", "Create account").
  - Cookie consent banner accept/reject buttons ("Accept All Cookies").
  - Form reset or cancel buttons.
- **Bounding Box Rule:**
  - Tight bounding box around the colored button shape and its embedded text/icon.

---

### Class 5: `payment_element`
- **Definition:** Visual elements and inputs designed to collect financial credentials, credit/debit card details, or process digital transactions.
- **In-Scope Examples:**
  - Credit/Debit card number input boxes (often labeled with card brand glyphs: Visa, Mastercard).
  - CVV / CVC / Security code input fields.
  - Expiration date selector fields (Month / Year).
  - Payment QR codes (e.g., UPI / WeChat Pay / crypto payment QR codes).
  - Embedded third-party payment checkout widgets or card silhouette previews.
- **Out-of-Scope (DO NOT ANNOTATE):**
  - Ordinary price tags or numerical text displaying a dollar amount ("$19.99").
  - Static non-clickable payment method logos in the footer ("We accept Visa/Mastercard") unless part of an active checkout selector.
- **Bounding Box Rule:**
  - For segmented card inputs (card number, exp, CVV) grouped into a dedicated checkout card, annotate individual financial inputs as `payment_element` or enclose the unified payment widget.

---

## 3. Handling Ambiguities & Edge Cases

| Scenario | Decision & Proper Handling |
| :--- | :--- |
| **Search bar vs Username input** | If the page is a search engine or e-commerce catalog, the prominent search input is NOT a `username_field`. Only annotate text inputs as `username_field` when they appear in an authentication or credential collection context. |
| **Multi-step login (SSO)** | In two-step login flows (e.g. Google Accounts), Step 1 displays only a username field and a "Next" button. Annotate the username field as `username_field` and "Next" as `submit_button`. Do not invent a phantom `password_field`. |
| **Cookie consent banners** | Cookie consent banners frequently appear in bottom overlays. Do NOT annotate their buttons as `submit_button` and do NOT annotate their banners as `login_form`. |
| **Empty or Error Screenshots** | Screenshots showing 404 Not Found, DNS resolution errors, blank white viewports, or parked domain holding pages contain zero target objects. They MUST receive a valid **0-byte empty annotation file** (`.txt`). Do not skip them or invent objects. |
| **Federated Login Buttons** | Buttons like "Sign in with Google" or "Continue with Apple" qualify as `submit_button`. The embedded mini-logo inside the button does NOT require a separate `brand_logo` annotation unless it serves as the primary site branding. |

---

## 4. Bounding Box Format & Normalization Formula

Given an image of width $W$ and height $H$ pixels, and a bounding box with pixel coordinates:
- Left edge: $x_{min}$
- Top edge: $y_{min}$
- Right edge: $x_{max}$
- Bottom edge: $y_{max}$

Compute YOLO normalized coordinates:

$$x_{center} = \frac{x_{min} + x_{max}}{2 \times W}$$

$$y_{center} = \frac{y_{min} + y_{max}}{2 \times H}$$

$$width = \frac{x_{max} - x_{min}}{W}$$

$$height = \frac{y_{max} - y_{min}}{H}$$

Text line format:
```text
<class_id> <x_center:.6f> <y_center:.6f> <width:.6f> <height:.6f>
```

Example (`0042.txt`):
```text
0 0.500000 0.150000 0.220000 0.080000
1 0.500000 0.450000 0.400000 0.500000
2 0.500000 0.350000 0.350000 0.060000
3 0.500000 0.450000 0.350000 0.060000
4 0.500000 0.580000 0.350000 0.070000
```

---

## 5. Quality Assurance & Verification Checklist

Before marking any annotated screenshot as `HUMAN_VERIFIED` in the manifest:
1. **Coordinate Validity:** Check that all $x_c, y_c, w, h \in [0.0, 1.0]$ and no coordinates are negative or NaN.
2. **Dimension Validity:** Verify that $w > 0$ and $h > 0$.
3. **Tightness:** Confirm that bounding boxes tightly hug element boundaries without excessive margins.
4. **Class Accuracy:** Verify that class IDs strictly correspond to the 6-class taxonomy.
5. **No Duplicates:** Ensure no overlapping duplicate boxes of the same class exist (IoU $< 0.90$).
6. **Provenance Tracking:** Set `verification_status="HUMAN_VERIFIED"` and record the reviewer ID in `annotation_manifest.csv`.
