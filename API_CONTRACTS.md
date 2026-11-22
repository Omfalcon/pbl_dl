# API_CONTRACTS.md — REST API Contracts & Data Schemas

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Status:** Approved for Baseline Architecture (Phase 0)  
**Base URL:** `/api/v1`  
**Last Updated:** 2026-11-22  

---

## 1. RESTful Endpoints Overview

| Method | Endpoint | Description | Status Codes |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/scan` | Submits a target URL for scanning and risk analysis. | `200 OK`, `202 Accepted`, `400 Bad Request`, `429 Too Many Requests` |
| `GET` | `/api/v1/scan/{scan_id}` | Polls the current status of an ongoing or completed scan. | `200 OK`, `404 Not Found` |
| `GET` | `/api/v1/scan/{scan_id}/report` | Retrieves the full, explainable security report for a scan. | `200 OK`, `404 Not Found`, `425 Too Early` |
| `GET` | `/api/v1/scan/{scan_id}/screenshot` | Returns a static image preview of the captured viewport. | `200 OK`, `404 Not Found` |
| `POST` | `/api/v1/analyze/url` | Synchronous lexical URL and domain intelligence only. | `200 OK`, `400 Bad Request` |
| `POST` | `/api/v1/analyze/brand-similarity` | Compares an uploaded image/logo against reference brands. | `200 OK`, `400 Bad Request` |
| `GET` | `/api/v1/brands` | Lists all monitored reference brands and protected domains. | `200 OK` |
| `GET` | `/api/v1/health` | System health check and model loading state. | `200 OK`, `503 Service Unavailable` |

---

## 2. Core Data Schemas (Pydantic / JSON Schema Equivalent)

### 2.1 Ingress Schemas

#### `ScanRequest`
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ScanRequest",
  "type": "object",
  "required": ["url"],
  "properties": {
    "url": {
      "type": "string",
      "format": "uri",
      "description": "Full HTTP or HTTPS target URL to scan",
      "example": "https://sbi.co.in.attacker.com/login"
    },
    "options": {
      "type": "object",
      "properties": {
        "fast_mode": {
          "type": "boolean",
          "default": false,
          "description": "If true, skips dynamic browser rendering and runs lexical URL models only"
        },
        "viewport": {
          "type": "object",
          "properties": {
            "width": { "type": "integer", "default": 1280 },
            "height": { "type": "integer", "default": 720 }
          }
        },
        "timeout_seconds": {
          "type": "integer",
          "default": 15,
          "minimum": 3,
          "maximum": 30
        }
      }
    }
  }
}
```

---

### 2.2 Forensic Intelligence Schemas

#### `URLIntelligenceReport`
```json
{
  "type": "object",
  "required": ["raw_url", "canonical_url", "hostname", "registrable_domain", "subdomain", "tld"],
  "properties": {
    "raw_url": { "type": "string" },
    "canonical_url": { "type": "string" },
    "hostname": { "type": "string" },
    "registrable_domain": { "type": "string" },
    "subdomain": { "type": "string" },
    "tld": { "type": "string" },
    "is_ip_address": { "type": "boolean" },
    "has_punycode": { "type": "boolean" },
    "decoded_punycode": { "type": ["string", "null"] },
    "has_homoglyphs": { "type": "boolean" },
    "homoglyph_details": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "char": { "type": "string" },
          "unicode_point": { "type": "string" },
          "resembles": { "type": "string" },
          "script": { "type": "string" }
        }
      }
    },
    "entropy": {
      "type": "object",
      "properties": {
        "url_entropy": { "type": "number" },
        "hostname_entropy": { "type": "number" },
        "path_entropy": { "type": "number" }
      }
    },
    "typosquatting": {
      "type": "object",
      "properties": {
        "target_brand": { "type": ["string", "null"] },
        "levenshtein_distance": { "type": ["integer", "null"] },
        "damerau_levenshtein_distance": { "type": ["integer", "null"] },
        "is_suspicious_match": { "type": "boolean" }
      }
    },
    "subdomain_masking_detected": { "type": "boolean" },
    "mlp_lexical_score": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
  }
}
```

#### `VisualIntelligenceReport`
```json
{
  "type": "object",
  "required": ["screenshot_captured", "resnet_phishing_probability"],
  "properties": {
    "screenshot_captured": { "type": "boolean" },
    "screenshot_hash_phash": { "type": ["string", "null"] },
    "resnet_phishing_probability": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "detections": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "class_name": { "type": "string", "enum": ["logo", "login_field", "password_field", "submit_button", "qr_code"] },
          "confidence": { "type": "number" },
          "bbox": {
            "type": "object",
            "properties": {
              "x_min": { "type": "number" },
              "y_min": { "type": "number" },
              "x_max": { "type": "number" },
              "y_max": { "type": "number" }
            }
          }
        }
      }
    },
    "brand_similarity": {
      "type": "object",
      "properties": {
        "matched_brand": { "type": ["string", "null"] },
        "similarity_score": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "reference_asset_id": { "type": ["string", "null"] },
        "is_unauthorized_brand_use": { "type": "boolean" }
      }
    },
    "segmentation_summary": {
      "type": "object",
      "properties": {
        "form_area_percentage": { "type": "number" },
        "header_area_percentage": { "type": "number" }
      }
    }
  }
}
```

#### `BehavioralReport`
```json
{
  "type": "object",
  "required": ["redirect_hops_count", "suspicious_form_actions"],
  "properties": {
    "redirect_hops_count": { "type": "integer" },
    "redirect_chain": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "step": { "type": "integer" },
          "url": { "type": "string" },
          "status_code": { "type": "integer" },
          "domain": { "type": "string" }
        }
      }
    },
    "cross_domain_redirects": { "type": "boolean" },
    "lstm_sequence_score": { "type": ["number", "null"] },
    "transformer_sequence_score": { "type": ["number", "null"] },
    "forms": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "form_id": { "type": "string" },
          "action_url": { "type": "string" },
          "action_registrable_domain": { "type": "string" },
          "is_external_submission": { "type": "boolean" },
          "has_password_field": { "type": "boolean" }
        }
      }
    },
    "external_script_domains": {
      "type": "array",
      "items": { "type": "string" }
    },
    "js_suspicious_patterns": {
      "type": "array",
      "items": { "type": "string" }
    }
  }
}
```

---

### 2.3 Master Assessment & Risk Fusion Schema

#### `ScanReport`
```json
{
  "type": "object",
  "required": ["scan_id", "status", "target_url", "timestamp", "risk_assessment"],
  "properties": {
    "scan_id": { "type": "string", "format": "uuid" },
    "status": { "type": "string", "enum": ["COMPLETED", "FAILED", "IN_PROGRESS", "TIMED_OUT"] },
    "target_url": { "type": "string" },
    "timestamp": { "type": "string", "format": "date-time" },
    "elapsed_seconds": { "type": "number" },
    "risk_assessment": {
      "type": "object",
      "required": ["category", "risk_score", "confidence", "summary"],
      "properties": {
        "category": {
          "type": "string",
          "enum": ["LEGITIMATE", "SUSPICIOUS", "PHISHING", "UNKNOWN / INSUFFICIENT EVIDENCE"]
        },
        "risk_score": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "summary": { "type": "string" },
        "contributing_factors": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "vector": { "type": "string" },
              "severity": { "type": "string", "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"] },
              "description": { "type": "string" },
              "evidence_ref": { "type": "string" }
            }
          }
        }
      }
    },
    "url_intelligence": { "$ref": "#/properties/URLIntelligenceReport" },
    "visual_intelligence": { "$ref": "#/properties/VisualIntelligenceReport" },
    "behavioral_intelligence": { "$ref": "#/properties/BehavioralReport" }
  }
}
```

---

## 3. Standard Error Contract

All error responses adhere strictly to the RFC 7807 Problem Details specification:

```json
{
  "type": "object",
  "required": ["error_code", "message", "timestamp"],
  "properties": {
    "error_code": {
      "type": "string",
      "enum": [
        "INVALID_URL_SCHEME",
        "SSRF_ATTEMPT_BLOCKED",
        "DNS_RESOLUTION_FAILED",
        "BROWSER_TIMEOUT",
        "RATE_LIMIT_EXCEEDED",
        "INTERNAL_INFERENCE_ERROR",
        "RESOURCE_NOT_FOUND"
      ]
    },
    "message": { "type": "string" },
    "detail": { "type": ["string", "object", "null"] },
    "timestamp": { "type": "string", "format": "date-time" }
  }
}
```
