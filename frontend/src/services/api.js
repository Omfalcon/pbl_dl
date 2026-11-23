/**
 * PhishVision AI — API Service Client (v1)
 *
 * Strictly interfaces with the backend endpoints specified in API_CONTRACTS.md:
 * - POST /api/v1/scan
 * - GET  /api/v1/scan/{scan_id}
 * - GET  /api/v1/scan/{scan_id}/report
 * - GET  /api/v1/scan/{scan_id}/screenshot
 * - GET  /api/v1/health
 * - GET  /api/v1/brands
 * - POST /api/v1/analyze/url
 */

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message, status, code, detail) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code || 'API_ERROR';
    this.detail = detail || null;
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const res = await fetch(url, config);
    
    // Status 425 Too Early (scan in progress)
    if (res.status === 425) {
      const data = await res.json().catch(() => ({}));
      throw new ApiError(data.message || 'Scan is still in progress', 425, 'SCAN_IN_PROGRESS', data);
    }

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new ApiError(
        errorData.message || `Request failed with status ${res.status}`,
        res.status,
        errorData.error_code || 'HTTP_ERROR',
        errorData.detail || null
      );
    }

    return await res.json();
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(err.message || 'Network error occurred while contacting PhishVision API', 0, 'NETWORK_ERROR');
  }
}

/**
 * Submit target URL for scanning
 */
export async function submitScan({ url, fast_mode = false, viewport = { width: 1280, height: 720 }, timeout_seconds = 15 }) {
  return request('/api/v1/scan', {
    method: 'POST',
    body: JSON.stringify({
      url,
      options: {
        fast_mode,
        viewport,
        timeout_seconds,
      },
    }),
  });
}

/**
 * Poll execution status of a scan
 */
export async function getScanStatus(scanId) {
  return request(`/api/v1/scan/${encodeURIComponent(scanId)}`);
}

/**
 * Retrieve completed forensic report
 */
export async function getScanReport(scanId) {
  return request(`/api/v1/scan/${encodeURIComponent(scanId)}/report`);
}

/**
 * Get screenshot endpoint URL
 */
export function getScreenshotUrl(scanId) {
  return `${API_BASE}/api/v1/scan/${encodeURIComponent(scanId)}/screenshot`;
}

/**
 * System health and model availability status
 */
export async function getHealth() {
  return request('/api/v1/health');
}

/**
 * List monitored reference brands
 */
export async function getMonitoredBrands() {
  return request('/api/v1/brands');
}

/**
 * Synchronous lexical URL intelligence (fast path)
 */
export async function analyzeUrlFast(url) {
  return request('/api/v1/analyze/url', {
    method: 'POST',
    body: JSON.stringify({ url }),
  });
}
