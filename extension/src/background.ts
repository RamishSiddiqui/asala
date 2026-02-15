// Background script for Asala Extension

// Store verification cache
const verificationCache = new Map();

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'VERIFY_CONTENT') {
    verifyContent(request.url, request.contentType)
      .then(result => {
        // Cache result
        verificationCache.set(request.url, {
          result,
          timestamp: Date.now()
        });
        sendResponse({ success: true, result });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    return true; // Keep channel open for async
  }

  if (request.type === 'GET_CACHE') {
    const cached = verificationCache.get(request.url);
    if (cached && Date.now() - cached.timestamp < 300000) { // 5 min cache
      sendResponse({ success: true, result: cached.result });
    } else {
      sendResponse({ success: false, error: 'Not cached or expired' });
    }
    return true;
  }
});

// Verify content from URL
async function verifyContent(url, contentType) {
  try {
    // Fetch content
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch: ${response.status}`);
    }

    // Check for C2PA manifest in headers
    const c2paManifest = response.headers.get('X-C2PA-Manifest');

    // Get content as array buffer
    const arrayBuffer = await response.arrayBuffer();
    // Buffer not available in browser, use Uint8Array instead
    const buffer = new Uint8Array(arrayBuffer);

    // For now, return a simulated result
    // In production, this would use @asala/core
    return {
      url,
      contentType,
      status: c2paManifest ? 'verified' : 'unverified',
      confidence: c2paManifest ? 95 : 0,
      hasManifest: !!c2paManifest,
      manifest: c2paManifest,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error('Verification error:', error);
    return {
      url,
      contentType,
      status: 'error',
      error: error.message,
      timestamp: new Date().toISOString()
    };
  }
}

// Clean up old cache entries periodically
setInterval(() => {
  const now = Date.now();
  for (const [url, data] of verificationCache.entries()) {
    if (now - data.timestamp > 600000) { // 10 minutes
      verificationCache.delete(url);
    }
  }
}, 300000); // Run every 5 minutes
