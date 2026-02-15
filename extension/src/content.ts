// Content script for Asala Extension

// Configuration
const CONFIG = {
  verifyOnHover: true,
  showBadges: true,
  badgePosition: 'top-right'
};

// Track processed elements
const processedElements = new WeakSet();

// Initialize
function init() {
  console.log('Asala: Content script loaded');

  // Process existing images
  scanAndProcessImages();

  // Watch for new images
  observeDOM();
}

// Scan page for images and videos
function scanAndProcessImages() {
  const images = document.querySelectorAll('img, video');
  images.forEach(processElement);
}

// Process a single media element
function processElement(element) {
  // Skip if already processed
  if (processedElements.has(element)) return;
  processedElements.add(element);

  const src = element.src || element.currentSrc;
  if (!src) return;

  // Skip data URLs and very small images
  if (src.startsWith('data:')) return;
  const rect = element.getBoundingClientRect();
  if (rect.width < 100 || rect.height < 100) return;

  // Check cache first
  checkCache(src).then(cached => {
    if (cached) {
      addBadge(element, cached.result);
    } else {
      // Verify in background
      verifyInBackground(src, element.tagName.toLowerCase());
    }
  });
}

// Check if result is cached
async function checkCache(url: string): Promise<any> {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(
      { type: 'GET_CACHE', url },
      (response: any) => {
        if (response && response.success) {
          resolve(response.result);
        } else {
          resolve(null);
        }
      }
    );
  });
}

// Verify content in background
async function verifyInBackground(url, contentType) {
  chrome.runtime.sendMessage(
    { type: 'VERIFY_CONTENT', url, contentType },
    response => {
      if (response && response.success) {
        // Find element and add badge
        const elements = document.querySelectorAll(`[src="${url}"]`);
        elements.forEach(el => addBadge(el, response.result));
      }
    }
  );
}

// Add verification badge to element
function addBadge(element, result) {
  if (!CONFIG.showBadges) return;

  // Remove existing badge
  const existing = element.parentElement?.querySelector('.asala-badge');
  if (existing) existing.remove();

  // Create badge
  const badge = document.createElement('div');
  badge.className = 'asala-badge';

  // Set badge content based on result
  if (result.status === 'verified') {
    badge.classList.add('verified');
    badge.innerHTML = '✓ Verified';
    badge.title = `Authenticity: ${result.confidence}%`;
  } else if (result.status === 'unverified') {
    badge.classList.add('unverified');
    badge.innerHTML = '? Unknown';
    badge.title = 'No provenance data found';
  } else {
    badge.classList.add('error');
    badge.innerHTML = '! Error';
    badge.title = result.error || 'Verification failed';
  }

  // Position badge
  element.parentElement.style.position = 'relative';
  badge.style.position = 'absolute';

  if (CONFIG.badgePosition === 'top-right') {
    badge.style.top = '5px';
    badge.style.right = '5px';
  }

  // Add click handler for details
  badge.addEventListener('click', (e) => {
    e.stopPropagation();
    showDetails(element, result);
  });

  element.parentElement.appendChild(badge);
}

// Show verification details
function showDetails(element, result) {
  // Remove existing popup
  const existing = document.querySelector('.asala-popup');
  if (existing) existing.remove();

  const popup = document.createElement('div');
  popup.className = 'asala-popup';

  popup.innerHTML = `
    <div class="asala-popup-header">
      <h3>Content Verification</h3>
      <button class="asala-close">×</button>
    </div>
    <div class="asala-popup-content">
      <div class="status ${result.status}">
        Status: ${result.status.toUpperCase()}
      </div>
      ${result.confidence ? `
        <div class="confidence">
          Confidence: ${result.confidence}%
        </div>
      ` : ''}
      ${result.hasManifest ? `
        <div class="manifest">
          <strong>C2PA Manifest:</strong> Present
        </div>
      ` : ''}
      ${result.error ? `
        <div class="error-detail">
          Error: ${result.error}
        </div>
      ` : ''}
      <div class="timestamp">
        Checked: ${new Date(result.timestamp).toLocaleString()}
      </div>
    </div>
  `;

  // Position popup near element
  const rect = element.getBoundingClientRect();
  popup.style.position = 'fixed';
  popup.style.top = `${rect.bottom + 10}px`;
  popup.style.left = `${rect.left}px`;
  popup.style.zIndex = '999999';

  // Close handler
  popup.querySelector('.asala-close').addEventListener('click', () => {
    popup.remove();
  });

  document.body.appendChild(popup);

  // Close on click outside
  setTimeout(() => {
    document.addEventListener('click', function closePopup(e: MouseEvent) {
      const target = e.target as Node;
      if (!popup.contains(target)) {
        popup.remove();
        document.removeEventListener('click', closePopup);
      }
    });
  }, 100);
}

// Observe DOM for new images
function observeDOM() {
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) {
          const element = node as Element;
          if (element.matches && (element.matches('img') || element.matches('video'))) {
            processElement(element);
          }
          if (element.querySelectorAll) {
            element.querySelectorAll('img, video').forEach(processElement);
          }
        }
      });
    });
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
