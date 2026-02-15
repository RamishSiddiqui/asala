document.addEventListener('DOMContentLoaded', async () => {
  const verifiedCount = document.getElementById('verified-count');
  const unverifiedCount = document.getElementById('unverified-count');
  const statusMessage = document.getElementById('status-message');
  const recentItems = document.getElementById('recent-items');
  const scanBtn = document.getElementById('scan-btn');
  const settingsBtn = document.getElementById('settings-btn');

  // Get current tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  // Request current page stats from content script
  chrome.tabs.sendMessage(tab.id, { type: 'GET_STATS' }, (response) => {
    if (response) {
      updateStats(response.verified || 0, response.unverified || 0);
      
      if (response.items && response.items.length > 0) {
        renderRecentItems(response.items);
        statusMessage.style.display = 'none';
      } else {
        statusMessage.textContent = 'No images or videos found on this page';
      }
    }
  });

  // Scan button handler
  scanBtn.addEventListener('click', () => {
    statusMessage.textContent = 'Scanning...';
    chrome.tabs.sendMessage(tab.id, { type: 'FORCE_SCAN' }, (response) => {
      if (response) {
        updateStats(response.verified || 0, response.unverified || 0);
        renderRecentItems(response.items || []);
      }
    });
  });

  // Settings button handler
  settingsBtn.addEventListener('click', () => {
    chrome.runtime.openOptionsPage?.() || 
    window.open(chrome.runtime.getURL('options.html'));
  });

  function updateStats(verified, unverified) {
    verifiedCount.textContent = verified;
    unverifiedCount.textContent = unverified;
  }

  function renderRecentItems(items) {
    recentItems.innerHTML = '';
    
    items.slice(0, 5).forEach(item => {
      const div = document.createElement('div');
      div.className = `item ${item.status}`;
      
      const icon = item.contentType === 'video' ? '🎥' : '🖼️';
      const statusText = item.status === 'verified' ? '✓ Verified' : 
                        item.status === 'unverified' ? '? Unknown' : '⚠ Error';
      
      div.innerHTML = `
        <div class="item-icon">${icon}</div>
        <div class="item-details">
          <div class="item-title">${truncate(item.url, 40)}</div>
          <div class="item-status">${statusText}${item.confidence ? ` (${item.confidence}%)` : ''}</div>
        </div>
      `;
      
      recentItems.appendChild(div);
    });
  }

  function truncate(str, maxLength) {
    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength) + '...';
  }
});
