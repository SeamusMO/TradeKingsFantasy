console.log("CONTENT.JS STARTED");

// Listen for startSelection from background
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log("CONTENT RECEIVED MESSAGE:", request.action);

    if (request.action === "startSelection") {
        console.log("STARTING OVERLAY...");
        createOverlay();
    }
});

function createOverlay() {
    // 1. Create the dark overlay
    const overlay = document.createElement('div');
    overlay.id = 'trade-overlay';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100vw';
    overlay.style.height = '100vh';
    overlay.style.background = 'rgba(0,0,0,0.2)';
    overlay.style.zIndex = '9999999999';
    overlay.style.cursor = 'crosshair';
    document.body.appendChild(overlay);

    let isDragging = false;
    let startX, startY;

    const selectionBox = document.createElement('div');
    selectionBox.id = 'trade-selection-box';
    selectionBox.style.position = 'fixed';
    selectionBox.style.border = '2px solid #00ff00';
    selectionBox.style.background = 'rgba(0,255,0,0.15)';
    selectionBox.style.zIndex = '10000000000';
    selectionBox.style.display = 'none';
    overlay.appendChild(selectionBox);

    // Start drag
    overlay.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        selectionBox.style.left = startX + 'px';
        selectionBox.style.top = startY + 'px';
        selectionBox.style.width = '0px';
        selectionBox.style.height = '0px';
        selectionBox.style.display = 'block';
    });

    // Dragging
    overlay.addEventListener('mousemove', (e) => {
        if (!isDragging) return;

        const currentX = e.clientX;
        const currentY = e.clientY;

        const width = currentX - startX;
        const height = currentY - startY;

        selectionBox.style.width = Math.abs(width) + 'px';
        selectionBox.style.height = Math.abs(height) + 'px';
        selectionBox.style.left = (width < 0 ? currentX : startX) + 'px';
        selectionBox.style.top = (height < 0 ? currentY : startY) + 'px';
    });

    // End drag
    overlay.addEventListener('mouseup', () => {
        isDragging = false;

        const rect = selectionBox.getBoundingClientRect();

        document.body.removeChild(overlay);

        if (rect.width > 10 && rect.height > 10) {
            chrome.runtime.sendMessage({
                action: "captureArea",
                coords: {
                    x: rect.x * window.devicePixelRatio,
                    y: rect.y * window.devicePixelRatio,
                    width: rect.width * window.devicePixelRatio,
                    height: rect.height * window.devicePixelRatio
                }
            });
        }
    });
}
