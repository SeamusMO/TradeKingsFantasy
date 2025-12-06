console.log("Background loaded and running...");

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {

    console.log("BACKGROUND RECEIVED MESSAGE:", request.action);

    // ------------------------------------------------------
    // 1. Start Selection
    // ------------------------------------------------------
    if (request.action === "startSelection") {

        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {

            if (!tabs || tabs.length === 0) {
                console.log("No active tab found");
                return;
            }

            const tabId = tabs[0].id;
            console.log("Injecting content.js into tab:", tabId);

            chrome.scripting.executeScript({
                target: { tabId },
                files: ["content.js"]
            })
            .then(() => {
                console.log("Injected content.js successfully, now sending startSelection to it...");

                chrome.tabs.sendMessage(tabId, {
                    action: "startSelection"
                });

            })
            .catch(err => console.error("Injection failed:", err));

        });

        sendResponse({ status: "selection_started" });
        return true;
    }


    // ------------------------------------------------------
    // 2. Handle captureArea from content.js
    // ------------------------------------------------------
    if (request.action === "captureArea") {

        console.log("captureArea received:", request.coords);

        chrome.tabs.captureVisibleTab(null, { format: "png" }, async (dataUrl) => {

            if (!dataUrl) {
                console.error("Failed to capture tab.");
                return;
            }

            console.log("Screenshot captured. Sending to backend...");

            try {
                const response = await fetch("http://127.0.0.1:5000/process-trade", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        image: dataUrl,
                        coords: request.coords
                    })
                });

                const result = await response.json();
                console.log("Backend returned:", result);

                chrome.storage.local.set({ lastResult: result });

                chrome.notifications.create({
                    type: "basic",
                    iconUrl: chrome.runtime.getURL("icon.png"),
                    title: "Trade Analyzer Complete",
                    message: "Your result is ready — open the extension."
                });

            } catch (err) {
                console.error("Backend error:", err);
            }
        });

        sendResponse({ status: "capture_started" });
        return true;
    }
});
