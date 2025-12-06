// When popup opens, load the most recent result
window.addEventListener("DOMContentLoaded", () => {
    const statusDiv = document.getElementById("status");
    const outputDiv = document.getElementById("resultOutput");

    chrome.storage.local.get("lastResult", (res) => {
        console.log("Popup loaded result:", res.lastResult);

        if (!res.lastResult) {
            statusDiv.textContent = "No recent analysis found.";
            outputDiv.textContent = "";
            return;
        }

        const result = res.lastResult;

        statusDiv.textContent = "Last Analysis Result:";

        if (result.status === "success") {
            // Display the advice or fallback
            const advice = result.data?.advice?.Advice || result.data?.advice || "No advice generated.";
            outputDiv.textContent = advice;
            outputDiv.style.color = "black";
        } else {
            outputDiv.textContent = `ERROR: ${result.message}`;
            outputDiv.style.color = "red";
        }
    });
});

// Start capture
document.getElementById("captureBtn").addEventListener("click", () => {
    chrome.runtime.sendMessage({ action: "startSelection" });
    window.close();
});
