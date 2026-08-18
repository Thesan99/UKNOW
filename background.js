// ======================================================================
// EDIT THIS: your Hugging Face Space URL (no trailing slash).
// e.g. "https://your-name-ai-detector.hf.space"
// ======================================================================
const SERVER_URL = "https://thesan99-uknow.hf.space"; 
// ======================================================================

// Firefox exposes the API as `browser`, Chrome/Edge as `chrome`.
const api = (typeof browser !== "undefined") ? browser : chrome;

// Create the right-click menu item on images.
api.runtime.onInstalled.addListener(() => {
  api.contextMenus.create({
    id: "detect-ai-image",
    title: "UKNOW 🔍",
    contexts: ["image"],
  });
});

api.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "detect-ai-image" || !info.srcUrl) return;
  try {
    await showPanel(tab.id, { loading: true });

    // Fetch the clicked image, then POST it to the server.
    const imgResp = await fetch(info.srcUrl);
    const blob = await imgResp.blob();
    const form = new FormData();
    form.append("file", blob, "image.png");

    const resp = await fetch(SERVER_URL + "/analyze", {
      method: "POST",
      body: form,
    });
    if (!resp.ok) throw new Error("Server " + resp.status);
    const data = await resp.json();

    await showPanel(tab.id, { data });
  } catch (e) {
    await showPanel(tab.id, { error: String(e) });
  }
});

// Inject a floating result panel into the current page.
function showPanel(tabId, payload) {
  return api.scripting.executeScript({
    target: { tabId },
    func: renderPanel,
    args: [payload],
  });
}

// This function runs IN THE PAGE (not the service worker).
function renderPanel(payload) {
  const ID = "ai-detector-panel";
  document.getElementById(ID)?.remove();

  const box = document.createElement("div");
  box.id = ID;
  box.style.cssText = [
    "position:fixed", "top:16px", "right:16px", "z-index:2147483647",
    "width:310px", "background:#fff", "color:#111",
    "font:13px/1.4 system-ui,sans-serif", "border-radius:12px",
    "box-shadow:0 8px 28px rgba(0,0,0,.28)", "padding:16px", "border:1px solid #e5e7eb",
  ].join(";");

  const close = document.createElement("div");
  close.textContent = "×";
  close.style.cssText =
    "position:absolute;top:8px;right:12px;cursor:pointer;font-size:18px;color:#888";
  close.onclick = () => box.remove();
  box.appendChild(close);

  if (payload.loading) {
    box.insertAdjacentHTML("beforeend",
      "<b>🔍 Analyzing image…</b><div style='color:#666;margin-top:6px'>Contacting detector…</div>");
  } else if (payload.error) {
    box.insertAdjacentHTML("beforeend",
      "<b style='color:#b91c1c'>Error</b><div style='margin-top:6px;color:#666'>" +
      payload.error + "</div><div style='margin-top:6px;color:#999'>Is the server URL set and the Space awake?</div>");
  } else {
    const d = payload.data;
    
    // Determine bar color based on score (0-100 real score)
    let color = "#16a34a"; // Green
    if (d.score <= 20) color = "#dc2626";       // Red
    else if (d.score <= 50) color = "#ea580c";  // Orange
    else if (d.score <= 75) color = "#d97706";  // Yellow/Amber

    box.insertAdjacentHTML("beforeend",
      // Emoji + Band Title
      "<div style='font-weight:700;font-size:15px;color:" + color + ";padding-right:15px'>" +
        (d.emoji || "🔍") + " " + (d.band || "Analysis Result") + 
      "</div>" +

      // Score / 100
      "<div style='margin:6px 0;color:#444;font-size:13px'>Real Score: <b>" + d.score + "/100</b></div>" +

      // Progress Bar
      "<div style='height:8px;background:#eee;border-radius:4px;overflow:hidden;margin-bottom:10px'>" +
        "<div style='height:100%;width:" + d.score + "%;background:" + color + "'></div>" +
      "</div>" +

      // Heatmap Image
      "<img alt='heatmap' style='width:100%;border-radius:8px' src='data:image/png;base64," +
        d.heatmap_png_base64 + "'/>" +

      // Advice Box (Placed above Region Explanation)
      "<div style='margin-top:10px;padding:8px;background:#f9fafb;border-radius:6px;border:1px solid #f3f4f6;color:#666;font-size:12px'>" + 
        "💡 " + d.advice + 
      "</div>" +

      // Region Explanation (Placed below Advice Box)
      "<div style='margin-top:10px;color:#333;font-weight:500'>" + d.explanation + "</div>"
    );
  }
  document.body.appendChild(box);
}