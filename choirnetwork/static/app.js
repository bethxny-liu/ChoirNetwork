const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const topKInput = document.getElementById("top-k");
const submitBtn = document.getElementById("submit-btn");
const status = document.getElementById("status");
const resultsSection = document.getElementById("results-section");
const resultsList = document.getElementById("results-list");

function renderResults(data) {
  resultsList.replaceChildren();
  document.getElementById("results-meta").textContent =
    `Showing ${data.results.length} hymns for "${data.query}".`;
  data.results.forEach((hymn, index) => {
    const item = document.createElement("li");
    item.className = "result-card";
    const rank = document.createElement("div");
    rank.className = "result-rank";
    rank.textContent = index + 1;
    rank.setAttribute("aria-hidden", "true");
    const body = document.createElement("div");
    body.className = "result-body";
    const title = document.createElement("h3");
    title.textContent = `Hymn ${hymn.label} — ${hymn.title}`;
    body.append(title);
    if (hymn.snippet) {
      const details = document.createElement("details");
      details.className = "result-context";
      const summary = document.createElement("summary");
      summary.textContent = "Why this hymn?";
      const snippet = document.createElement("p");
      snippet.textContent = hymn.snippet;
      details.append(summary, snippet);
      body.append(details);
    }
    const link = document.createElement("a");
    link.className = "result-link";
    link.href = hymn.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "Open hymnal";
    item.append(rank, body, link);
    resultsList.append(item);
  });
  resultsSection.classList.remove("hidden");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  submitBtn.disabled = true;
  submitBtn.textContent = "Searching...";
  status.classList.add("hidden");
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: Number(topKInput.value) }),
    });
    if (!response.ok) throw new Error("Search failed. Please try again.");
    renderResults(await response.json());
  } catch (error) {
    resultsSection.classList.add("hidden");
    status.textContent = error.message;
    status.className = "status error";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Find hymns";
  }
});

topKInput.addEventListener("input", () => {
  document.getElementById("top-k-value").textContent = topKInput.value;
});
