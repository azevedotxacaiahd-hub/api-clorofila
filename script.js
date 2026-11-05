const API_URL = "http://127.0.0.1:5000/predict";

const imageInput = document.querySelector("#image-input");
const uploadArea = document.querySelector("#upload-area");
const previewWrap = document.querySelector("#preview-wrap");
const imagePreview = document.querySelector("#image-preview");
const fileName = document.querySelector("#file-name");
const changeImageButton = document.querySelector("#change-image-button");
const analyzeButton = document.querySelector("#analyze-button");
const emptyState = document.querySelector("#empty-state");
const loadingState = document.querySelector("#loading-state");
const errorMessage = document.querySelector("#error-message");
const resultsContent = document.querySelector("#results-content");
const resultGrid = document.querySelector("#result-grid");
const statusBanner = document.querySelector("#status-banner");
const statusIcon = document.querySelector("#status-icon");
const statusValue = document.querySelector("#status-value");
const interpretation = document.querySelector("#interpretation");

let selectedImage = null;
let previewUrl = null;

changeImageButton.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files;
  if (!file) return;
  imageInput.value = "";

  clearResults();
  emptyState.hidden = false;
  if (!file.type.startsWith("image/")) {
    selectedImage = null;
    imageInput.value = "";
    analyzeButton.disabled = true;
    previewWrap.hidden = true;
    uploadArea.hidden = false;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    showError("Selecione um ficheiro de imagem válido.");
    return;
  }

  selectedImage = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  imagePreview.alt = `Pré-visualização de ${file.name}`;
  fileName.textContent = file.name;
  previewWrap.hidden = false;
  uploadArea.hidden = true;
  analyzeButton.disabled = false;
});

analyzeButton.addEventListener("click", async () => {
  if (!selectedImage) return;

  clearResults();
  loadingState.hidden = false;
  analyzeButton.disabled = true;
  imageInput.disabled = true;

  const formData = new FormData();
  formData.append("image", selectedImage);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("A API devolveu uma resposta inválida. Tente novamente.");
    }

    if (!response.ok) {
      throw new Error(data.error || `Não foi possível analisar a imagem (erro ${response.status}).`);
    }
    if (!data || !data.success || !data.resultados || typeof data.resultados !== "object") {
      throw new Error(data.error || "A resposta da API não contém os resultados esperados.");
    }

    renderResults(data.resultados);
  } catch (error) {
    showError(error instanceof TypeError
      ? "Não foi possível ligar à API. Confirme se está disponível em 127.0.0.1:5000."
      : error.message);
  } finally {
    loadingState.hidden = true;
    analyzeButton.disabled = !selectedImage;
    imageInput.disabled = false;
  }
});

function clearResults() {
  emptyState.hidden = true;
  loadingState.hidden = true;
  errorMessage.hidden = true;
  resultsContent.hidden = true;
  interpretation.hidden = true;
  resultGrid.replaceChildren();
}

function showError(message) {
  emptyState.hidden = true;
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function renderResults(resultados) {
  const status = resultados.status;
  const statusClass = status === "deficiencia"
    ? "status-deficiency"
    : status === "saudavel"
      ? "status-healthy"
      : "status-moderate";
  const statusText = status === "deficiencia"
    ? "Possível deficiência"
    : status === "saudavel"
      ? "Saudável"
      : status === "moderado"
        ? "Moderado"
        : status || "Não identificado";

  statusBanner.className = `status-banner ${statusClass}`;
  statusValue.textContent = statusText;
  statusIcon.innerHTML = status === "deficiencia"
    ? '<svg viewBox="0 0 24 24" fill="none"><path d="M12 8v5m0 3h.01M10.3 3.8 2.8 17a2 2 0 0 0 1.7 3h15a2 2 0 0 0 1.7-3l-7.5-13.2a2 2 0 0 0-3.4 0Z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    : '<svg viewBox="0 0 24 24" fill="none"><path d="m5 12 4.5 4.5L19 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  const cards = [
    { label: "Clorofila A", value: resultados.clorofila_a, unit: "" },
    { label: "Clorofila B", value: resultados.clorofila_b, unit: "" },
    { label: "Carotenoides", value: resultados.carotenoides, unit: "" },
    { label: "Razão A/B", value: resultados.razao_chl_a_b, unit: "" },
  ];

  for (const card of cards) {
    const element = document.createElement("div");
    element.className = "result-card";

    const label = document.createElement("span");
    label.className = "result-card-label";
    label.textContent = card.label;

    const value = document.createElement("strong");
    value.className = "result-card-value";
    value.textContent = card.value ?? "—";

    if (card.unit) {
      const unit = document.createElement("span");
      unit.className = "result-card-unit";
      unit.textContent = card.unit;
      value.append(unit);
    }

    element.append(label, value);
    resultGrid.append(element);
  }

  if (resultados.interpretacao) {
    interpretation.textContent = resultados.interpretacao;
    interpretation.hidden = false;
  }

  resultsContent.hidden = false;
}
