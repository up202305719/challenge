const form = document.getElementById("upload-form");
const resultBox = document.getElementById("result");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("file-input");
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    displayResult(data);

  } catch (err) {
    resultBox.textContent = "Erro: " + err;
  }
});

// Função para mostrar resultados bonitos
function displayResult(data) {
  resultBox.innerHTML = "";

  // Diagnóstico principal
  const pred = document.createElement("div");
  pred.className = `prediction ${data.prediction}`;
  pred.innerText = `Diagnóstico: ${data.prediction}`;
  resultBox.appendChild(pred);

  // Probabilidades detalhadas
  const probList = document.createElement("ul");
  probList.className = "probabilities";

  for (const [label, value] of Object.entries(data.probabilities)) {
    const li = document.createElement("li");
    li.innerText = `${label}: ${value.toFixed(1)}%`;
    probList.appendChild(li);
  }

  resultBox.appendChild(probList);
}
