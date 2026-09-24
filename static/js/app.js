const dropzone = document.getElementById("dropzone");
const inputImagen = document.getElementById("input-imagen");
const preview = document.getElementById("preview");
const dropzoneTexto = document.getElementById("dropzone-texto");
const dropzoneIcono = document.getElementById("dropzone-icono");
const form = document.getElementById("form-analisis");
const btnAnalizar = document.getElementById("btn-analizar");
const btnLimpiar = document.getElementById("btn-limpiar");
const estado = document.getElementById("estado");
const resultados = document.getElementById("resultados");
const ensambleClase = document.getElementById("ensamble-clase");
const ensambleConfianza = document.getElementById("ensamble-confianza");
const consensoTexto = document.getElementById("consenso-texto");
const barrasClases = document.getElementById("barras-clases");
const tablaModelosCuerpo = document.getElementById("tabla-modelos-cuerpo");
const recomendacionUrgencia = document.getElementById("recomendacion-urgencia");
const recomendacionAccion = document.getElementById("recomendacion-accion");
const recomendacionDetalle = document.getElementById("recomendacion-detalle");

const ETIQUETAS_URGENCIA = {
  baja: "Urgencia baja",
  media: "Urgencia media",
  alta: "Urgencia alta",
};

function mostrarPreview(archivo) {
  const lector = new FileReader();
  lector.onload = () => {
    preview.src = lector.result;
    preview.hidden = false;
    dropzoneIcono.hidden = true;
    dropzoneTexto.textContent = archivo.name;
  };
  lector.readAsDataURL(archivo);
  btnAnalizar.disabled = false;
}

function limpiar() {
  inputImagen.value = "";
  preview.hidden = true;
  preview.src = "";
  dropzoneIcono.hidden = false;
  dropzoneTexto.textContent = "Arrastra una imagen aquí o haz clic para elegirla";
  btnAnalizar.disabled = true;
  estado.textContent = "";
  resultados.hidden = true;
}

dropzone.addEventListener("dragover", (evento) => {
  evento.preventDefault();
  dropzone.classList.add("dropzone--activa");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dropzone--activa");
});

dropzone.addEventListener("drop", (evento) => {
  evento.preventDefault();
  dropzone.classList.remove("dropzone--activa");
  const archivo = evento.dataTransfer.files[0];
  if (archivo) {
    inputImagen.files = evento.dataTransfer.files;
    mostrarPreview(archivo);
  }
});

inputImagen.addEventListener("change", () => {
  const archivo = inputImagen.files[0];
  if (archivo) {
    mostrarPreview(archivo);
  }
});

btnLimpiar.addEventListener("click", limpiar);

function renderizarResultados(datos) {
  ensambleClase.textContent = datos.ensamble.clase_es;
  ensambleConfianza.textContent = `Confianza del ensamble: ${datos.ensamble.confianza}%`;
  consensoTexto.textContent = `${datos.consenso.votos} de ${datos.consenso.total} modelos coinciden en "${datos.consenso.clase_es}".`;

  const recomendacion = datos.ensamble.recomendacion;
  if (recomendacion) {
    recomendacionUrgencia.textContent = ETIQUETAS_URGENCIA[recomendacion.urgencia] || recomendacion.urgencia;
    recomendacionUrgencia.className = `pill pill--${recomendacion.urgencia}`;
    recomendacionAccion.textContent = recomendacion.accion;
    recomendacionDetalle.textContent = recomendacion.detalle;
  }

  const probabilidades = Object.entries(datos.ensamble.probabilidades).sort(
    (a, b) => b[1] - a[1]
  );

  barrasClases.innerHTML = "";
  for (const [clase, valor] of probabilidades) {
    const item = document.createElement("li");
    item.className = "barra";
    item.innerHTML = `
      <span>${clase}</span>
      <span class="barra__pista"><span class="barra__relleno" style="width: ${valor}%"></span></span>
      <span class="barra__valor">${valor}%</span>
    `;
    barrasClases.appendChild(item);
  }

  tablaModelosCuerpo.innerHTML = "";
  for (const modelo of datos.modelos) {
    const fila = document.createElement("tr");
    fila.innerHTML = `
      <td>${modelo.modelo}</td>
      <td>${modelo.clase_es}</td>
      <td>${modelo.confianza}%</td>
    `;
    tablaModelosCuerpo.appendChild(fila);
  }

  resultados.hidden = false;
}

form.addEventListener("submit", async (evento) => {
  evento.preventDefault();

  const archivo = inputImagen.files[0];
  if (!archivo) {
    return;
  }

  btnAnalizar.disabled = true;
  estado.textContent = "Analizando con las 4 redes neuronales...";
  resultados.hidden = true;

  const datosFormulario = new FormData();
  datosFormulario.append("imagen", archivo);

  try {
    const respuesta = await fetch("/api/predecir", {
      method: "POST",
      body: datosFormulario,
    });

    const datos = await respuesta.json();

    if (!respuesta.ok) {
      throw new Error(datos.error || "No se pudo analizar la imagen.");
    }

    estado.textContent = "";
    renderizarResultados(datos);
  } catch (error) {
    estado.textContent = error.message;
  } finally {
    btnAnalizar.disabled = false;
  }
});
