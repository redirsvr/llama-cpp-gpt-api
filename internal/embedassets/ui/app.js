(function () {
  const API = "/v1/rag";
  const INGEST_TIMEOUT_MS = 180000;

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  let selectedFile = null;
  let overlayDepth = 0;
  let activeAbort = null;

  function showToast(msg, type) {
    const el = $("#toast");
    el.textContent = msg;
    el.className = "toast " + (type || "");
    el.hidden = false;
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => {
      el.hidden = true;
    }, 4500);
  }

  function setOverlay(show, text) {
    const o = $("#overlay");
    if (!o) return;
    if (show) {
      overlayDepth += 1;
    } else {
      overlayDepth = Math.max(0, overlayDepth - 1);
      if (overlayDepth > 0) return;
    }
    o.classList.toggle("is-open", overlayDepth > 0);
    o.setAttribute("aria-hidden", overlayDepth > 0 ? "false" : "true");
    if (text) $("#overlay-text").textContent = text;
    document.body.classList.toggle("overlay-active", overlayDepth > 0);
    setIngestControlsDisabled(overlayDepth > 0);
  }

  function forceHideOverlay() {
    overlayDepth = 0;
    const o = $("#overlay");
    if (o) {
      o.classList.remove("is-open");
      o.setAttribute("aria-hidden", "true");
    }
    document.body.classList.remove("overlay-active");
    setIngestControlsDisabled(false);
  }

  function setIngestControlsDisabled(disabled) {
    ["#btn-upload-file", "#btn-upload-text", "#btn-scan", "#btn-overlay-cancel"].forEach((sel) => {
      const el = $(sel);
      if (el) el.disabled = disabled && sel !== "#btn-overlay-cancel";
    });
    if ($("#btn-overlay-cancel")) {
      $("#btn-overlay-cancel").disabled = !disabled;
    }
  }

  function cancelActiveRequest() {
    if (activeAbort) {
      activeAbort.abort();
      activeAbort = null;
    }
    forceHideOverlay();
    showToast("Операция отменена", "error");
  }

  async function api(path, opts) {
    const timeoutMs = opts?.timeoutMs ?? 30000;
    const parent = opts?.signal;
    const ctrl = new AbortController();
    if (parent) {
      if (parent.aborted) ctrl.abort();
      else parent.addEventListener("abort", () => ctrl.abort(), { once: true });
    }
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);

    const fetchOpts = { ...opts, signal: ctrl.signal };
    delete fetchOpts.timeoutMs;

    let res;
    try {
      res = await fetch(API + path, fetchOpts);
    } catch (e) {
      if (e.name === "AbortError") {
        throw new Error(
          "Превышено время ожидания (" +
            Math.round(timeoutMs / 1000) +
            " с) или операция отменена. Проверьте логи сервера."
        );
      }
      throw new Error(e.message || "Сеть недоступна");
    } finally {
      clearTimeout(timer);
    }

    const ct = res.headers.get("content-type") || "";
    let body = null;
    if (ct.includes("application/json")) {
      body = await res.json();
    } else if (res.status !== 204) {
      body = await res.text();
    }
    if (!res.ok) {
      const err = body?.error?.message || body?.message || res.statusText;
      throw new Error(err || "Ошибка " + res.status);
    }
    return body;
  }

  function ingestDocumentResponse(data) {
    const doc = data?.document;
    if (!doc || !doc.title) {
      throw new Error("Некорректный ответ сервера (нет поля document)");
    }
    return doc;
  }

  async function withIngestOverlay(text, run) {
    const ctrl = new AbortController();
    activeAbort = ctrl;
    setOverlay(true, text);
    try {
      return await run(ctrl.signal);
    } finally {
      activeAbort = null;
      setOverlay(false);
    }
  }

  async function checkStatus() {
    const pill = $("#rag-status");
    try {
      await api("/documents", { timeoutMs: 15000 });
      pill.textContent = "RAG активен";
      pill.className = "status-pill ok";
    } catch (e) {
      pill.textContent = e.message || "RAG недоступен";
      pill.className = "status-pill err";
    }
  }

  function initTabs() {
    $$(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        const name = tab.dataset.tab;
        $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
        $$(".panel").forEach((p) => {
          const on = p.id === "panel-" + name;
          p.classList.toggle("active", on);
          p.hidden = !on;
        });
        if (name === "library") loadDocuments();
      });
    });
  }

  function initDropzone() {
    const zone = $("#dropzone");
    const input = $("#file-input");
    const btn = $("#btn-upload-file");
    const nameEl = $("#file-name");

    zone.addEventListener("click", () => input.click());
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        input.click();
      }
    });

    input.addEventListener("change", () => {
      selectedFile = input.files[0] || null;
      btn.disabled = !selectedFile || overlayDepth > 0;
      if (selectedFile) {
        nameEl.textContent = selectedFile.name + " (" + formatSize(selectedFile.size) + ")";
        nameEl.hidden = false;
        if (!$("#file-title").value) {
          $("#file-title").value = selectedFile.name.replace(/\.[^.]+$/, "");
        }
      } else {
        nameEl.hidden = true;
      }
    });

    ["dragenter", "dragover"].forEach((ev) => {
      zone.addEventListener(ev, (e) => {
        e.preventDefault();
        zone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((ev) => {
      zone.addEventListener(ev, (e) => {
        e.preventDefault();
        zone.classList.remove("dragover");
        if (ev === "drop" && e.dataTransfer.files.length) {
          input.files = e.dataTransfer.files;
          input.dispatchEvent(new Event("change"));
        }
      });
    });

    btn.addEventListener("click", uploadFile);
  }

  function formatSize(n) {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / (1024 * 1024)).toFixed(1) + " MB";
  }

  async function uploadFile() {
    if (!selectedFile || overlayDepth > 0) return;
    const fd = new FormData();
    fd.append("file", selectedFile);
    const title = $("#file-title").value.trim();
    if (title) fd.append("title", title);

    try {
      const data = await withIngestOverlay("Индексация файла…", (signal) =>
        api("/documents/upload", {
          method: "POST",
          body: fd,
          timeoutMs: INGEST_TIMEOUT_MS,
          signal,
        })
      );
      const doc = ingestDocumentResponse(data);
      showToast(
        "Документ «" + doc.title + "» проиндексирован (" + (doc.chunk_count || 0) + " чанков)",
        "success"
      );
      selectedFile = null;
      $("#file-input").value = "";
      $("#file-name").hidden = true;
      $("#btn-upload-file").disabled = true;
    } catch (e) {
      showToast(e.message, "error");
    }
  }

  async function uploadText() {
    if (overlayDepth > 0) return;
    const title = $("#text-title").value.trim() || "Документ";
    const content = $("#text-content").value.trim();
    if (!content) {
      showToast("Введите текст", "error");
      return;
    }
    try {
      const data = await withIngestOverlay("Семантический чанкинг…", (signal) =>
        api("/documents", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title, content }),
          timeoutMs: INGEST_TIMEOUT_MS,
          signal,
        })
      );
      const doc = ingestDocumentResponse(data);
      showToast("«" + doc.title + "»: " + (doc.chunk_count || 0) + " чанков", "success");
      $("#text-content").value = "";
    } catch (e) {
      showToast(e.message, "error");
    }
  }

  async function scanDirs() {
    if (overlayDepth > 0) return;
    try {
      const data = await withIngestOverlay("Сканирование каталогов…", (signal) =>
        api("/documents/scan", { method: "POST", timeoutMs: INGEST_TIMEOUT_MS, signal })
      );
      showToast("Проиндексировано файлов: " + (data.ingested || 0), "success");
    } catch (e) {
      showToast(e.message, "error");
    }
  }

  function formatDate(iso) {
    if (!iso) return "—";
    try {
      return new Date(iso).toLocaleString("ru-RU");
    } catch {
      return iso;
    }
  }

  async function loadDocuments() {
    const list = $("#docs-list");
    const empty = $("#docs-empty");
    list.innerHTML = "";
    try {
      const data = await api("/documents");
      const docs = data.data || [];
      empty.hidden = docs.length > 0;
      if (!docs.length) {
        empty.textContent = "Пока нет документов. Загрузите файл или текст.";
      }
      docs.forEach((doc) => {
        const el = document.createElement("article");
        el.className = "doc-item";
        el.innerHTML =
          "<main>" +
          "<h3>" +
          escapeHtml(doc.title) +
          "</h3>" +
          "<p class=\"doc-meta\">" +
          escapeHtml(doc.id) +
          " · чанков: " +
          (doc.chunk_count || 0) +
          (doc.source_path ? " · " + escapeHtml(doc.source_path) : "") +
          "<br>обновлён: " +
          formatDate(doc.updated_at) +
          "</p></main>";
        const del = document.createElement("button");
        del.type = "button";
        del.className = "btn danger";
        del.textContent = "Удалить";
        del.addEventListener("click", () => deleteDoc(doc.id, doc.title));
        el.appendChild(del);
        list.appendChild(el);
      });
    } catch (e) {
      empty.hidden = false;
      empty.textContent = e.message;
    }
  }

  async function deleteDoc(id, title) {
    if (!confirm("Удалить документ «" + title + "»?")) return;
    try {
      await api("/documents/" + encodeURIComponent(id), { method: "DELETE" });
      showToast("Удалено", "success");
      loadDocuments();
    } catch (e) {
      showToast(e.message, "error");
    }
  }

  async function runSearch() {
    const query = $("#search-query").value.trim();
    if (!query) {
      showToast("Введите запрос", "error");
      return;
    }
    const topK = parseInt($("#search-topk").value, 10) || 8;
    const box = $("#search-results");
    box.innerHTML = "<p class=\"hint\">Поиск…</p>";
    try {
      const data = await api("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK }),
        timeoutMs: 120000,
      });
      const hits = data.results || [];
      if (!hits.length) {
        box.innerHTML = "<p class=\"empty\">Ничего не найдено</p>";
        return;
      }
      box.innerHTML = "";
      hits.forEach((h) => {
        const el = document.createElement("article");
        el.className = "hit";
        el.innerHTML =
          "<header><span>" +
          escapeHtml(h.title) +
          "</span><span>score " +
          (h.score?.toFixed(4) || "—") +
          "</span></header>" +
          "<p>" +
          escapeHtml(h.content) +
          "</p>";
        box.appendChild(el);
      });
    } catch (e) {
      box.innerHTML = "<p class=\"empty\">" + escapeHtml(e.message) + "</p>";
      showToast(e.message, "error");
    }
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  $("#btn-upload-text").addEventListener("click", uploadText);
  $("#btn-scan").addEventListener("click", scanDirs);
  $("#btn-refresh").addEventListener("click", loadDocuments);
  $("#btn-search").addEventListener("click", runSearch);
  $("#btn-overlay-cancel").addEventListener("click", cancelActiveRequest);
  $("#search-query").addEventListener("keydown", (e) => {
    if (e.key === "Escape" && overlayDepth > 0) {
      cancelActiveRequest();
      return;
    }
    if (e.key === "Enter") runSearch();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && overlayDepth > 0) cancelActiveRequest();
  });

  forceHideOverlay();
  initTabs();
  initDropzone();
  checkStatus();
})();
