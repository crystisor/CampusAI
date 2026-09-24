// CampusAI Dashboard Frontend Logic
// Built to Impeccable & UI/UX Pro Max standards

let currentSubjectId = "calculus_1";
let currentSubjectName = "Calculus 1";
let currentDocName = "";
let currentPageNum = 1;
let totalPagesNum = 1;
let isEditing = false;
let subjectToDelete = null;

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    await fetchHealth();
    await loadSubjects();
    setupDropzone();
    setupEventListeners();
    // Poll hardware health every 15s
    setInterval(fetchHealth, 15000);
}

// TOAST NOTIFICATIONS
function showToast(message, type = "success") {
    const container = document.getElementById("toastContainer");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    const iconName = type === "success" ? "check-circle-2" : "alert-triangle";
    toast.innerHTML = `
        <i data-lucide="${iconName}"></i>
        <span>${message}</span>
    `;
    container.appendChild(toast);
    lucide.createIcons();

    setTimeout(() => {
        toast.style.transition = "opacity 0.3s ease, transform 0.3s ease";
        toast.style.opacity = "0";
        toast.style.transform = "translateY(10px)";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// 1. HARDWARE HEALTH & STATUS
async function fetchHealth() {
    try {
        const res = await fetch("/api/system/health");
        if (!res.ok) return;
        const data = await res.json();

        // GPU VRAM
        if (data.gpu) {
            const usedGb = (data.gpu.used_mb / 1024).toFixed(1);
            const totalGb = (data.gpu.total_mb / 1024).toFixed(1);
            document.getElementById("vramText").innerText = `${usedGb} / ${totalGb} GB`;
            document.getElementById("vramBar").style.width = `${data.gpu.utilization_pct}%`;
        }

        // Services
        const ollamaDot = document.querySelector("#ollamaPill .dot");
        if (data.ollama && data.ollama.online) {
            ollamaDot.className = "dot online";
        } else {
            ollamaDot.className = "dot";
        }

        const qdrantDot = document.querySelector("#qdrantPill .dot");
        if (data.qdrant && data.qdrant.online) {
            qdrantDot.className = "dot online";
        } else {
            qdrantDot.className = "dot";
        }
    } catch (err) {
        console.warn("Health check error:", err);
    }
}

// 2. SUBJECTS LISTING & SELECTION
async function loadSubjects() {
    try {
        const res = await fetch("/api/subjects");
        const subjects = await res.json();
        const listEl = document.getElementById("subjectList");
        listEl.innerHTML = "";

        const btnDeleteCurrent = document.getElementById("btnDeleteCurrentSubject");

        if (!subjects || subjects.length === 0) {
            listEl.innerHTML = '<div class="skeleton-item">No subjects yet. Click + to add one.</div>';
            currentSubjectId = "";
            currentSubjectName = "";
            document.getElementById("currentSubjectTitle").innerText = "No Subject Selected";
            if (btnDeleteCurrent) btnDeleteCurrent.style.display = "none";
            document.getElementById("metricDocs").innerText = 0;
            document.getElementById("metricPages").innerText = 0;
            document.getElementById("metricVectors").innerText = 0;
            document.getElementById("subjectChannelTags").innerHTML = '<span class="channel-tag" style="color: #64748B;"><i data-lucide="hash"></i> None</span>';
            clearInspectionStudio();
            lucide.createIcons();
            return;
        }

        if (btnDeleteCurrent) btnDeleteCurrent.style.display = "inline-flex";

        let activeFound = false;
        subjects.forEach(sub => {
            const item = document.createElement("div");
            item.className = `subject-item ${sub.id === currentSubjectId ? "active" : ""}`;
            if (sub.id === currentSubjectId) activeFound = true;

            const channelBadge = sub.channels && sub.channels.length > 0 
                ? `<span class="subject-channel-badge"><i data-lucide="hash"></i>${sub.channels[0]}</span>` 
                : '';

            item.innerHTML = `
                <div class="subject-item-main">
                    <span class="subject-item-title">${sub.name}</span>
                    ${channelBadge}
                </div>
                <div class="subject-item-actions">
                    <button class="btn-delete-subject" title="Delete subject ${sub.name}">
                        <i data-lucide="trash-2"></i>
                    </button>
                </div>
            `;

            item.addEventListener("click", () => switchSubject(sub));

            const delBtn = item.querySelector(".btn-delete-subject");
            if (delBtn) {
                delBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    promptDeleteSubject(sub.id, sub.name);
                });
            }

            listEl.appendChild(item);
        });

        if (!activeFound && subjects.length > 0) {
            switchSubject(subjects[0]);
        } else {
            const current = subjects.find(s => s.id === currentSubjectId);
            if (current) updateSubjectHeader(current);
        }

        lucide.createIcons();
    } catch (err) {
        console.error("Failed to load subjects:", err);
    }
}

function updateSubjectHeader(subject) {
    currentSubjectId = subject.id;
    currentSubjectName = subject.name;
    document.getElementById("currentSubjectTitle").innerText = subject.name;
    document.getElementById("metricDocs").innerText = subject.document_count || 0;
    document.getElementById("metricPages").innerText = subject.pages_count || 0;
    document.getElementById("metricVectors").innerText = subject.vector_count || 0;

    const btnDeleteCurrent = document.getElementById("btnDeleteCurrentSubject");
    if (btnDeleteCurrent) btnDeleteCurrent.style.display = "inline-flex";

    const channelTagsEl = document.getElementById("subjectChannelTags");
    if (subject.channels && subject.channels.length > 0) {
        channelTagsEl.innerHTML = subject.channels.map(ch => `
            <span class="channel-tag"><i data-lucide="hash"></i> ${ch}</span>
        `).join("");
    } else {
        channelTagsEl.innerHTML = '<span class="channel-tag" style="color: #64748B;"><i data-lucide="hash"></i> unbound</span>';
    }
    lucide.createIcons();
}

async function switchSubject(subject) {
    currentSubjectId = subject.id;
    currentSubjectName = subject.name;
    document.querySelectorAll(".subject-item").forEach(el => el.classList.remove("active"));
    const activeItem = [...document.querySelectorAll(".subject-item")].find(el => el.querySelector(".subject-item-title")?.textContent === subject.name);
    if (activeItem) activeItem.classList.add("active");

    updateSubjectHeader(subject);
    currentDocName = "";
    currentPageNum = 1;
    await loadSubjectDocuments();
}

function promptDeleteSubject(id, name) {
    subjectToDelete = { id, name };
    const targetNameEl = document.getElementById("deleteSubjectTargetName");
    if (targetNameEl) targetNameEl.innerText = `"${name}" (${id})`;
    const modal = document.getElementById("modalDeleteSubject");
    if (modal) modal.style.display = "flex";
    lucide.createIcons();
}

// 3. DOCUMENTS & STUDIO INSPECTION
async function loadSubjectDocuments() {
    const docSelect = document.getElementById("docSelect");
    docSelect.innerHTML = '<option value="">Select a document to inspect...</option>';

    if (!currentSubjectId) {
        clearInspectionStudio();
        return;
    }

    try {
        const res = await fetch(`/api/subjects/${currentSubjectId}/documents`);
        const docs = await res.json();

        docs.forEach(doc => {
            const opt = document.createElement("option");
            opt.value = doc.name;
            opt.innerText = `${doc.filename} (${doc.pages_count} pages)`;
            docSelect.appendChild(opt);
        });

        if (docs.length > 0) {
            docSelect.value = docs[0].name;
            currentDocName = docs[0].name;
            totalPagesNum = docs[0].pages_count || 1;
            currentPageNum = 1;
            await loadDocumentPage();
        } else {
            clearInspectionStudio();
        }
    } catch (err) {
        console.error("Failed to load subject documents:", err);
    }
}

async function loadDocumentPage() {
    if (!currentDocName) return;

    document.getElementById("pageNavigator").style.display = "flex";
    document.getElementById("currentPageNum").innerText = currentPageNum;
    document.getElementById("totalPagesNum").innerText = totalPagesNum;

    try {
        const res = await fetch(`/api/subjects/${currentSubjectId}/documents/${currentDocName}/page/${currentPageNum}`);
        if (!res.ok) return;
        const data = await res.json();

        // 1. Update left pane image
        const imgContainer = document.getElementById("pdfViewContainer");
        if (data.image_url) {
            imgContainer.innerHTML = `<img src="${data.image_url}?t=${Date.now()}" alt="PDF Page ${currentPageNum}">`;
        } else {
            imgContainer.innerHTML = `
                <div class="empty-state">
                    <i data-lucide="file-text"></i>
                    <p>No preview scan available for Page ${currentPageNum}</p>
                </div>
            `;
            lucide.createIcons();
        }

        // 2. Update right pane Markdown & KaTeX
        const rawEditor = document.getElementById("rawMarkdownEditor");
        const renderedView = document.getElementById("markdownRenderedView");
        rawEditor.value = data.markdown;

        renderMarkdownContent(data.markdown);
    } catch (err) {
        console.error("Failed to load page data:", err);
    }
}

function renderMarkdownContent(text) {
    const renderedView = document.getElementById("markdownRenderedView");
    // Parse markdown using marked
    const html = marked.parse(text || "");
    renderedView.innerHTML = html;

    // Render KaTeX math in container
    renderMathInElement(renderedView, {
        delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\[", right: "\\]", display: true },
            { left: "\\(", right: "\\)", display: false }
        ],
        throwOnError: false
    });
}

function clearInspectionStudio() {
    document.getElementById("pageNavigator").style.display = "none";
    document.getElementById("pdfViewContainer").innerHTML = `
        <div class="empty-state">
            <i data-lucide="file-search"></i>
            <p>Select or upload a document to inspect scan pages</p>
        </div>
    `;
    document.getElementById("markdownRenderedView").innerHTML = `
        <div class="empty-state">
            <i data-lucide="terminal"></i>
            <p>Extracted formulas and text will render with KaTeX math here</p>
        </div>
    `;
    document.getElementById("rawMarkdownEditor").value = "";
    lucide.createIcons();
}

// 4. DROPZONE & SSE LIVE STEPPER
function setupDropzone() {
    const dropzone = document.getElementById("pdfDropzone");
    const fileInput = document.getElementById("fileInput");

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    if (!currentSubjectId) {
        showToast("Please select or create a subject before uploading documents.", "error");
        return;
    }
    if (!file.name.toLowerCase().endsWith(".pdf")) {
        showToast("Please upload a valid PDF file.", "error");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    // Show stepper
    const stepper = document.getElementById("stepperContainer");
    stepper.style.display = "block";
    document.getElementById("stepperFilename").innerText = file.name;
    document.getElementById("stepperDetail").innerText = "Uploading to CampusAI...";
    document.getElementById("stepperProgressPercent").innerText = "5%";
    resetStepperNodes();

    try {
        const res = await fetch(`/api/subjects/${currentSubjectId}/upload`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            throw new Error(`Upload failed with status ${res.status}`);
        }

        const data = await res.json();
        const taskId = data.task_id;

        // Connect SSE
        listenToIngestionSSE(taskId);
    } catch (err) {
        document.getElementById("stepperDetail").innerText = `Error: ${err.message}`;
        console.error(err);
    }
}

function listenToIngestionSSE(taskId) {
    const evtSource = new EventSource(`/api/ingest/status/${taskId}`);

    const stepOrder = ["step_pdf_split", "step_layout", "step_ocr", "step_markdown", "step_embed", "step_indexed"];

    evtSource.onmessage = (e) => {
        if (!e.data) return;
        const msg = JSON.parse(e.data);

        document.getElementById("stepperProgressPercent").innerText = `${msg.progress}%`;
        document.getElementById("stepperDetail").innerText = msg.detail;

        if (msg.step) {
            const currentStepId = `step_${msg.step}`;
            let currentIdx = stepOrder.indexOf(currentStepId);
            if (currentIdx !== -1) {
                stepOrder.forEach((id, idx) => {
                    const node = document.getElementById(id);
                    if (idx < currentIdx) {
                        node.className = "step-node completed";
                    } else if (idx === currentIdx) {
                        node.className = "step-node active";
                    } else {
                        node.className = "step-node";
                    }
                });
            }
        }

        if (msg.step === "complete" || msg.progress === 100) {
            stepOrder.forEach(id => {
                document.getElementById(id).className = "step-node completed";
            });
            evtSource.close();
            // Refresh subjects and current document view
            setTimeout(async () => {
                await loadSubjects();
                await loadSubjectDocuments();
            }, 1000);
        }

        if (msg.step === "error") {
            evtSource.close();
        }
    };

    evtSource.addEventListener("close", () => {
        evtSource.close();
    });

    evtSource.onerror = () => {
        evtSource.close();
    };
}

function resetStepperNodes() {
    const stepOrder = ["step_pdf_split", "step_layout", "step_ocr", "step_markdown", "step_embed", "step_indexed"];
    stepOrder.forEach(id => {
        document.getElementById(id).className = "step-node";
    });
}

// 5. EVENT LISTENERS & MODALS
function setupEventListeners() {
    // Document selector change
    document.getElementById("docSelect").addEventListener("change", (e) => {
        if (e.target.value) {
            currentDocName = e.target.value;
            currentPageNum = 1;
            loadDocumentPage();
        }
    });

    // Pagination
    document.getElementById("btnPrevPage").addEventListener("click", () => {
        if (currentPageNum > 1) {
            currentPageNum--;
            loadDocumentPage();
        }
    });

    document.getElementById("btnNextPage").addEventListener("click", () => {
        if (currentPageNum < totalPagesNum) {
            currentPageNum++;
            loadDocumentPage();
        }
    });

    // Edit Toggle
    const btnToggleEdit = document.getElementById("btnToggleEdit");
    const btnSaveMarkdown = document.getElementById("btnSaveMarkdown");
    const rawEditor = document.getElementById("rawMarkdownEditor");
    const renderedView = document.getElementById("markdownRenderedView");

    btnToggleEdit.addEventListener("click", () => {
        isEditing = !isEditing;
        if (isEditing) {
            rawEditor.style.display = "block";
            renderedView.style.display = "none";
            btnSaveMarkdown.style.display = "inline-flex";
            document.getElementById("toggleEditText").innerText = "View Rendered";
        } else {
            rawEditor.style.display = "none";
            renderedView.style.display = "block";
            btnSaveMarkdown.style.display = "none";
            document.getElementById("toggleEditText").innerText = "Edit Markdown";
            renderMarkdownContent(rawEditor.value);
        }
    });

    // Save Markdown
    btnSaveMarkdown.addEventListener("click", async () => {
        if (!currentDocName) return;
        try {
            const res = await fetch(`/api/subjects/${currentSubjectId}/documents/${currentDocName}/page/${currentPageNum}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ markdown: rawEditor.value }),
            });
            if (res.ok) {
                btnToggleEdit.click(); // Return to rendered view
            }
        } catch (err) {
            console.error("Save markdown failed:", err);
        }
    });

    // Modal: New Subject
    const modal = document.getElementById("modalNewSubject");
    document.getElementById("btnNewSubject").addEventListener("click", () => {
        modal.style.display = "flex";
    });
    document.getElementById("btnCloseModal").addEventListener("click", () => {
        modal.style.display = "none";
    });
    document.getElementById("btnCancelModal").addEventListener("click", () => {
        modal.style.display = "none";
    });

    // Delete current subject button in top nav
    const btnDeleteCurrent = document.getElementById("btnDeleteCurrentSubject");
    if (btnDeleteCurrent) {
        btnDeleteCurrent.addEventListener("click", () => {
            if (currentSubjectId) {
                promptDeleteSubject(currentSubjectId, currentSubjectName || currentSubjectId);
            }
        });
    }

    // Modal: Delete Subject
    const modalDelete = document.getElementById("modalDeleteSubject");
    const btnCloseDelete = document.getElementById("btnCloseDeleteModal");
    const btnCancelDelete = document.getElementById("btnCancelDeleteModal");
    const btnConfirmDelete = document.getElementById("btnConfirmDeleteSubject");

    if (btnCloseDelete) {
        btnCloseDelete.addEventListener("click", () => {
            if (modalDelete) modalDelete.style.display = "none";
            subjectToDelete = null;
        });
    }

    if (btnCancelDelete) {
        btnCancelDelete.addEventListener("click", () => {
            if (modalDelete) modalDelete.style.display = "none";
            subjectToDelete = null;
        });
    }

    if (btnConfirmDelete) {
        btnConfirmDelete.addEventListener("click", async () => {
            if (!subjectToDelete) return;

            const { id, name } = subjectToDelete;
            btnConfirmDelete.disabled = true;
            const confirmTextEl = document.getElementById("confirmDeleteText");
            const originalText = confirmTextEl ? confirmTextEl.innerText : "Delete Subject";
            if (confirmTextEl) confirmTextEl.innerText = "Deleting...";

            try {
                const res = await fetch(`/api/subjects/${id}`, {
                    method: "DELETE"
                });

                if (res.ok) {
                    if (modalDelete) modalDelete.style.display = "none";
                    showToast(`Subject "${name}" deleted from database and storage.`, "success");
                    if (id === currentSubjectId) {
                        currentSubjectId = "";
                        currentSubjectName = "";
                    }
                    await loadSubjects();
                } else {
                    const errData = await res.json().catch(() => ({}));
                    showToast(`Failed to delete subject: ${errData.detail || res.statusText}`, "error");
                }
            } catch (err) {
                console.error("Delete subject error:", err);
                showToast(`Network error deleting subject: ${err.message}`, "error");
            } finally {
                btnConfirmDelete.disabled = false;
                if (confirmTextEl) confirmTextEl.innerText = originalText;
                subjectToDelete = null;
            }
        });
    }

    // Click outside modal backdrop to close
    window.addEventListener("click", (e) => {
        if (e.target === modal) {
            modal.style.display = "none";
        }
        if (e.target === modalDelete) {
            modalDelete.style.display = "none";
            subjectToDelete = null;
        }
    });

    document.getElementById("formNewSubject").addEventListener("submit", async (e) => {
        e.preventDefault();
        const id = document.getElementById("inputSubjectId").value;
        const name = document.getElementById("inputSubjectName").value;

        const formData = new FormData();
        formData.append("subject_id", id);
        if (name) formData.append("name", name);

        try {
            const res = await fetch("/api/subjects", {
                method: "POST",
                body: formData,
            });
            if (res.ok) {
                modal.style.display = "none";
                document.getElementById("formNewSubject").reset();
                showToast(`Subject "${name || id}" created successfully.`, "success");
                await loadSubjects();
            } else {
                const errData = await res.json().catch(() => ({}));
                showToast(`Failed to create subject: ${errData.detail || res.statusText}`, "error");
            }
        } catch (err) {
            console.error("Failed to create subject:", err);
            showToast(`Network error: ${err.message}`, "error");
        }
    });
}
