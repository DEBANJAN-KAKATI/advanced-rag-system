let currentStyle = 'medium';
let conversationId = null;
let currentCitations = [];

document.addEventListener('DOMContentLoaded', () => {
  initDropzone();
  initStyleSelector();
  initChatInput();
  loadDocuments();
});

// Style Selector
function initStyleSelector() {
  const btns = document.querySelectorAll('.style-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentStyle = btn.getAttribute('data-style');
    });
  });
}

// Drag & Drop Upload
function initDropzone() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length) uploadFiles(files);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadFiles(fileInput.files);
  });
}

async function uploadFiles(files) {
  const statusEl = document.getElementById('upload-status');
  const statusText = document.getElementById('upload-status-text');
  statusEl.classList.remove('hidden');
  statusText.textContent = `Uploading ${files.length} document(s)...`;

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    statusEl.classList.add('hidden');
    loadDocuments();
  } catch (err) {
    console.error('Upload error:', err);
    statusText.textContent = 'Upload failed!';
    setTimeout(() => statusEl.classList.add('hidden'), 3000);
  }
}

// Documents Inventory
async function loadDocuments() {
  const docList = document.getElementById('doc-list');
  const docCount = document.getElementById('doc-count');

  try {
    const res = await fetch('/api/documents');
    const data = await res.json();
    docCount.textContent = data.total_count;

    if (data.documents.length === 0) {
      docList.innerHTML = `<div class="empty-docs">No documents uploaded yet. Upload a PDF, DOCX, HTML, or TXT document to begin.</div>`;
      return;
    }

    docList.innerHTML = data.documents.map(doc => `
      <div class="doc-card">
        <div class="doc-info">
          <span class="doc-badge">${doc.file_type}</span>
          <div>
            <div class="doc-name" title="${doc.filename}">${doc.filename}</div>
            <div class="doc-meta">${doc.chunk_count} chunks • ${doc.page_count} page(s)</div>
          </div>
        </div>
        <button class="btn-icon" onclick="deleteDocument('${doc.doc_id}')" title="Delete Document">🗑️</button>
      </div>
    `).join('');
  } catch (err) {
    console.error('Failed to load documents:', err);
  }
}

async function deleteDocument(docId) {
  if (!confirm('Are you sure you want to remove this document from the knowledge base?')) return;
  try {
    await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
    loadDocuments();
  } catch (err) {
    console.error('Delete error:', err);
  }
}

// Chat Input
function initChatInput() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuestion();
    }
  });

  sendBtn.addEventListener('click', sendQuestion);
}

async function sendQuestion(prefilledQuestion) {
  const input = document.getElementById('chat-input');
  const question = prefilledQuestion || input.value.trim();
  if (!question) return;

  input.value = '';
  const welcomeCard = document.querySelector('.welcome-card');
  if (welcomeCard) welcomeCard.remove();

  appendMessage('user', question);
  const loadingRow = appendLoadingMessage();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: question,
        style: currentStyle,
        conversation_id: conversationId
      })
    });

    const data = await res.json();
    loadingRow.remove();

    conversationId = data.conversation_id;
    currentCitations = data.citations;

    appendMessage('assistant', data.answer, data.confidence, data.citations, data.query_expansions, data.suggested_questions || []);
  } catch (err) {
    console.error('Chat error:', err);
    loadingRow.remove();
    appendMessage('assistant', 'Sorry, an error occurred while generating the answer. Please try again.');
  }
}

function appendMessage(role, text, confidence = null, citations = [], expansions = [], suggestedQuestions = []) {
  const container = document.getElementById('chat-container');
  const row = document.createElement('div');
  row.className = `msg-row ${role}`;

  const avatar = role === 'user' ? '👤' : '⚡';
  const htmlContent = typeof marked !== 'undefined' ? marked.parse(text) : text;

  let metaHtml = '';
  if (role === 'assistant') {
    let confClass = `conf-${confidence || 'Medium'}`;
    let citeHtml = citations.map(c => `
      <span class="cite-badge" onclick="openSourceDrawer('${c.doc_id}', '${c.filename}', ${c.page}, \`${escapeQuotes(c.snippet)}\`)">
        📄 ${c.filename} (p.${c.page})
      </span>
    `).join('');

    let expHtml = expansions.length > 1 ? `<div style="font-size:0.7rem; color:#64748b; margin-top:4px;">Multi-query search: ${expansions.join(' | ')}</div>` : '';

    // Build follow-up questions HTML
    let followupHtml = '';
    if (suggestedQuestions && suggestedQuestions.length > 0) {
      followupHtml = `
        <div class="followup-section">
          <div class="followup-label">💡 Keep exploring:</div>
          <div class="followup-chips">
            ${suggestedQuestions.map(q => `<button class="followup-chip" onclick="askFollowup(this)">${q}</button>`).join('')}
          </div>
        </div>
      `;
    }

    metaHtml = `
      <div class="msg-meta">
        <div>
          <span>Confidence: </span>
          <span class="conf-badge ${confClass}">${confidence || 'Medium'}</span>
          ${expHtml}
        </div>
      </div>
      <div class="citations-list">${citeHtml}</div>
      ${followupHtml}
    `;
  }

  row.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-bubble">
      <div>${htmlContent}</div>
      ${metaHtml}
    </div>
  `;

  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function askFollowup(btn) {
  const question = btn.textContent.trim();
  if (question) {
    sendQuestion(question);
  }
}

function appendLoadingMessage() {
  const container = document.getElementById('chat-container');
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  row.innerHTML = `
    <div class="msg-avatar">⚡</div>
    <div class="msg-bubble" style="display:flex; align-items:center; gap:0.5rem;">
      <div class="spinner"></div>
      <span style="color:var(--text-muted); font-size:0.9rem;">Searching context & generating citation-backed response...</span>
    </div>
  `;
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
  return row;
}

function escapeQuotes(str) {
  return (str || '').replace(/'/g, "\\'").replace(/"/g, '&quot;').replace(/\n/g, ' ');
}

// Source Drawer
function openSourceDrawer(docId, filename, page, snippet) {
  const drawer = document.getElementById('source-drawer');
  const overlay = document.getElementById('drawer-overlay');
  const body = document.getElementById('drawer-body');

  body.innerHTML = `
    <div style="background:var(--bg-card); padding:1rem; border-radius:10px; border:var(--glass-border);">
      <h4 style="color:var(--cyan); margin-bottom:0.4rem;">${filename}</h4>
      <div style="font-size:0.75rem; color:var(--text-subtle); margin-bottom:1rem;">Page Number: ${page}</div>
      <h5 style="color:var(--text-main); margin-bottom:0.5rem;">Retrieved Text Passage:</h5>
      <div style="background:rgba(0,0,0,0.3); padding:0.8rem; border-radius:6px; font-family:monospace; font-size:0.85rem; color:var(--text-main); white-space:pre-wrap;">${snippet}</div>
    </div>
  `;

  drawer.classList.remove('hidden');
  overlay.classList.remove('hidden');
}

function closeSourceDrawer() {
  document.getElementById('source-drawer').classList.add('hidden');
  document.getElementById('drawer-overlay').classList.add('hidden');
}
