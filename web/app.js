'use strict';

const state = {
  style: 'medium',
  conversationId: null,
  selectedDocs: new Set(),
  documents: [],
  busy: false,
  abort: null,
};

document.addEventListener('DOMContentLoaded', () => {
  restorePreferences();
  initDropzone();
  initStyleSelector();
  initChatInput();
  initDrawer();
  document.getElementById('refresh-docs').addEventListener('click', loadDocuments);
  document.getElementById('new-chat-btn').addEventListener('click', newChat);
  loadDocuments();
  loadStatus();
});

// ---------------------------------------------------------------- helpers
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function storageGet(key) {
  try { return localStorage.getItem(key); } catch (_) { return null; }
}

function storageSet(key, value) {
  try { localStorage.setItem(key, value); } catch (_) { /* storage unavailable */ }
}

function pageLabel(c) {
  if (!c.filename || !c.filename.toLowerCase().endsWith('.pdf') || !c.page) return '';
  return c.page_end && c.page_end !== c.page ? `pp. ${c.page}-${c.page_end}` : `p. ${c.page}`;
}

async function apiError(res) {
  try {
    const data = await res.json();
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail)) return data.detail.map(d => d.msg).join('; ');
  } catch (_) { /* not JSON */ }
  return `Request failed (${res.status})`;
}

/** Markdown -> sanitized DOM. Falls back to plain text if the CDN scripts failed to load. */
function renderMarkdown(container, text) {
  if (window.marked && window.DOMPurify) {
    container.innerHTML = DOMPurify.sanitize(marked.parse(text || ''));
  } else {
    container.textContent = text || '';
    container.style.whiteSpace = 'pre-wrap';
  }
}

/** Turns [n] markers in rendered text into clickable citation buttons. */
function linkCitations(container, citationsByNumber) {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
    acceptNode: node => (node.parentElement.closest('code, pre, a') ? NodeFilter.FILTER_REJECT
      : /\[\d+\]/.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP),
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);

  nodes.forEach(node => {
    const frag = document.createDocumentFragment();
    node.nodeValue.split(/(\[\d+\])/).forEach(part => {
      const m = part.match(/^\[(\d+)\]$/);
      const citation = m && citationsByNumber.get(Number(m[1]));
      if (citation) {
        const btn = el('button', 'cite-ref', m[1]);
        btn.type = 'button';
        btn.title = `${citation.filename}${pageLabel(citation) ? ', ' + pageLabel(citation) : ''}`;
        btn.addEventListener('click', () => openSourceDrawer(citation));
        frag.appendChild(btn);
      } else if (part) {
        frag.appendChild(document.createTextNode(part));
      }
    });
    node.replaceWith(frag);
  });
}

// ---------------------------------------------------------------- preferences & status
function restorePreferences() {
  const saved = storageGet('rag.style');
  if (saved && document.querySelector(`.style-btn[data-style="${saved}"]`)) state.style = saved;
  document.querySelectorAll('.style-btn').forEach(b => b.classList.toggle('active', b.dataset.style === state.style));
}

async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const s = await res.json();
    const llm = s.llm_available ? `${s.llm_model}` : 'no LLM (extractive mode)';
    document.getElementById('engine-label').textContent = `${llm} • ${s.vectors_ready ? 'hybrid' : 'keyword'} search`;
    document.getElementById('engine-footer').textContent =
      `Model: ${llm} • Embeddings: ${s.embedding_provider} • Reranker: ${s.reranker}`;
    document.getElementById('drop-subtitle').textContent = `${s.supported_formats} • max ${s.max_upload_mb} MB`;
    const banner = document.getElementById('status-banner');
    banner.textContent = '';
    s.warnings.forEach(w => banner.appendChild(el('div', null, '⚠️ ' + w)));
    banner.classList.toggle('hidden', s.warnings.length === 0);
  } catch (err) {
    console.error('Status error:', err);
  }
}

function initStyleSelector() {
  document.querySelectorAll('.style-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.style = btn.dataset.style;
      storageSet('rag.style', state.style);
    });
  });
}

// ---------------------------------------------------------------- upload
function initDropzone() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

  document.getElementById('browse-btn').addEventListener('click', e => {
    e.stopPropagation();
    fileInput.click();
  });
  ['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, e => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  }));
  dropzone.addEventListener('drop', e => {
    if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadFiles(fileInput.files);
    fileInput.value = '';  // allow re-selecting the same file
  });
}

async function uploadFiles(files) {
  const statusEl = document.getElementById('upload-status');
  const statusText = document.getElementById('upload-status-text');
  const results = document.getElementById('upload-results');
  results.textContent = '';
  statusEl.classList.remove('hidden');
  statusText.textContent = `Processing ${files.length} document(s)... large PDFs can take a minute.`;

  const formData = new FormData();
  Array.from(files).forEach(f => formData.append('files', f));

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    if (!res.ok) throw new Error(await apiError(res));
    const data = await res.json();
    data.forEach(r => {
      const item = el('li', `upload-result ${r.success ? 'ok' : 'fail'}`);
      item.appendChild(el('strong', null, r.filename));
      item.appendChild(el('span', null, ` - ${r.message}`));
      results.appendChild(item);
    });
  } catch (err) {
    results.appendChild(el('li', 'upload-result fail', `Upload failed: ${err.message}`));
  } finally {
    statusEl.classList.add('hidden');
    loadDocuments();
    loadStatus();
  }
}

// ---------------------------------------------------------------- documents
async function loadDocuments() {
  const docList = document.getElementById('doc-list');
  try {
    const res = await fetch('/api/documents');
    if (!res.ok) throw new Error(await apiError(res));
    const data = await res.json();
    state.documents = data.documents;
    document.getElementById('doc-count').textContent = data.total_count;

    // Drop selections for documents that no longer exist
    const ids = new Set(data.documents.map(d => d.doc_id));
    state.selectedDocs.forEach(id => { if (!ids.has(id)) state.selectedDocs.delete(id); });
    updateScopeBadge();

    docList.textContent = '';
    if (data.documents.length === 0) {
      docList.appendChild(el('div', 'empty-docs', 'No documents uploaded yet. Upload a document to begin.'));
      return;
    }
    data.documents.forEach(doc => docList.appendChild(renderDocCard(doc)));
  } catch (err) {
    console.error('Failed to load documents:', err);
  }
}

function renderDocCard(doc) {
  const card = el('label', 'doc-card');
  const checkbox = el('input', 'doc-check');
  checkbox.type = 'checkbox';
  checkbox.checked = state.selectedDocs.has(doc.doc_id);
  checkbox.title = 'Only search this document';
  checkbox.addEventListener('change', () => {
    if (checkbox.checked) state.selectedDocs.add(doc.doc_id); else state.selectedDocs.delete(doc.doc_id);
    updateScopeBadge();
  });

  const info = el('div', 'doc-info');
  info.appendChild(el('span', 'doc-badge', doc.file_type));
  const text = el('div', 'doc-text');
  const name = el('div', 'doc-name', doc.filename);
  name.title = doc.filename;
  text.appendChild(name);
  const pages = doc.file_type === 'pdf' ? ` • ${doc.page_count} page(s)` : '';
  text.appendChild(el('div', 'doc-meta', `${doc.chunk_count} chunks${pages}`));
  info.appendChild(text);

  const del = el('button', 'btn-icon', '🗑️');
  del.type = 'button';
  del.title = 'Delete document';
  del.addEventListener('click', e => {
    e.preventDefault();
    deleteDocument(doc);
  });

  card.append(checkbox, info, del);
  return card;
}

function updateScopeBadge() {
  const badge = document.getElementById('active-doc-badge');
  const n = state.selectedDocs.size;
  if (n === 0) {
    badge.textContent = 'All Documents';
  } else if (n === 1) {
    const doc = state.documents.find(d => state.selectedDocs.has(d.doc_id));
    badge.textContent = doc ? doc.filename : '1 document';
  } else {
    badge.textContent = `${n} documents`;
  }
}

async function deleteDocument(doc) {
  if (!confirm(`Remove "${doc.filename}" from the knowledge base?`)) return;
  try {
    const res = await fetch(`/api/documents/${encodeURIComponent(doc.doc_id)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(await apiError(res));
  } catch (err) {
    alert(`Delete failed: ${err.message}`);
  }
  state.selectedDocs.delete(doc.doc_id);
  loadDocuments();
  loadStatus();
}

// ---------------------------------------------------------------- chat
function initChatInput() {
  const input = document.getElementById('chat-input');
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      sendQuestion();
    }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  });
  document.getElementById('send-btn').addEventListener('click', () => {
    if (state.busy && state.abort) state.abort.abort();
    else sendQuestion();
  });
}

async function newChat() {
  if (state.busy && state.abort) state.abort.abort();
  if (state.conversationId) {
    fetch(`/api/conversations/${encodeURIComponent(state.conversationId)}`, { method: 'DELETE' }).catch(() => {});
  }
  state.conversationId = null;
  const container = document.getElementById('chat-container');
  container.textContent = '';
  const card = el('div', 'welcome-card');
  card.appendChild(el('h2', null, 'New conversation'));
  card.appendChild(el('p', null, 'Ask a question about your documents.'));
  container.appendChild(card);
}

function setBusy(busy) {
  state.busy = busy;
  document.getElementById('send-label').textContent = busy ? 'Stop' : 'Send';
  document.getElementById('send-btn').classList.toggle('stop', busy);
}

function scrollToBottom() {
  const container = document.getElementById('chat-container');
  container.scrollTop = container.scrollHeight;
}

function appendUserMessage(text) {
  const row = el('div', 'msg-row user');
  row.appendChild(el('div', 'msg-avatar', '👤'));
  const bubble = el('div', 'msg-bubble');
  bubble.appendChild(el('div', 'msg-text', text));
  row.appendChild(bubble);
  document.getElementById('chat-container').appendChild(row);
  scrollToBottom();
}

function createAssistantMessage() {
  const row = el('div', 'msg-row assistant');
  row.appendChild(el('div', 'msg-avatar', '⚡'));
  const bubble = el('div', 'msg-bubble');
  const loading = el('div', 'msg-loading');
  loading.appendChild(el('div', 'spinner'));
  const loadingText = el('span', null, 'Searching your documents...');
  loading.appendChild(loadingText);
  const content = el('div', 'msg-content');
  const meta = el('div', 'msg-footer');
  bubble.append(loading, content, meta);
  row.appendChild(bubble);
  document.getElementById('chat-container').appendChild(row);
  scrollToBottom();
  return { row, bubble, loading, loadingText, content, meta };
}

/** Parses a Server-Sent Events stream from a fetch Response, calling onEvent(name, data). */
async function readEventStream(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = 'message';
      const dataLines = [];
      raw.split('\n').forEach(line => {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
      });
      if (dataLines.length) onEvent(event, JSON.parse(dataLines.join('\n')));
    }
  }
}

async function sendQuestion(prefilled) {
  if (state.busy) return;
  const input = document.getElementById('chat-input');
  const question = (prefilled || input.value).trim();
  if (!question) return;

  input.value = '';
  input.style.height = 'auto';
  document.querySelectorAll('.welcome-card').forEach(c => c.remove());

  appendUserMessage(question);
  const msg = createAssistantMessage();
  let answerText = '';
  let citationsByNumber = new Map();
  let renderPending = false;

  const render = () => {
    renderPending = false;
    renderMarkdown(msg.content, answerText);
    linkCitations(msg.content, citationsByNumber);
    scrollToBottom();
  };

  state.abort = new AbortController();
  setBusy(true);
  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        style: state.style,
        conversation_id: state.conversationId,
        document_ids: state.selectedDocs.size ? Array.from(state.selectedDocs) : null,
      }),
      signal: state.abort.signal,
    });
    if (!res.ok) throw new Error(await apiError(res));

    await readEventStream(res, (event, data) => {
      if (event === 'sources') {
        state.conversationId = data.conversation_id;
        citationsByNumber = new Map(data.citations.map(c => [c.number, c]));
        msg.loadingText.textContent = data.citations.length
          ? `Found ${data.citations.length} relevant passage(s). Writing answer...`
          : 'No relevant passages found.';
      } else if (event === 'token') {
        msg.loading.classList.add('hidden');
        answerText += data.text;
        if (!renderPending) {
          renderPending = true;
          requestAnimationFrame(render);
        }
      } else if (event === 'done') {
        state.conversationId = data.conversation_id;
        answerText = data.answer;
        citationsByNumber = new Map(data.citations.map(c => [c.number, c]));
        msg.loading.classList.add('hidden');
        render();
        renderAnswerFooter(msg, data);
      } else if (event === 'error') {
        throw new Error(data.message);
      }
    });
  } catch (err) {
    msg.loading.classList.add('hidden');
    if (err.name === 'AbortError') {
      msg.meta.appendChild(el('div', 'msg-notice', 'Stopped.'));
    } else {
      console.error('Chat error:', err);
      msg.meta.appendChild(el('div', 'msg-error', `Sorry, something went wrong: ${err.message}`));
    }
    if (answerText) render();
  } finally {
    setBusy(false);
    state.abort = null;
    scrollToBottom();
  }
}

function renderAnswerFooter(msg, data) {
  const meta = msg.meta;
  meta.textContent = '';

  if (data.notice) meta.appendChild(el('div', 'msg-notice', '⚠️ ' + data.notice));

  const bar = el('div', 'msg-meta');
  const left = el('div', 'meta-left');
  left.appendChild(el('span', null, 'Confidence: '));
  const conf = el('span', `conf-badge conf-${(data.confidence || 'Low').split(' ')[0]}`, data.confidence);
  left.appendChild(conf);
  if (data.query_expansions && data.query_expansions.length > 1) {
    left.appendChild(el('div', 'meta-expansions', `Searched: ${data.query_expansions.join(' | ')}`));
  }
  bar.appendChild(left);

  const copy = el('button', 'btn-icon copy-btn', 'Copy');
  copy.type = 'button';
  copy.title = 'Copy answer';
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(data.answer);
      copy.textContent = 'Copied';
    } catch (_) {
      copy.textContent = 'Copy failed';
    }
    setTimeout(() => { copy.textContent = 'Copy'; }, 1500);
  });
  bar.appendChild(copy);
  meta.appendChild(bar);

  if (data.citations.length) {
    const list = el('div', 'citations-list');
    data.citations.forEach(c => {
      const badge = el('button', 'cite-badge');
      badge.type = 'button';
      const page = pageLabel(c);
      badge.textContent = `[${c.number}] ${c.filename}${page ? ' (' + page + ')' : ''}`;
      badge.title = c.section && c.section !== 'General' ? c.section : c.filename;
      badge.addEventListener('click', () => openSourceDrawer(c));
      list.appendChild(badge);
    });
    meta.appendChild(list);
  }

  if (data.suggested_questions && data.suggested_questions.length) {
    const section = el('div', 'followup-section');
    section.appendChild(el('div', 'followup-label', '💡 Keep exploring:'));
    const chips = el('div', 'followup-chips');
    data.suggested_questions.forEach(q => {
      const chip = el('button', 'followup-chip', q);
      chip.type = 'button';
      chip.addEventListener('click', () => sendQuestion(q));
      chips.appendChild(chip);
    });
    section.appendChild(chips);
    meta.appendChild(section);
  }
}

// ---------------------------------------------------------------- source drawer
function initDrawer() {
  document.getElementById('close-drawer').addEventListener('click', closeSourceDrawer);
  document.getElementById('drawer-overlay').addEventListener('click', closeSourceDrawer);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeSourceDrawer(); });
}

function openSourceDrawer(c) {
  const body = document.getElementById('drawer-body');
  body.textContent = '';
  const card = el('div', 'source-card');
  card.appendChild(el('h4', 'source-title', `[${c.number}] ${c.filename}`));
  const details = [];
  if (pageLabel(c)) details.push(pageLabel(c));
  if (c.section && c.section !== 'General') details.push(`Section: ${c.section}`);
  details.push(`Relevance: ${Math.round((c.score || 0) * 100)}%`);
  card.appendChild(el('div', 'source-details', details.join(' • ')));
  card.appendChild(el('div', 'source-text', c.text || c.snippet));
  body.appendChild(card);

  document.getElementById('source-drawer').classList.remove('hidden');
  document.getElementById('drawer-overlay').classList.remove('hidden');
}

function closeSourceDrawer() {
  document.getElementById('source-drawer').classList.add('hidden');
  document.getElementById('drawer-overlay').classList.add('hidden');
}
