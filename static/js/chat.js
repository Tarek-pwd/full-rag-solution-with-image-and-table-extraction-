// ============================================
// MOI Document Intelligence - Chat JS
// ============================================

const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
let isProcessing = false;

// Handle keyboard input
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Auto-resize textarea
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

// Insert prompt from quick actions
function insertPrompt(text) {
    messageInput.value = text;
    messageInput.focus();
}

// Toggle sidebar on mobile
function toggleSidebar() {
    document.querySelector('.sidebar').classList.toggle('open');
}

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isProcessing) return;

    // Clear welcome message
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.remove();

    // Add user message
    addMessage(message, 'user');
    messageInput.value = '';

    // Show typing indicator
    isProcessing = true;
    sendBtn.disabled = true;
    const typingId = showTypingIndicator();

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message }),
        });

        const data = await response.json();
        removeTypingIndicator(typingId);

        // Add assistant response with images and tables
        addMessage(
            data.response || data.text_out || 'No response received.',
            'assistant',
            data.images || [],
            data.tables || []
        );

    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator(typingId);
        addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
    }

    isProcessing = false;
    sendBtn.disabled = false;
}

// Add message to chat
function addMessage(content, type, images = [], tables = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = type === 'user' ? 'You' : 'AI';

    const contentWrapper = document.createElement('div');
    contentWrapper.className = 'message-content';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    // Add text content
    if (type === 'assistant') {
        bubble.innerHTML = formatText(content);
    } else {
        bubble.textContent = content;
    }

    contentWrapper.appendChild(bubble);

    // Add images if present
    if (images && images.length > 0) {
        const imagesDiv = document.createElement('div');
        imagesDiv.className = 'message-images';

        images.forEach((img, i) => {
            const imgWrapper = document.createElement('div');
            imgWrapper.className = 'message-image';

            const imgEl = document.createElement('img');
            imgEl.src = img.src;
            imgEl.alt = img.caption || `Image ${i + 1}`;
            imgEl.onclick = () => openImageModal(img.src, img.caption);
            imgEl.onerror = () => {
                imgWrapper.innerHTML = `<div class="image-error">Image not found</div>`;
            };

            imgWrapper.appendChild(imgEl);

            if (img.caption) {
                const caption = document.createElement('div');
                caption.className = 'message-image-caption';
                caption.textContent = img.caption;
                imgWrapper.appendChild(caption);
            }

            imagesDiv.appendChild(imgWrapper);
        });

        contentWrapper.appendChild(imagesDiv);
    }

    // Add tables if present
    if (tables && tables.length > 0) {
        const tablesDiv = document.createElement('div');
        tablesDiv.className = 'message-tables';

        tables.forEach((tableHtml, i) => {
            const tableWrapper = document.createElement('div');
            tableWrapper.className = 'table-wrapper';

            const tableLabel = document.createElement('div');
            tableLabel.className = 'table-label';
            tableLabel.textContent = `Table ${i + 1}`;

            const tableContent = document.createElement('div');
            tableContent.className = 'table-content';
            tableContent.innerHTML = tableHtml;

            tableWrapper.appendChild(tableLabel);
            tableWrapper.appendChild(tableContent);
            tablesDiv.appendChild(tableWrapper);
        });

        contentWrapper.appendChild(tablesDiv);
    }

    // Add timestamp
    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = new Date().toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    });
    contentWrapper.appendChild(time);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentWrapper);
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
}

// Format text with basic markdown
function formatText(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
}

// Typing indicator
function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// Image modal
function openImageModal(src, caption) {
    document.getElementById('modalImage').src = src;
    document.getElementById('modalCaption').textContent = caption || '';
    document.getElementById('imageModal').classList.add('active');
}

function closeImageModal() {
    document.getElementById('imageModal').classList.remove('active');
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeImageModal();
});