"""
GenAI RAG Web Application
==========================
A production-ready Retrieval-Augmented Generation system with:
- Vector search using FAISS
- Local LLM generation with TinyLlama
- Streaming responses via Server-Sent Events
- AI image generation with Stable Diffusion XL
- Interactive web interface

Server: http://127.0.0.1:8080
Endpoints:
  - GET  /           : Web interface
  - POST /api/query  : Standard query (instant response)
  - POST /stream     : Streaming query (word-by-word)
  - POST /answer     : Alias for /api/query
"""
import os
import json
import re
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, Response
from src.genai_app.index.index_manager import IndexManager
from src.genai_app.rag.llm_query import RAGQuerySystem
import requests
from collections import Counter
import re
import math
from collections import Counter, defaultdict

app = Flask(__name__)

# Global variables
index_manager = None
rag_system = None

# Hugging Face API configuration

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "INSERT_YOUR_TOKEN_HERE")  # Set via environment variable: export HF_API_TOKEN=your_token_here

_DEFAULT_STOP = set("""
a an and are as at be been but by for from has have he her him his i in is it its itself me my our ours
she the their them they this those to was we were what when where which who will with you your
on into over under above below because while during across through between among within without about after before
can could should would may might must do does did doing done not no nor own same so than too very up down out off
any each few more most other some such only both just being also ever every per via using used based approach paper
method methods system systems result results study studies show shows shown propose present presents presented
include includes including included overall time field research researchers work works data model models
""".split())

def extract_key_concepts(
    text: str,
    top_n: int = 10,
    min_ngram: int = 1,
    max_ngram: int = 3,
    stopwords: set | None = None,
    return_scores: bool = False,
):
    """
    General-purpose concept/keyword extractor.
    - Pure Python (no deps)
    - Works on arbitrary English prose (abstracts, docs, news, etc.)
    - Returns ranked n-grams (1..max_ngram)

    Scoring:
      score = frequency * (len(ngram)^1.25) + 0.15 * position_bonus
      - position_bonus favors early-appearing phrases (first third of text)

    Dedup:
      Keeps longer, more specific phrases; removes shorter ones that are strict substrings of already-selected items.
    """
    if not text:
        return [] if not return_scores else []

    stop = _DEFAULT_STOP if stopwords is None else set(stopwords)

    # Basic tokenization (keep internal hyphens/apostrophes), lowercase
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())

    # Light normalization (strip simple plurals/gerunds)
    def _norm(w: str) -> str:
        if len(w) > 3 and w.endswith('s'):
            w = w[:-1]
        if len(w) > 4 and (w.endswith('ing') or w.endswith('ed')):
            w = re.sub(r'(ing|ed)$', '', w)
        return w

    words = [_norm(w) for w in tokens]

    # Build n-gram candidates
    ngrams = []
    n = len(words)
    for L in range(min_ngram, max_ngram + 1):
        for i in range(n - L + 1):
            gram = words[i:i + L]
            # filter if any token is a stopword
            if any(g in stop or len(g) < 3 for g in gram):
                continue
            # boundary must be content words
            if gram[0] in stop or gram[-1] in stop:
                continue
            phrase = " ".join(gram)
            ngrams.append((phrase, i))  # keep position for a tiny bonus

    if not ngrams:
        return [] if not return_scores else []

    # Frequency and position bonuses
    freq = Counter([p for p, _ in ngrams])
    first_pos = defaultdict(lambda: 10**9)
    for p, i in ngrams:
        if i < first_pos[p]:
            first_pos[p] = i

    # Score: freq * length^1.25 + early-position bonus
    total_tokens = max(1, len(words))
    third = total_tokens // 3 or 1
    scores = {}
    for p, c in freq.items():
        length_weight = (len(p.split()) ** 1.25)
        pos_bonus = 0.15 if first_pos[p] <= third else 0.0
        scores[p] = c * length_weight + pos_bonus

    # Rank
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Deduplicate: prefer longer phrases; drop items that are substrings of already-chosen
    selected: list[tuple[str, float]] = []
    selected_texts: list[str] = []
    for phrase, score in ranked:
        if any(phrase != s and phrase in s for s in selected_texts):
            continue  # phrase is contained in an already selected longer one
        selected.append((phrase, score))
        selected_texts.append(phrase)
        if len(selected) >= top_n:
            break

    return selected if return_scores else [p for p, _ in selected]


def _extract_key_concepts(text, top_n=8):
    """
    Extract key concepts/keywords from text.
    Returns most important words (excluding common stop words).
    """
    # Simple stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
                  'this', 'that', 'these', 'those', 'it', 'its', 'which', 'what', 'how'}
    
    # Extract words (alphanumeric only)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter stop words and count
    filtered_words = [w for w in words if w not in stop_words]
    word_counts = Counter(filtered_words)
    
    # Get top N keywords
    top_keywords = [word for word, count in word_counts.most_common(top_n)]
    
    return top_keywords


def generate_image_prompt(answer, key_concepts, query):
    """
    Generate a descriptive prompt for image generation based on the answer.
    """
    # Create a concise prompt focusing on visualization
    concepts_str = ', '.join(key_concepts[:5])
    
    prompt = f"A professional scientific diagram illustrating {concepts_str}. "
    prompt += "Technical visualization with clear labels, modern design, educational style, "
    prompt += "high quality, detailed, informative."
    
    return prompt


def generate_image_from_hf(prompt):
    """
    Generate image using Hugging Face Inference API (Stable Diffusion XL).
    Returns base64 encoded image or None if failed.
    """
    if not HF_API_TOKEN:
        return None, "No Hugging Face API token provided. Set HF_API_TOKEN environment variable."
    
    # Use Stable Diffusion XL (tested and working)
    model_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    model_name = "stable-diffusion-xl-base-1.0"
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    print(f"🎨 Generating image with {model_name}...")
    
    try:
        response = requests.post(
            model_url, 
            headers=headers, 
            json={"inputs": prompt},
            timeout=60
        )
        
        if response.status_code == 200:
            # Success! Convert image bytes to base64
            import base64
            image_b64 = base64.b64encode(response.content).decode('utf-8')
            print(f"✓ Image generated successfully!")
            return f"data:image/png;base64,{image_b64}", None
        
        elif response.status_code == 503:
            # Model is loading
            error_msg = "Model is loading. Please try again in 30-60 seconds."
            print(f"⏳ {error_msg}")
            return None, error_msg
        
        elif response.status_code == 500:
            # Server error
            error_msg = "Server error. API might be temporarily down."
            print(f"❌ {error_msg}")
            return None, error_msg
        
        elif response.status_code == 404:
            # Model not found or access denied
            error_msg = "Model not accessible. Check API token permissions."
            print(f"❌ {error_msg}")
            return None, error_msg
        
        else:
            error_msg = f"Error {response.status_code}: {response.text[:200]}"
            print(f"❌ {error_msg}")
            return None, error_msg
            
    except requests.exceptions.Timeout:
        error_msg = "Request timed out. Model might be overloaded."
        print(f"⏱️ {error_msg}")
        return None, error_msg
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Exception: {error_msg}")
        return None, error_msg


# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>GenAI RAG Query System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .query-box {
            margin: 20px 0;
        }
        textarea {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            resize: vertical;
        }
        button {
            background: #007bff;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .results {
            margin-top: 30px;
        }
        .answer-box, .context-box, .json-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #007bff;
        }
        .answer-box h3, .context-box h3, .json-box h3 {
            margin-top: 0;
            color: #007bff;
        }
        .citation {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .citation strong {
            color: #007bff;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #f5c6cb;
        }
        pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .stream-option {
            margin: 15px 0;
            padding: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        .stream-option input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
            margin: 0;
        }
        .stream-option label {
            cursor: pointer;
            font-size: 15px;
            margin: 0;
            user-select: none;
        }
        .image-box {
            background: #f0f8ff;
            padding: 20px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #17a2b8;
        }
        .image-box h3 {
            margin-top: 0;
            color: #17a2b8;
        }
        .key-concepts {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .key-concepts span {
            display: inline-block;
            background: #e3f2fd;
            padding: 5px 12px;
            margin: 3px;
            border-radius: 15px;
            font-size: 14px;
            color: #1976d2;
        }
        .image-prompt {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #ddd;
            font-style: italic;
            color: #555;
        }
        .generated-image {
            text-align: center;
            margin: 15px 0;
        }
        .generated-image img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>GenAI Demo</h1>
        <p style="text-align: center; color: #666;">Ask questions about research papers</p>
        
        <div class="query-box">
            <textarea id="queryInput" rows="4" placeholder="What does the Transformer architecture improve compared to RNNs?"></textarea>
            <div class="stream-option">
                <input type="checkbox" id="streamCheckbox">
                <label for="streamCheckbox">Enable Streaming (answer appears word-by-word)</label>
            </div>
            <div class="stream-option">
                <input type="checkbox" id="imageCheckbox">
                <label for="imageCheckbox">🖼️ Generate Image (with key concepts & prompt)</label>
            </div>
            <button id="submitBtn">Answer</button>
        </div>
        
        <div id="results" class="results"></div>
    </div>
    
    <script>
        console.log('✓ Script loaded successfully');
        
        function submitQuery() {
            console.log('✓ submitQuery called');
            const query = document.getElementById('queryInput').value.trim();
            const resultsDiv = document.getElementById('results');
            const submitBtn = document.getElementById('submitBtn');
            const useStream = document.getElementById('streamCheckbox').checked;
            const generateImage = document.getElementById('imageCheckbox').checked;
            
            console.log('Query:', query, 'Stream:', useStream, 'Image:', generateImage);
            
            if (!query) {
                alert('Please enter a question');
                return;
            }
            
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';
            resultsDiv.innerHTML = '<div class="loading">🔍 Searching documents and generating answer...</div>';
            
            if (useStream) {
                handleStreamingQuery(query, resultsDiv, submitBtn, generateImage);
            } else {
                handleNormalQuery(query, resultsDiv, submitBtn, generateImage);
            }
        }
        
        function handleNormalQuery(query, resultsDiv, submitBtn, generateImage) {
            fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query, k: 3, generate_image: generateImage })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultsDiv.innerHTML = '<div class="error">Error: ' + data.error + '</div>';
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                resultsDiv.innerHTML = '<div class="error">Error: ' + error.message + '</div>';
            })
            .finally(() => {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Answer';
            });
        }
        
        function handleStreamingQuery(query, resultsDiv, submitBtn, generateImage) {
            fetch('/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query, k: 3, generate_image: generateImage })
            })
            .then(function(response) {
                var reader = response.body.getReader();
                var decoder = new TextDecoder();
                var buffer = '';
                var metadata = null;
                var answerText = '';
                var streamDone = false;
                var imageData = null;
                
                function processStream() {
                    return reader.read().then(function(result) {
                        if (result.done) {
                            // Stream finished - now display complete results
                            if (streamDone) {
                                var completeData = {
                                    query: metadata.query,
                                    answer: answerText,
                                    sources: metadata.sources,
                                    citations: metadata.citations,
                                    num_sources: metadata.num_sources
                                };
                                
                                // Add image data if available
                                if (imageData) {
                                    completeData.image_data = imageData;
                                }
                                
                                displayResults(completeData);
                            }
                            
                            submitBtn.disabled = false;
                            submitBtn.textContent = 'Answer';
                            return;
                        }
                        
                        buffer = buffer + decoder.decode(result.value, {stream: true});
                        var lines = buffer.split('\\n');
                        buffer = lines.pop() || '';
                        
                        for (var i = 0; i < lines.length; i++) {
                            var line = lines[i];
                            if (line.indexOf('data: ') === 0) {
                                try {
                                    var data = JSON.parse(line.slice(6));
                                    
                                    if (data.type === 'metadata') {
                                        metadata = data;
                                        displayStreamStart(metadata, resultsDiv);
                                    } else if (data.type === 'token') {
                                        answerText = answerText + data.content;
                                        updateStreamAnswer(answerText);
                                    } else if (data.type === 'done') {
                                        streamDone = true;
                                        // Don't display yet if image generation is enabled
                                        if (!generateImage) {
                                            displayStreamComplete(metadata, answerText, resultsDiv);
                                        }
                                    } else if (data.type === 'image_data') {
                                        // Save image data to include in final display
                                        imageData = {
                                            key_concepts: data.key_concepts,
                                            image_prompt: data.image_prompt,
                                            image_url: data.image_url,
                                            error: data.error
                                        };
                                    } else if (data.type === 'error') {
                                        resultsDiv.innerHTML = '<div class="error">Error: ' + data.content + '</div>';
                                        submitBtn.disabled = false;
                                        submitBtn.textContent = 'Answer';
                                        return;
                                    }
                                } catch (e) {
                                    console.error('Parse error:', e, 'Line:', line);
                                }
                            }
                        }
                        
                        return processStream();
                    });
                }
                
                return processStream();
            })
            .catch(function(error) {
                resultsDiv.innerHTML = '<div class="error">Error: ' + error.message + '</div>';
                submitBtn.disabled = false;
                submitBtn.textContent = 'Answer';
            });
        }
        
        function displayStreamStart(metadata, resultsDiv) {
            let html = '<div class="answer-box">';
            html += '<h3>Answer</h3>';
            html += '<p id="streamAnswer"><em>Generating...</em></p>';
            html += '</div>';
            resultsDiv.innerHTML = html;
        }
        
        function updateStreamAnswer(text) {
            const answerEl = document.getElementById('streamAnswer');
            if (answerEl) {
                answerEl.textContent = text;
            }
        }
        
        function displayStreamComplete(metadata, answer, resultsDiv) {
            const data = {
                query: metadata.query,
                answer: answer,
                sources: metadata.sources,
                citations: metadata.citations,
                num_sources: metadata.num_sources
            };
            displayResults(data);
        }
        
        function addImageDataToResults(imageData) {
            // Find the results div and add image section after answer box
            var resultsDiv = document.getElementById('results');
            var answerBox = resultsDiv.querySelector('.answer-box');
            
            if (!answerBox) return;
            
            // Create image box HTML
            var html = '<div class="image-box">';
            html += '<h3>🖼️ Image Generation</h3>';
            
            // Key concepts
            if (imageData.key_concepts && imageData.key_concepts.length > 0) {
                html += '<h4>🔑 Key Concepts Extracted:</h4>';
                html += '<div class="key-concepts">';
                for (var i = 0; i < imageData.key_concepts.length; i++) {
                    html += '<span>' + escapeHtml(imageData.key_concepts[i]) + '</span>';
                }
                html += '</div>';
            }
            
            // Image prompt
            if (imageData.image_prompt) {
                html += '<h4>📝 Generated Prompt for Image API:</h4>';
                html += '<div class="image-prompt">' + escapeHtml(imageData.image_prompt) + '</div>';
            }
            
            // Generated image
            if (imageData.image_url) {
                html += '<h4>🎨 Generated Image:</h4>';
                html += '<div class="generated-image">';
                html += '<img src="' + escapeHtml(imageData.image_url) + '" alt="Generated visualization">';
                html += '</div>';
            } else if (imageData.error) {
                html += '<div class="error">Image generation failed: ' + escapeHtml(imageData.error) + '</div>';
            }
            
            html += '</div>';
            
            // Insert after answer box
            answerBox.insertAdjacentHTML('afterend', html);
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            let html = '<div class="answer-box">';
            html += '<h3>Answer</h3>';
            html += '<p>' + escapeHtml(data.answer) + '</p>';
            html += '</div>';
            
            // Display image generation details if available
            if (data.image_data) {
                html += '<div class="image-box">';
                html += '<h3>🖼️ Image Generation</h3>';
                
                // Key concepts
                if (data.image_data.key_concepts && data.image_data.key_concepts.length > 0) {
                    html += '<h4>🔑 Key Concepts Extracted:</h4>';
                    html += '<div class="key-concepts">';
                    for (let i = 0; i < data.image_data.key_concepts.length; i++) {
                        html += '<span>' + escapeHtml(data.image_data.key_concepts[i]) + '</span>';
                    }
                    html += '</div>';
                }
                
                // Image prompt
                if (data.image_data.image_prompt) {
                    html += '<h4>📝 Generated Prompt for Image API:</h4>';
                    html += '<div class="image-prompt">' + escapeHtml(data.image_data.image_prompt) + '</div>';
                }
                
                // Generated image
                if (data.image_data.image_url) {
                    html += '<h4>🎨 Generated Image:</h4>';
                    html += '<div class="generated-image">';
                    html += '<img src="' + escapeHtml(data.image_data.image_url) + '" alt="Generated visualization">';
                    html += '</div>';
                } else if (data.image_data.error) {
                    html += '<div class="error">Image generation failed: ' + escapeHtml(data.image_data.error) + '</div>';
                }
                
                html += '</div>';
            }
            
            html += '<div class="context-box">';
            html += '<h3>Retrieved Context (' + data.num_sources + ' sources)</h3>';
            for (let i = 0; i < data.citations.length; i++) {
                const cite = data.citations[i];
                html += '<div class="citation">';
                html += '<strong>[' + (i+1) + '] ' + escapeHtml(cite.title) + '</strong><br>';
                html += 'Text: ' + escapeHtml(cite.citation);
                html += '</div>';
            }
            html += '</div>';
            
            html += '<div class="json-box">';
            html += '<h3>Raw JSON Response</h3>';
            html += '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
            html += '</div>';
            
            resultsDiv.innerHTML = html;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Event listeners
        console.log('✓ Setting up event listeners');
        
        const btn = document.getElementById('submitBtn');
        console.log('Button element:', btn);
        
        if (btn) {
            btn.addEventListener('click', function() {
                console.log('✓ Button click event fired');
                submitQuery();
            });
        } else {
            console.error('❌ Button not found!');
        }
        
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitQuery();
            }
        });
        
        console.log('✓ Event listeners attached');
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    """Serve the web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/query', methods=['POST'])
def query_documents():
    """API endpoint for querying documents (non-streaming)"""
    try:
        data = request.json
        query_text = data.get('query', '')
        k = data.get('k', 3)
        generate_image = data.get('generate_image', False)
        
        if not query_text:
            return jsonify({'error': 'No query provided'}), 400
        
        # Search documents
        documents = index_manager.search_with_documents(query_text, k=k)
        
        # Generate answer (non-streaming)
        result = rag_system.answer_query(query_text, documents, stream=False)
        
        # Generate image if requested
        if generate_image:
            answer_text = result.get('answer', '')
            
            # Extract key concepts
            key_concepts = extract_key_concepts(answer_text)
            
            # Generate image prompt
            image_prompt = generate_image_prompt(answer_text, key_concepts, query_text)
            
            # Generate image
            image_url, error = generate_image_from_hf(image_prompt)
            
            # Add image data to result
            result['image_data'] = {
                'key_concepts': key_concepts,
                'image_prompt': image_prompt,
                'image_url': image_url,
                'error': error
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stream', methods=['POST'])
def stream_query():
    """API endpoint for streaming query responses"""
    try:
        data = request.json
        query_text = data.get('query', '')
        k = data.get('k', 3)
        generate_image = data.get('generate_image', False)
        
        if not query_text:
            return jsonify({'error': 'No query provided'}), 400
        
        # Search documents
        documents = index_manager.search_with_documents(query_text, k=k)
        
        def generate():
            try:
                answer_text = ''
                
                # Stream the answer
                for chunk in rag_system.answer_query(query_text, documents, stream=True):
                    # Collect answer text for image generation
                    if chunk.get('type') == 'token':
                        answer_text += chunk.get('content', '')
                    
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Generate image after streaming is complete
                if generate_image and answer_text:
                    # Extract key concepts
                    key_concepts = extract_key_concepts(answer_text)
                    
                    # Generate image prompt
                    image_prompt = generate_image_prompt(answer_text, key_concepts, query_text)
                    
                    # Generate image
                    image_url, error = generate_image_from_hf(image_prompt)
                    
                    # Send image data as a separate chunk
                    image_chunk = {
                        'type': 'image_data',
                        'key_concepts': key_concepts,
                        'image_prompt': image_prompt,
                        'image_url': image_url,
                        'error': error
                    }
                    yield f"data: {json.dumps(image_chunk)}\n\n"
                    
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Alias for /api/query (bonus requirement)
@app.route('/answer', methods=['POST'])
def answer():
    """Alias endpoint for /api/query"""
    return query_documents()


if __name__ == '__main__':
    print("=" * 60)
    print("Initializing GenAI RAG Application")
    print("=" * 60)
    print()
    
    # Initialize index
    print("[1/2] Initializing vector index...")
    
    # Support DATA_PATH environment variable (for Docker)
    data_path = os.getenv("DATA_PATH")
    if data_path:
        data_file = Path(data_path)
        print(f"Using DATA_PATH from environment: {data_file}")
    else:
        data_file = Path(__file__).parent / "data" / "arxiv_2.9k.jsonl"
        print(f"Using default data path: {data_file}")
    
    index_manager = IndexManager(
        data_path=str(data_file),
        model_name="all-MiniLM-L6-v2"
    )
    index_manager.initialize()
    print()
    
    # Initialize LLM
    print("[2/2] Initializing LLM for text generation...")
    rag_system = RAGQuerySystem()
    print()
    
    print("=" * 60)
    print("✅ Application initialized successfully!")
    print("=" * 60)
    print()
    print("🚀 Starting server at http://0.0.0.0:8080")
    print("   Access from host: http://127.0.0.1:8080")
    print()
    
    app.run(host='0.0.0.0', port=8080)
