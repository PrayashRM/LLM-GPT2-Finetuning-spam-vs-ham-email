<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-2 Finetuning for Spam Classification</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        pre code {
            font-family: 'Menlo', 'Monaco', 'Consolas', 'Courier New', monospace;
        }
        /* Custom scrollbar for code blocks */
        pre::-webkit-scrollbar {
            height: 8px;
        }
        pre::-webkit-scrollbar-track {
            background: #1f2937; /* bg-gray-800 */
            border-radius: 4px;
        }
        pre::-webkit-scrollbar-thumb {
            background: #4b5563; /* bg-gray-600 */
            border-radius: 4px;
        }
        pre::-webkit-scrollbar-thumb:hover {
            background: #6b7280; /* bg-gray-500 */
        }
    </style>
</head>
<body class="bg-gray-900 text-gray-200 font-sans antialiased">

    <main class="max-w-4xl mx-auto p-4 sm:p-8 bg-gray-800 shadow-2xl rounded-2xl my-8 ring-1 ring-gray-700">

        <!-- 1. HEADER SECTION -->
        <header class="text-center mb-8">
            <h1 class="text-4xl sm:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 pb-2">
                🔢 GPT-2 Finetuning for Spam/Ham Classification
            </h1>
            <p class="text-lg text-gray-400 mt-2">
                <b>PyTorch | NLP | Transfer Learning</b>
            </p>
            <p class="text-xl text-gray-300 mt-3">
                Finetuning a from-scratch GPT-2 model to classify SMS messages with <b>95.67% Test Accuracy</b> 🎯
            </p>
            <div class="flex justify-center flex-wrap gap-2 my-6">
                <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&style=for-the-badge" alt="Python">
                <img src="https://img.shields.io/badge/PyTorch-LTS-orange?logo=pytorch&style=for-the-badge" alt="PyTorch">
                <img src="https://img.shields.io/badge/Model-GPT--2_124M-green?style=for-the-badge" alt="GPT-2 124M">
                <img src="https://img.shields.io/badge/Test_Accuracy-95.67%25-brightgreen?style=for-the-badge" alt="Accuracy">
                <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" alt="Status">
            </div>
        </header>

        <hr class="border-gray-700 my-8">

        <!-- 2. OVERVIEW -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">🚀</span> Overview
            </h2>
            <p class="text-base text-gray-300 mb-4 leading-relaxed">
                This project demonstrates the power of transfer learning by adapting a large language model (LLM) for a specific downstream task. It provides a complete, from-scratch implementation of the GPT-2 (124M) architecture in PyTorch, including components like `MultiHeadAttention`, `TransformerBlock`, and `LayerNorm`.
            </p>
            <p class="text-base text-gray-300 mb-4 leading-relaxed">
                The model is first initialized with pre-trained weights from OpenAI's "124M" checkpoint. Then, an efficient finetuning strategy is applied: all model parameters are frozen except for the final transformer block and the final layer normalization. This small fraction of the model is then trained on the UCI SMS Spam Collection dataset to perform binary classification (spam/ham).
            </p>
            <p class="text-base text-gray-300 mb-4 leading-relaxed">
                This approach is highly effective, achieving **95.67% accuracy** on the test set after just 5 epochs of training, showcasing a practical and compute-efficient way to leverage large pre-trained models.
            </p>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 3. KEY FEATURES -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">🎯</span> Key Features
            </h2>
            <ul class="list-disc list-inside space-y-3 mb-4 text-gray-300">
                <li><span class="mr-2">✨</span> <strong>GPT-2 from Scratch:</strong> Includes a complete, commented implementation of the GPT-2 architecture.</li>
                <li><span class="mr-2">⚙️</span> <strong>Efficient Finetuning:</strong> Demonstrates transfer learning by freezing most layers and only training the final `TransformerBlock` and `LayerNorm`.</li>
                <li><span class="mr-2">📊</span> <strong>High Accuracy:</strong> Achieves **95.67%** accuracy on the test set after just 5 epochs.</li>
                <li><span class="mr-2">💾</span> <strong>Pre-trained Weights:</strong> Integrates a helper script (`gpt_download3`) to automatically fetch and load the official OpenAI "124M" model weights.</li>
                <li><span class="mr-2">🔡</span> <strong>Modern Tokenization:</strong> Uses `tiktoken` (GPT-2's official tokenizer) for text encoding.</li>
                <li><span class="mr-2">⚖️</span> <strong>Data Preprocessing:</strong> Includes a clear pipeline for balancing the imbalanced UCI SMS dataset via downsampling.</li>
            </ul>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 4. DATASET & PERFORMANCE -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">📊</span> Dataset & Performance
            </h2>
            <p class="text-gray-300 mb-4">
                The model is trained on the <a href="https://archive.ics.uci.edu/dataset/228/sms+spam+collection" class="text-blue-400 hover:underline">UCI SMS Spam Collection Dataset</a>. The original dataset is imbalanced (4825 'ham' vs. 747 'spam'). To prevent bias, the 'ham' class was downsampled to 747 samples to create a 1:1 balanced dataset.
            </p>
            <div class="overflow-x-auto rounded-lg ring-1 ring-gray-700">
                <table class="w-full text-left">
                    <thead class="bg-gray-700 text-gray-100">
                        <tr>
                            <th class="p-3">Metric</th>
                            <th class="p-3">Value</th>
                        </tr>
                    </thead>
                    <tbody class="bg-gray-800 divide-y divide-gray-700">
                        <tr>
                            <td class="p-3 font-medium">Test Accuracy</td>
                            <td class="p-3 font-bold text-green-400">95.67%</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Validation Accuracy</td>
                            <td class="p-3">97.32%</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Training Accuracy</td>
                            <td class="p-3">97.21%</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Training Samples</td>
                            <td class="p-3">1045 (70%)</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Validation Samples</td>
                            <td class="p-3">149 (10%)</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Test Samples</td>
                            <td class="p-3">300 (20%)</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 5. MODEL ARCHITECTURE -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">🏗️</span> Model Architecture
            </h2>
            <p class="text-gray-300 mb-4">
                The model is a from-scratch implementation of the GPT-2 (124M) architecture. The core components are the `TransformerBlock` and the main `GPTModel` class.
            </p>
            
            <h3 class="text-xl font-semibold mb-3 text-blue-300">Transformer Block</h3>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mb-4"><code class="language-python">
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
            </code></pre>

            <h3 class="text-xl font-semibold mb-3 text-blue-300">Main GPT Model</h3>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mb-4"><code class="language-python">
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
            </code></pre>

            <h3 class="text-xl font-semibold mb-3 text-blue-300">Architecture Flow</h3>
            <ul class="list-disc list-inside space-y-2 mb-4 text-gray-300">
                <li><strong>Base Model:</strong> GPT-2 Small (124M Parameters)</li>
                <li><strong>Embedding Dim:</strong> 768</li>
                <li><strong>Layers:</strong> 12 `TransformerBlock`s</li>
                <li><strong>Heads:</strong> 12 per attention layer</li>
                <li><strong>Base Context Length:</strong> 1024 tokens</li>
                <li><strong>Tokenizer:</strong> `tiktoken` (gpt2 encoding)</li>
                <li><strong>Finetuning Head:</strong> The `out_head` is replaced with an `nn.Linear(768, 2)` for classification.</li>
            </ul>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 6. DATA PREPROCESSING -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">🔧</span> Data Preprocessing Pipeline
            </h2>
            <ol class="list-decimal list-inside space-y-4 text-gray-300">
                <li>
                    <strong>Download & Load:</strong> The dataset is downloaded from `archive.ics.uci.edu` and loaded into a Pandas DataFrame.
                </li>
                <li>
                    <strong>Balance Dataset:</strong> The imbalanced dataset is balanced by downsampling the majority 'ham' class to match the 'spam' class count (747 samples each).
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-python">
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df
                    </code></pre>
                </li>
                <li>
                    <strong>Encode Labels:</strong> Labels are mapped to integers: `{"ham": 0, "spam": 1}`.
                </li>
                <li>
                    <strong>Split Data:</strong> The balanced DataFrame is split into 70% train, 10% validation, and 20% test sets.
                </li>
                <li>
                    <strong>Tokenize & Pad:</strong> A custom `SpamDataset` class handles tokenization using `tiktoken`. All sequences are padded with the `pad_token_id` (50256) to the maximum sequence length found in the training set (120 tokens).
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-python">
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        # ... (truncation and padding logic) ...
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    # ...
                    </code></pre>
                </li>
            </ol>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 7. TRAINING CONFIGURATION -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">⚙️</span> Training Configuration
            </h2>
            <p class="text-gray-300 mb-4">The model was finetuned with the following hyperparameters:</p>
            <div class="overflow-x-auto rounded-lg ring-1 ring-gray-700">
                <table class="w-full text-left">
                    <thead class="bg-gray-700 text-gray-100">
                        <tr>
                            <th class="p-3">Hyperparameter</th>
                            <th class="p-3">Value</th>
                            <th class="p-3">Description</th>
                        </tr>
                    </thead>
                    <tbody class="bg-gray-800 divide-y divide-gray-700">
                        <tr>
                            <td class="p-3 font-medium">Optimizer</td>
                            <td class="p-3">`AdamW`</td>
                            <td class="p-3">AdamW optimizer</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Learning Rate</td>
                            <td class="p-3">`5e-5`</td>
                            <td class="p-3">(5 x 10^-5)</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Weight Decay</td>
                            <td class="p-3">`0.1`</td>
                            <td class="p-3">L2 Regularization</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Batch Size</td>
                            <td class="p-3">`8`</td>
                            <td class="p-3">Samples per training batch</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Epochs</td>
                            <td class="p-3">`5`</td>
                            <td class="p-3">Full passes through the training data</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Loss Function</td>
                            <td class="p-3">`CrossEntropyLoss`</td>
                            <td class="p-3">Standard for classification</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 8. FINETUNING STRATEGY -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">🎨</span> Finetuning Strategy (Transfer Learning)
            </h2>
            <p class="text-gray-300 mb-4">
                This project's core is its efficient finetuning strategy. Instead of training the entire 124M parameter model, we freeze almost all layers and only update the weights of the final layers. This is significantly faster and less resource-intensive.
            </p>
            <ol class="list-decimal list-inside space-y-4 text-gray-300">
                <li>
                    <strong>Load Pre-trained Model:</strong> A full `GPTModel` (124M) is instantiated and loaded with official weights from the "124M" checkpoint.
                </li>
                <li>
                    <strong>Freeze All Weights:</strong> All parameters in the model are frozen to prevent them from being updated during training.
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-python">
for param in model.parameters():
    param.requires_grad = False
                    </code></pre>
                </li>
                <li>
                    <strong>Replace Classification Head:</strong> The original `out_head` (which maps 768 dims to 50257 vocab tokens) is replaced with a new, randomly initialized head for 2-class classification.
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-python">
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], 
                                 out_features=num_classes)
                    </code></pre>
                </li>
                <li>
                    <strong>Selectively Unfreeze Layers:</strong> Only the parameters for the *last* `TransformerBlock` and the `final_norm` layer are unfrozen. These, along with the new `out_head`, are the only parts of the model that will be trained.
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-python">
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True
                    </code></pre>
                </li>
                <li>
                    <strong>Train:</strong> The `AdamW` optimizer is initialized (it only registers parameters with `requires_grad=True`) and the model is trained for 5 epochs. Only the ~10M parameters of the last block are updated.
                </li>
            </ol>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 9. TECHNOLOGIES -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">🛠️</span> Technologies & Dependencies
            </h2>
            <ul class="list-disc list-inside space-y-2 mb-4 text-gray-300">
                <li><code class="bg-gray-700 px-2 py-1 rounded">torch</code> (PyTorch)</li>
                <li><code class="bg-gray-700 px-2 py-1 rounded">pandas</code></li>
                <li><code class="bg-gray-700 px-2 py-1 rounded">numpy</code></li>
                <li><code class="bg-gray-700 px-2 py-1 rounded">tiktoken</code></li>
                <li><code class="bg-gray-700 px-2 py-1 rounded">matplotlib</code></li>
                <li><code class="bg-gray-700 px-2 py-1 rounded">gpt_download3.py</code> (helper script, included)</li>
            </ul>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 10. INSTALLATION & SETUP -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">📦</span> Installation & Setup
            </h2>
            <ol class="list-decimal list-inside space-y-3 text-gray-300">
                <li>
                    <strong>Clone the Repository:</strong>
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-bash">
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
                    </code></pre>
                </li>
                <li>
                    <strong>Install Dependencies:</strong>
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-bash">
pip install torch pandas numpy tiktoken matplotlib
                    </code></pre>
                </li>
                <li>
                    <strong>Run the Notebook:</strong> The notebook (`GPT_2_Finetuning_spam_ham.ipynb`) is self-contained. It will download the dataset and the model weights automatically upon first run.
                </li>
            </ol>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 11. USAGE (PREDICTION) -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">🚀</span> Usage
            </h2>
            <p class="text-gray-300 mb-4">
                After training, the model can be used for inference with the `classify_review` function. First, load the saved model state:
            </p>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mb-4"><code class="language-python">
# Assumes 'model' is an instance of GPTModel(BASE_CONFIG)
# and 'device' is set (e.g., torch.device("cuda"))
model_state_dict = torch.load("review_classifier.pth")
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

# Helper function for classification
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate and pad
    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"
            </code></pre>

            <h3 class="text-xl font-semibold mb-3 text-blue-300">Example 1: Spam</h3>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mb-4"><code class="language-python">
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

# Output: spam
            </code></pre>

            <h3 class="text-xl font-semibold mb-3 text-blue-300">Example 2: Ham (Not Spam)</h3>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mb-4"><code class="language-python">
text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

# Output: not spam
            </code></pre>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 12. PROJECT STRUCTURE -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">📁</span> Project Structure
            </h2>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700"><code>
.
├── gpt2/                          # Created by gpt_download3 to store model weights
│   └── 124M/
│       ├── model.ckpt.data-00000-of-00001
│       └── ...
├── sms_spam_collection/           # Created by download script
│   └── SMSSpamCollection.tsv
├── GPT_2_Finetuning_spam_ham.ipynb  # Main Jupyter Notebook
├── gpt_download3.py               # Helper script to download weights
├── train.csv                      # Generated by the notebook
├── validation.csv                 # Generated by the notebook
├── test.csv                       # Generated by the notebook
├── review_classifier.pth          # Saved model weights after finetuning
└── README.md
            </code></pre>
        </section>

        <hr class="border-gray-700 my-8">

<!-- 13. TRAINING PROGRESS -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">📈</span> Training Progress
            </h2>
            <p class="text-gray-300 mb-4">
                The model was trained for 5 epochs. Training and validation loss decreased steadily, while accuracy for both sets quickly rose to over 95%, indicating successful and efficient learning without significant overfitting.
            </p>
            <div class="bg-gray-900 p-4 rounded-lg ring-1 ring-gray-700 text-center space-y-6">
                <!-- Plot for Training/Validation Loss -->
                <div>
                    <p class="text-gray-400 italic mb-3">Training & Validation Loss</p>
                    <img src="loss-plot.png" alt="Training and Validation Loss Plot" class="w-full max-w-lg mx-auto h-auto rounded-lg shadow-lg object-cover" onerror="this.alt='Failed to load training loss plot. Please ensure loss-plot.png is in the repository root.'; this.style.display='block'; this.style.color='#ef4444'; this.style.border='1px dashed #ef4444'; this.style.padding='1rem'; this.style.width='100%'; this.src=''; this.parentElement.innerHTML = this.alt;">
                </div>
                <!-- Plot for Training/Validation Accuracy -->
                <div>
                    <p class="text-gray-400 italic mb-3">Training & Validation Accuracy</p>
                    <img src="accuracy-plot.png" alt="Training and Validation Accuracy Plot" class="w-full max-w-lg mx-auto h-auto rounded-lg shadow-lg object-cover" onerror="this.alt='Failed to load training accuracy plot. Please ensure accuracy-plot.png is in the repository root.'; this.style.display='block'; this.style.color='#ef4444'; this.style.border='1px dashed #ef4444'; this.style.padding='1rem'; this.style.width='100%'; this.src=''; this.parentElement.innerHTML = this.alt;">
                </div>
            </div>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 14. AUTHOR -->
        <footer class="text-center mt-12">
            <p class="text-lg font-semibold text-white">
                <b>[Your Name]</b>
            </p>
            <p class="text-base text-gray-400">
                [Your Degree & Institution]
            </p>
            <p class="text-base text-gray-400">
                📧 <a href="mailto:email@example.com" class="text-blue-400 hover:underline">email@example.com</a>
            </p>
            
            <hr class="border-gray-700 my-8">
            
            <p class="text-sm text-gray-500 italic">
                "The best way to predict the future is to create it."
            </p>
        </footer>

    </main>

</body>
</html>


