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
        pre::-webkit-scrollbar {
            height: 8px;
        }
        pre::-webkit-scrollbar-track {
            background: #1f2937;
            border-radius: 4px;
        }
        pre::-webkit-scrollbar-thumb {
            background: #4b5563;
            border-radius: 4px;
        }
        pre::-webkit-scrollbar-thumb:hover {
            background: #6b7280;
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
                This project demonstrates the power of transfer learning by adapting a large language model (LLM) for a specific downstream task. It provides a complete, from-scratch implementation of the GPT-2 (124M) architecture in PyTorch, including components like <code class="bg-gray-700 px-2 py-1 rounded">MultiHeadAttention</code>, <code class="bg-gray-700 px-2 py-1 rounded">TransformerBlock</code>, and <code class="bg-gray-700 px-2 py-1 rounded">LayerNorm</code>.
            </p>
            <p class="text-base text-gray-300 mb-4 leading-relaxed">
                The model is first initialized with pre-trained weights from OpenAI's "124M" checkpoint. Then, an efficient finetuning strategy is applied: all model parameters are frozen except for the final transformer block and the final layer normalization. This small fraction of the model is then trained on the UCI SMS Spam Collection dataset to perform binary classification (spam/ham).
            </p>
            <p class="text-base text-gray-300 mb-4 leading-relaxed">
                This approach is highly effective, achieving <strong>95.67% accuracy</strong> on the test set after just 5 epochs of training, showcasing a practical and compute-efficient way to leverage large pre-trained models.
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
                <li><span class="mr-2">⚙️</span> <strong>Efficient Finetuning:</strong> Demonstrates transfer learning by freezing most layers and only training the final TransformerBlock and LayerNorm.</li>
                <li><span class="mr-2">📊</span> <strong>High Accuracy:</strong> Achieves <strong>95.67%</strong> accuracy on the test set after just 5 epochs.</li>
                <li><span class="mr-2">💾</span> <strong>Pre-trained Weights:</strong> Integrates a helper script (gpt_download3) to automatically fetch and load the official OpenAI "124M" model weights.</li>
                <li><span class="mr-2">🔡</span> <strong>Modern Tokenization:</strong> Uses tiktoken (GPT-2's official tokenizer) for text encoding.</li>
                <li><span class="mr-2">⚖️</span> <strong>Data Preprocessing:</strong> Includes a clear pipeline for balancing the imbalanced UCI SMS dataset via downsampling.</li>
            </ul>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 4. DATASET & DATA SPLITS -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">📊</span> Dataset & Data Splits
            </h2>
            <p class="text-gray-300 mb-4">
                The model is trained on the <a href="https://archive.ics.uci.edu/dataset/228/sms+spam+collection" class="text-blue-400 hover:underline">UCI SMS Spam Collection Dataset</a>. The original dataset is imbalanced (4825 'ham' vs. 747 'spam'). To prevent bias, the 'ham' class was downsampled to 747 samples to create a 1:1 balanced dataset of 1494 total samples.
            </p>
            <p class="text-gray-300 mb-4">
                This balanced dataset was then split into training, validation, and test sets:
            </p>
            <div class="overflow-x-auto rounded-lg ring-1 ring-gray-700">
                <table class="w-full text-left">
                    <thead class="bg-gray-700 text-gray-100">
                        <tr>
                            <th class="p-3">Dataset Split</th>
                            <th class="p-3">Percentage</th>
                            <th class="p-3">Sample Count</th>
                        </tr>
                    </thead>
                    <tbody class="bg-gray-800 divide-y divide-gray-700">
                        <tr>
                            <td class="p-3 font-medium">Training</td>
                            <td class="p-3">70%</td>
                            <td class="p-3">1045</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Validation</td>
                            <td class="p-3">10%</td>
                            <td class="p-3">149</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Test</td>
                            <td class="p-3">20%</td>
                            <td class="p-3">300</td>
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
                The model is a from-scratch implementation of the GPT-2 (124M) architecture. The core components are the TransformerBlock and the main GPTModel class.
            </p>
            
            <h3 class="text-xl font-semibold mb-3 text-blue-300">Transformer Block</h3>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mb-4"><code class="language-python">class TransformerBlock(nn.Module):
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
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x</code></pre>

            <h3 class="text-xl font-semibold mb-3 text-blue-300">Main GPT Model</h3>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mb-4"><code class="language-python">class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits</code></pre>

            <h3 class="text-xl font-semibold mb-3 text-blue-300">Architecture Flow</h3>
            <ul class="list-disc list-inside space-y-2 mb-4 text-gray-300">
                <li><strong>Base Model:</strong> GPT-2 Small (124M Parameters)</li>
                <li><strong>Embedding Dim:</strong> 768</li>
                <li><strong>Layers:</strong> 12 TransformerBlocks</li>
                <li><strong>Heads:</strong> 12 per attention layer</li>
                <li><strong>Base Context Length:</strong> 1024 tokens</li>
                <li><strong>Tokenizer:</strong> tiktoken (gpt2 encoding)</li>
                <li><strong>Finetuning Head:</strong> The out_head is replaced with an nn.Linear(768, 2) for classification.</li>
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
                    <strong>Download & Load:</strong> The dataset is downloaded from archive.ics.uci.edu and loaded into a Pandas DataFrame.
                </li>
                <li>
                    <strong>Balance Dataset:</strong> The imbalanced dataset is balanced by downsampling the majority 'ham' class to match the 'spam' class count (747 samples each).
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-python">def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df</code></pre>
                </li>
                <li>
                    <strong>Encode Labels:</strong> Labels are mapped to integers: {"ham": 0, "spam": 1}.
                </li>
                <li>
                    <strong>Split Data:</strong> The balanced DataFrame is split into 70% train, 10% validation, and 20% test sets.
                </li>
                <li>
                    <strong>Tokenize & Pad:</strong> A custom SpamDataset class handles tokenization using tiktoken. All sequences are padded with the pad_token_id (50256) to the maximum sequence length found in the training set (120 tokens).
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
                            <td class="p-3">AdamW</td>
                            <td class="p-3">AdamW optimizer</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Learning Rate</td>
                            <td class="p-3">5e-5</td>
                            <td class="p-3">(5 x 10^-5)</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Weight Decay</td>
                            <td class="p-3">0.1</td>
                            <td class="p-3">L2 Regularization</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Batch Size</td>
                            <td class="p-3">8</td>
                            <td class="p-3">Samples per training batch</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Epochs</td>
                            <td class="p-3">5</td>
                            <td class="p-3">Full passes through the training data</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Loss Function</td>
                            <td class="p-3">CrossEntropyLoss</td>
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
                    <strong>Load Pre-trained Model:</strong> A full GPTModel (124M) is instantiated and loaded with official weights from the "124M" checkpoint.
                </li>
                <li>
                    <strong>Freeze All Weights:</strong> All parameters in the model are frozen to prevent them from being updated during training.
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-python">for param in model.parameters():
    param.requires_grad = False</code></pre>
                </li>
                <li>
                    <strong>Replace Classification Head:</strong> The original out_head (which maps 768 dims to 50257 vocab tokens) is replaced with a new, randomly initialized head for 2-class classification.
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-python">num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"], 
    out_features=num_classes
)</code></pre>
                </li>
                <li>
                    <strong>Selectively Unfreeze Layers:</strong> Only the parameters for the last TransformerBlock and the final_norm layer are unfrozen. These, along with the new out_head, are the only parts of the model that will be trained.
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-python">for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True</code></pre>
                </li>
                <li>
                    <strong>Train:</strong> The AdamW optimizer is initialized (it only registers parameters with requires_grad=True) and the model is trained for 5 epochs. Only the ~10M parameters of the last block are updated.
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
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-bash">git clone https://github.com/PrayashRM/GPT-2-Spam-Classification.git
cd GPT-2-Spam-Classification</code></pre>
                </li>
                <li>
                    <strong>Install Dependencies:</strong>
                    <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mt-2"><code class="language-bash">pip install torch pandas numpy tiktoken matplotlib</code></pre>
                </li>
                <li>
                    <strong>Run the Notebook:</strong> The notebook (GPT_2_Finetuning_spam_ham.ipynb) is self-contained. It will download the dataset and the model weights automatically upon first run.
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
                After training, the model can be used for inference with the classify_review function. First, load the saved model state:
            </p>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mb-4"><code class="language-python">model_state_dict = torch.load("review_classifier.pth")
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()</code></pre>

            <h3 class="text-xl font-semibold mb-3 text-blue-300">Example 1: Spam</h3>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mb-4"><code class="language-python">text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

# Output: spam</code></pre>

            <h3 class="text-xl font-semibold mb-3 text-blue-300">Example 2: Ham (Not Spam)</h3>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mb-4"><code class="language-python">text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

# Output: not spam</code></pre>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 12. PROJECT STRUCTURE -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">📁</span> Project Structure
            </h2>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700"><code>.
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
└── README.md</code></pre>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 13. TRAINING & PERFORMANCE ANALYSIS -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">📈</span> Training & Performance Analysis
            </h2>
            <p class="text-gray-300 mb-4 leading-relaxed">
                The model was trained for 5 epochs, which completed in just <strong>0.99 minutes</strong> on a T4 GPU. The training process was highly effective, with validation accuracy quickly matching and even exceeding training accuracy, indicating that the model learned to generalize well without overfitting.
            </p>
            <p class="text-gray-300 mb-4 leading-relaxed">
                By Epoch 3, the model already achieved 90% accuracy on both training and validation sets. By Epoch 4, it reached 97.5% validation accuracy, demonstrating the power of finetuning even a single transformer block.
            </p>

            <h3 class="text-xl font-semibold mb-3 text-blue-300">End-of-Epoch Summary</h3>
            <div class="overflow-x-auto rounded-lg ring-1 ring-gray-700 mb-6">
                <table class="w-full text-left">
                    <thead class="bg-gray-700 text-gray-100">
                        <tr>
                            <th class="p-3">Epoch</th>
                            <th class="p-3">Training Accuracy</th>
                            <th class="p-3">Validation Accuracy</th>
                        </tr>
                    </thead>
                    <tbody class="bg-gray-800 divide-y divide-gray-700">
                        <tr>
                            <td class="p-3 font-medium">1</td>
                            <td class="p-3">70.00%</td>
                            <td class="p-3">72.50%</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">2</td>
                            <td class="p-3">82.50%</td>
                            <td class="p-3">85.00%</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">3</td>
                            <td class="p-3">90.00%</td>
                            <td class="p-3">90.00%</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">4</td>
                            <td class="p-3">100.00%</td>
                            <td class="p-3">97.50%</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">5</td>
                            <td class="p-3">100.00%</td>
                            <td class="p-3">97.50%</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <h3 class="text-xl font-semibold mb-3 text-blue-300">Final Model Performance</h3>
            <p class="text-gray-300 mb-4">
                After 5 epochs, the model's final performance was evaluated on the unseen test set, achieving a robust accuracy of 95.67%.
            </p>
            <div class="overflow-x-auto rounded-lg ring-1 ring-gray-700 mb-6">
                <table class="w-full text-left">
                    <thead class="bg-gray-700 text-gray-100">
                        <tr>
                            <th class="p-3">Metric</th>
                            <th class="p-3">Value</th>
                        </tr>
                    </thead>
                    <tbody class="bg-gray-800 divide-y divide-gray-700">
                        <tr class="bg-green-900/30">
                            <td class="p-3 font-medium">Test Accuracy (Final)</td>
                            <td class="p-3 font-bold text-green-400">95.67%</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Validation Accuracy (Final)</td>
                            <td class="p-3">97.32%</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Training Accuracy (Final)</td>
                            <td class="p-3">97.21%</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Training Time</td>
                            <td class="p-3">0.99 minutes (T4 GPU)</td>
                        </tr>
                        <tr>
                            <td class="p-3 font-medium">Parameters Trained</td>
                            <td class="p-3">~10M (Last Block Only)</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <h3 class="text-xl font-semibold mb-3 text-blue-300">Key Insights</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div class="bg-gradient-to-br from-blue-900/40 to-purple-900/40 p-5 rounded-lg ring-1 ring-blue-500/30">
                    <div class="text-3xl mb-2">⚡</div>
                    <h4 class="font-semibold text-lg mb-2 text-blue-300">Lightning Fast Training</h4>
                    <p class="text-sm text-gray-300">Completed in under 1 minute on a T4 GPU by only training the final transformer block</p>
                </div>
                <div class="bg-gradient-to-br from-green-900/40 to-teal-900/40 p-5 rounded-lg ring-1 ring-green-500/30">
                    <div class="text-3xl mb-2">🎯</div>
                    <h4 class="font-semibold text-lg mb-2 text-green-300">High Accuracy</h4>
                    <p class="text-sm text-gray-300">Achieved 95.67% test accuracy with minimal overfitting</p>
                </div>
                <div class="bg-gradient-to-br from-orange-900/40 to-red-900/40 p-5 rounded-lg ring-1 ring-orange-500/30">
                    <div class="text-3xl mb-2">💡</div>
                    <h4 class="font-semibold text-lg mb-2 text-orange-300">Efficient Transfer Learning</h4>
                    <p class="text-sm text-gray-300">Only ~8% of model parameters were updated during training</p>
                </div>
                <div class="bg-gradient-to-br from-pink-900/40 to-purple-900/40 p-5 rounded-lg ring-1 ring-pink-500/30">
                    <div class="text-3xl mb-2">🚀</div>
                    <h4 class="font-semibold text-lg mb-2 text-pink-300">Quick Convergence</h4>
                    <p class="text-sm text-gray-300">Model reached 90% accuracy by epoch 3, showing rapid learning</p>
                </div>
            </div>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 14. RESULTS VISUALIZATION -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">📊</span> Training Visualizations
            </h2>
            <p class="text-gray-300 mb-4">
                The following plots show the training and validation metrics throughout the training process:
            </p>
            
            <div class="space-y-6">
                <div class="bg-gray-900 p-6 rounded-lg ring-1 ring-gray-700">
                    <h3 class="text-lg font-semibold mb-3 text-center text-gray-300">Loss Over Training</h3>
                    <div class="bg-gray-800 p-4 rounded text-center">
                        <p class="text-gray-400 italic">Training and validation loss decreased steadily, indicating effective learning without overfitting</p>
                    </div>
                </div>

                <div class="bg-gray-900 p-6 rounded-lg ring-1 ring-gray-700">
                    <h3 class="text-lg font-semibold mb-3 text-center text-gray-300">Accuracy Over Training</h3>
                    <div class="bg-gray-800 p-4 rounded text-center">
                        <p class="text-gray-400 italic">Both training and validation accuracy improved rapidly, reaching over 95% by epoch 4</p>
                    </div>
                </div>
            </div>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 15. COMPARISON & ADVANTAGES -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">⚖️</span> Why This Approach Works
            </h2>
            <div class="bg-gradient-to-r from-blue-900/30 to-purple-900/30 p-6 rounded-lg ring-1 ring-blue-500/30 mb-4">
                <h3 class="text-xl font-semibold mb-3 text-blue-300">Transfer Learning Benefits</h3>
                <ul class="space-y-3 text-gray-300">
                    <li class="flex items-start">
                        <span class="text-green-400 mr-2">✓</span>
                        <span><strong>Leverages Pre-trained Knowledge:</strong> The model starts with linguistic understanding from billions of tokens</span>
                    </li>
                    <li class="flex items-start">
                        <span class="text-green-400 mr-2">✓</span>
                        <span><strong>Minimal Data Required:</strong> Only ~1500 samples needed for 95%+ accuracy</span>
                    </li>
                    <li class="flex items-start">
                        <span class="text-green-400 mr-2">✓</span>
                        <span><strong>Fast Training:</strong> Under 1 minute vs hours for training from scratch</span>
                    </li>
                    <li class="flex items-start">
                        <span class="text-green-400 mr-2">✓</span>
                        <span><strong>Resource Efficient:</strong> Only need to store gradients for ~10M params instead of 124M</span>
                    </li>
                    <li class="flex items-start">
                        <span class="text-green-400 mr-2">✓</span>
                        <span><strong>Better Generalization:</strong> Pre-trained features help avoid overfitting on small datasets</span>
                    </li>
                </ul>
            </div>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 16. FUTURE IMPROVEMENTS -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">🔮</span> Future Improvements
            </h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="bg-gray-900/50 p-4 rounded-lg ring-1 ring-gray-700">
                    <h3 class="font-semibold mb-2 text-blue-300">📱 Multi-class Classification</h3>
                    <p class="text-sm text-gray-400">Extend to classify different types of spam (promotional, phishing, etc.)</p>
                </div>
                <div class="bg-gray-900/50 p-4 rounded-lg ring-1 ring-gray-700">
                    <h3 class="font-semibold mb-2 text-blue-300">🌍 Multilingual Support</h3>
                    <p class="text-sm text-gray-400">Train on multilingual datasets for cross-language spam detection</p>
                </div>
                <div class="bg-gray-900/50 p-4 rounded-lg ring-1 ring-gray-700">
                    <h3 class="font-semibold mb-2 text-blue-300">⚡ Model Optimization</h3>
                    <p class="text-sm text-gray-400">Implement quantization and pruning for faster inference</p>
                </div>
                <div class="bg-gray-900/50 p-4 rounded-lg ring-1 ring-gray-700">
                    <h3 class="font-semibold mb-2 text-blue-300">🔄 Active Learning</h3>
                    <p class="text-sm text-gray-400">Implement active learning pipeline for continuous improvement</p>
                </div>
            </div>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 17. REFERENCES & RESOURCES -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">📚</span> References & Resources
            </h2>
            <ul class="space-y-3 text-gray-300">
                <li>
                    <strong>Dataset:</strong> 
                    <a href="https://archive.ics.uci.edu/dataset/228/sms+spam+collection" class="text-blue-400 hover:underline ml-2">
                        UCI SMS Spam Collection
                    </a>
                </li>
                <li>
                    <strong>Model:</strong> 
                    <a href="https://github.com/openai/gpt-2" class="text-blue-400 hover:underline ml-2">
                        OpenAI GPT-2
                    </a>
                </li>
                <li>
                    <strong>Framework:</strong> 
                    <a href="https://pytorch.org/" class="text-blue-400 hover:underline ml-2">
                        PyTorch
                    </a>
                </li>
                <li>
                    <strong>Tokenizer:</strong> 
                    <a href="https://github.com/openai/tiktoken" class="text-blue-400 hover:underline ml-2">
                        TikToken
                    </a>
                </li>
            </ul>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 18. CONTRIBUTING -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">🤝</span> Contributing
            </h2>
            <p class="text-gray-300 mb-4">
                Contributions are welcome! If you'd like to improve this project, please:
            </p>
            <ol class="list-decimal list-inside space-y-2 text-gray-300 mb-4">
                <li>Fork the repository</li>
                <li>Create a feature branch (<code class="bg-gray-700 px-2 py-1 rounded">git checkout -b feature/AmazingFeature</code>)</li>
                <li>Commit your changes (<code class="bg-gray-700 px-2 py-1 rounded">git commit -m 'Add some AmazingFeature'</code>)</li>
                <li>Push to the branch (<code class="bg-gray-700 px-2 py-1 rounded">git push origin feature/AmazingFeature</code>)</li>
                <li>Open a Pull Request</li>
            </ol>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 19. LICENSE -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3 text-2xl">⚖️</span> License
            </h2>
            <p class="text-gray-300">
                This project is open source and available for educational purposes. The GPT-2 model weights are subject to OpenAI's license terms.
            </p>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- 20. AUTHOR -->
        <footer class="text-center mt-12">
            <div class="bg-gradient-to-r from-blue-900/30 to-purple-900/30 p-8 rounded-xl ring-1 ring-blue-500/30">
                <div class="mb-4">
                    <div class="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full mx-auto mb-4 flex items-center justify-center text-3xl font-bold">
                        PM
                    </div>
                </div>
                <p class="text-2xl font-bold text-white mb-2">
                    Prayash Ranjan Mohanty
                </p>
                <p class="text-base text-gray-400 mb-4">
                    Machine Learning Enthusiast | NLP Developer
                </p>
                
                <div class="flex justify-center gap-4 mb-6">
                    <a href="https://github.com/PrayashRM" 
                       class="flex items-center gap-2 bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg transition-colors">
                        <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                        </svg>
                        GitHub
                    </a>
                    <a href="https://www.linkedin.com/in/prayash-mohanty-209303382" 
                       class="flex items-center gap-2 bg-blue-700 hover:bg-blue-600 px-4 py-2 rounded-lg transition-colors">
                        <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
                        </svg>
                        LinkedIn
                    </a>
                </div>
                
                <div class="text-gray-400">
                    <p class="mb-2">
                        📧 <a href="mailto:prayashranjanmohanty11@gmail.com" class="text-blue-400 hover:underline">
                            prayashranjanmohanty11@gmail.com
                        </a>
                    </p>
                </div>
            </div>
            
            <div class="mt-8 pt-6 border-t border-gray-700">
                <p class="text-sm text-gray-500 italic mb-4">
                    "The best way to predict the future is to create it." - Abraham Lincoln
                </p>
                <p class="text-xs text-gray-600">
                    Built with ❤️ using PyTorch and GPT-2 | © 2024 Prayash Ranjan Mohanty
                </p>
            </div>
        </footer>

    </main>

</body>
</html>
