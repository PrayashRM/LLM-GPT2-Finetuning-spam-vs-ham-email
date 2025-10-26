<!DOCTYPE html>
<html lang="en">
<body>

<h1 align="center">🔢 GPT-2 Finetuning for Spam/Ham Classification</h1>

<p align="center">
  <b>PyTorch | NLP | Transfer Learning</b><br>
  Finetuning a from-scratch GPT-2 model to classify SMS messages with <b>95.67% Test Accuracy</b> 🎯
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-LTS-orange?logo=pytorch&style=for-the-badge" alt="PyTorch">
  <img src="https://img.shields.io/badge/Model-GPT--2_124M-green?style=for-the-badge" alt="GPT-2 124M">
  <img src="https://img.shields.io/badge/Test_Accuracy-95.67%25-brightgreen?style=for-the-badge" alt="Accuracy">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" alt="Status">
</p>

<hr>

<h2>🚀 Overview</h2>

<p>
This project demonstrates the power of transfer learning by adapting a large language model (LLM) for a specific downstream task. It provides a complete, from-scratch implementation of the GPT-2 (124M) architecture in PyTorch, including components like <code>MultiHeadAttention</code>, <code>TransformerBlock</code>, and <code>LayerNorm</code>.
</p>

<p>
The model is first initialized with pre-trained weights from OpenAI's "124M" checkpoint. Then, an efficient finetuning strategy is applied: all model parameters are frozen except for the final transformer block and the final layer normalization. This small fraction of the model is then trained on the UCI SMS Spam Collection dataset to perform binary classification (spam/ham).
</p>

<p>
This approach is highly effective, achieving <b>95.67% accuracy</b> on the test set after just 5 epochs of training, showcasing a practical and compute-efficient way to leverage large pre-trained models.
</p>

<hr>

<h2>🎯 Key Features</h2>

<ul>
  <li>✨ <b>GPT-2 from Scratch:</b> Includes a complete, commented implementation of the GPT-2 architecture.</li>
  <li>⚙️ <b>Efficient Finetuning:</b> Demonstrates transfer learning by freezing most layers and only training the final TransformerBlock and LayerNorm.</li>
  <li>📊 <b>High Accuracy:</b> Achieves <b>95.67%</b> accuracy on the test set after just 5 epochs.</li>
  <li>💾 <b>Pre-trained Weights:</b> Integrates a helper script (gpt_download3) to automatically fetch and load the official OpenAI "124M" model weights.</li>
  <li>🔡 <b>Modern Tokenization:</b> Uses tiktoken (GPT-2's official tokenizer) for text encoding.</li>
  <li>⚖️ <b>Data Preprocessing:</b> Includes a clear pipeline for balancing the imbalanced UCI SMS dataset via downsampling.</li>
</ul>

<hr>

<h2>📊 Dataset & Data Splits</h2>

<p>
The model is trained on the <a href="https://archive.ics.uci.edu/dataset/228/sms+spam+collection">UCI SMS Spam Collection Dataset</a>. The original dataset is imbalanced (4825 'ham' vs. 747 'spam'). To prevent bias, the 'ham' class was downsampled to 747 samples to create a 1:1 balanced dataset of 1494 total samples.
</p>

<p>
This balanced dataset was then split into training, validation, and test sets:
</p>

<table align="center">
  <tr>
    <th>Dataset Split</th>
    <th>Percentage</th>
    <th>Sample Count</th>
  </tr>
  <tr>
    <td>Training</td>
    <td>70%</td>
    <td>1045</td>
  </tr>
  <tr>
    <td>Validation</td>
    <td>10%</td>
    <td>149</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>20%</td>
    <td>300</td>
  </tr>
</table>

<hr>

<h2>🏗️ Model Architecture</h2>

<p>
The model is a from-scratch implementation of the GPT-2 (124M) architecture. The core components are the TransformerBlock and the main GPTModel class.
</p>

<h3>Transformer Block</h3>

<pre><code>class TransformerBlock(nn.Module):
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

<h3>Main GPT Model</h3>

<pre><code>class GPTModel(nn.Module):
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

<h3>Architecture Flow:</h3>
<ul>
  <li><b>Base Model:</b> GPT-2 Small (124M Parameters)</li>
  <li><b>Embedding Dim:</b> 768</li>
  <li><b>Layers:</b> 12 TransformerBlocks</li>
  <li><b>Heads:</b> 12 per attention layer</li>
  <li><b>Base Context Length:</b> 1024 tokens</li>
  <li><b>Tokenizer:</b> tiktoken (gpt2 encoding)</li>
  <li><b>Finetuning Head:</b> The out_head is replaced with an nn.Linear(768, 2) for classification.</li>
</ul>

<hr>

<h2>🔧 Data Preprocessing Pipeline</h2>

<ol>
  <li>
    <b>Download & Load:</b> The dataset is downloaded from archive.ics.uci.edu and loaded into a Pandas DataFrame.
  </li>
  <li>
    <b>Balance Dataset:</b> The imbalanced dataset is balanced by downsampling the majority 'ham' class to match the 'spam' class count (747 samples each).
    <pre><code>def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df
    </code></pre>
  </li>
  <li>
    <b>Encode Labels:</b> Labels are mapped to integers: {"ham": 0, "spam": 1}.
  </li>
  <li>
    <b>Split Data:</b> The balanced DataFrame is split into 70% train, 10% validation, and 20% test sets.
  </li>
  <li>
    <b>Tokenize & Pad:</b> A custom SpamDataset class handles tokenization using tiktoken. All sequences are padded with the pad_token_id (50256) to the maximum sequence length found in the training set (120 tokens).
  </li>
</ol>

<hr>

<h2>⚙️ Training Configuration</h2>

<p>The model was finetuned with the following hyperparameters:</p>

<table align="center">
  <tr>
    <th>Hyperparameter</th>
    <th>Value</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Optimizer</td>
    <td>AdamW</td>
    <td>AdamW optimizer</td>
  </tr>
  <tr>
    <td>Learning Rate</td>
    <td>5e-5</td>
    <td>(5 x 10^-5)</td>
  </tr>
  <tr>
    <td>Weight Decay</td>
    <td>0.1</td>
    <td>L2 Regularization</td>
  </tr>
  <tr>
    <td>Batch Size</td>
    <td>8</td>
    <td>Samples per training batch</td>
  </tr>
  <tr>
    <td>Epochs</td>
    <td>5</td>
    <td>Full passes through the training data</td>
  </tr>
  <tr>
    <td>Loss Function</td>
    <td>CrossEntropyLoss</td>
    <td>Standard for classification</td>
  </tr>
</table>

<hr>

<h2>🎨 Finetuning Strategy (Transfer Learning)</h2>

<p>
This project's core is its efficient finetuning strategy. Instead of training the entire 124M parameter model, we freeze almost all layers and only update the weights of the final layers. This is significantly faster and less resource-intensive.
</p>

<ol>
  <li>
    <b>Load Pre-trained Model:</b> A full GPTModel (124M) is instantiated and loaded with official weights from the "124M" checkpoint.
  </li>
  <li>
    <b>Freeze All Weights:</b> All parameters in the model are frozen to prevent them from being updated during training.
    <pre><code>for param in model.parameters():
    param.requires_grad = False
    </code></pre>
  </li>
  <li>
    <b>Replace Classification Head:</b> The original out_head (which maps 768 dims to 50257 vocab tokens) is replaced with a new, randomly initialized head for 2-class classification.
    <pre><code>num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"], 
    out_features=num_classes
)
    </code></pre>
  </li>
  <li>
    <b>Selectively Unfreeze Layers:</b> Only the parameters for the last TransformerBlock and the final_norm layer are unfrozen. These, along with the new out_head, are the only parts of the model that will be trained.
    <pre><code>for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True
    </code></pre>
  </li>
  <li>
    <b>Train:</b> The AdamW optimizer is initialized (it only registers parameters with requires_grad=True) and the model is trained for 5 epochs. Only the ~10M parameters of the last block are updated.
  </li>
</ol>

<hr>

<h2>🛠️ Technologies & Dependencies</h2>

<ul>
  <li><code>torch</code> (PyTorch)</li>
  <li><code>pandas</code></li>
  <li><code>numpy</code></li>
  <li><code>tiktoken</code></li>
  <li><code>matplotlib</code></li>
  <li><code>gpt_download3.py</code> (helper script, included)</li>
</ul>

<hr>

<h2>📦 Installation & Setup</h2>

<h3>1. Clone the Repository:</h3>
<pre><code>git clone https://github.com/PrayashRM/GPT-2-Spam-Classification.git
cd GPT-2-Spam-Classification
</code></pre>

<h3>2. Install Dependencies:</h3>
<pre><code>pip install torch pandas numpy tiktoken matplotlib
</code></pre>

<h3>3. Run the Notebook:</h3>
<p>The notebook (GPT_2_Finetuning_spam_ham.ipynb) is self-contained. It will download the dataset and the model weights automatically upon first run.</p>

<hr>

<h2>🚀 Usage</h2>

<p>
After training, the model can be used for inference with the classify_review function. First, load the saved model state:
</p>

<pre><code>model_state_dict = torch.load("review_classifier.pth")
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()
</code></pre>

<h3>Example 1: Spam</h3>
<pre><code>text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

# Output: spam
</code></pre>

<h3>Example 2: Ham (Not Spam)</h3>
<pre><code>text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

# Output: not spam
</code></pre>

<hr>

<h2>📁 Project Structure</h2>

<pre><code>.
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

<hr>

<h2>📈 Training & Performance Analysis</h2>

<p>
The model was trained for 5 epochs, which completed in just <b>0.99 minutes</b> on a T4 GPU. The training process was highly effective, with validation accuracy quickly matching and even exceeding training accuracy, indicating that the model learned to generalize well without overfitting.
</p>

<p>
By Epoch 3, the model already achieved 90% accuracy on both training and validation sets. By Epoch 4, it reached 97.5% validation accuracy, demonstrating the power of finetuning even a single transformer block.
</p>

<h3>End-of-Epoch Summary</h3>
<table align="center">
  <tr>
    <th>Epoch</th>
    <th>Training Accuracy</th>
    <th>Validation Accuracy</th>
  </tr>
  <tr>
    <td>1</td>
    <td>70.00%</td>
    <td>72.50%</td>
  </tr>
  <tr>
    <td>2</td>
    <td>82.50%</td>
    <td>85.00%</td>
  </tr>
  <tr>
    <td>3</td>
    <td>90.00%</td>
    <td>90.00%</td>
  </tr>
  <tr>
    <td>4</td>
    <td>100.00%</td>
    <td>97.50%</td>
  </tr>
  <tr>
    <td>5</td>
    <td>100.00%</td>
    <td>97.50%</td>
  </tr>
</table>

<h3>Final Model Performance</h3>
<p>
After 5 epochs, the model's final performance was evaluated on the unseen test set, achieving a robust accuracy of 95.67%.
</p>

<table align="center">
  <tr>
    <th>Metric</th>
    <th>Value</th>
  </tr>
  <tr style="background-color: #2d5016;">
    <td><b>Test Accuracy (Final)</b></td>
    <td><b>95.67%</b></td>
  </tr>
  <tr>
    <td>Validation Accuracy (Final)</td>
    <td>97.32%</td>
  </tr>
  <tr>
    <td>Training Accuracy (Final)</td>
    <td>97.21%</td>
  </tr>
  <tr>
    <td>Training Time</td>
    <td>0.99 minutes (T4 GPU)</td>
  </tr>
  <tr>
    <td>Parameters Trained</td>
    <td>~10M (Last Block Only)</td>
  </tr>
</table>

<h3>Key Insights</h3>
<ul>
  <li>⚡ <b>Lightning Fast Training:</b> Completed in under 1 minute on a T4 GPU by only training the final transformer block</li>
  <li>🎯 <b>High Accuracy:</b> Achieved 95.67% test accuracy with minimal overfitting</li>
  <li>💡 <b>Efficient Transfer Learning:</b> Only ~8% of model parameters were updated during training</li>
  <li>🚀 <b>Quick Convergence:</b> Model reached 90% accuracy by epoch 3, showing rapid learning</li>
</ul>

<hr>

<h2>⚖️ Why This Approach Works</h2>

<h3>Transfer Learning Benefits</h3>
<ul>
  <li>✓ <b>Leverages Pre-trained Knowledge:</b> The model starts with linguistic understanding from billions of tokens</li>
  <li>✓ <b>Minimal Data Required:</b> Only ~1500 samples needed for 95%+ accuracy</li>
  <li>✓ <b>Fast Training:</b> Under 1 minute vs hours for training from scratch</li>
  <li>✓ <b>Resource Efficient:</b> Only need to store gradients for ~10M params instead of 124M</li>
  <li>✓ <b>Better Generalization:</b> Pre-trained features help avoid overfitting on small datasets</li>
</ul>

<hr>

<h2>🔮 Future Improvements</h2>

<ul>
  <li>📱 <b>Multi-class Classification:</b> Extend to classify different types of spam (promotional, phishing, etc.)</li>
  <li>🌍 <b>Multilingual Support:</b> Train on multilingual datasets for cross-language spam detection</li>
  <li>⚡ <b>Model Optimization:</b> Implement quantization and pruning for faster inference</li>
  <li>🔄 <b>Active Learning:</b> Implement active learning pipeline for continuous improvement</li>
</ul>

<hr>

<h2>📚 References & Resources</h2>

<ul>
  <li><b>Dataset:</b> <a href="https://archive.ics.uci.edu/dataset/228/sms+spam+collection">UCI SMS Spam Collection</a></li>
  <li><b>Model:</b> <a href="https://github.com/openai/gpt-2">OpenAI GPT-2</a></li>
  <li><b>Framework:</b> <a href="https://pytorch.org/">PyTorch</a></li>
  <li><b>Tokenizer:</b> <a href="https://github.com/openai/tiktoken">TikToken</a></li>
</ul>

<hr>

<h2>🤝 Contributing</h2>

<p>
Contributions are welcome! If you'd like to improve this project, please:
</p>

<ol>
  <li>Fork the repository</li>
  <li>Create a feature branch (<code>git checkout -b feature/AmazingFeature</code>)</li>
  <li>Commit your changes (<code>git commit -m 'Add some AmazingFeature'</code>)</li>
  <li>Push to the branch (<code>git push origin feature/AmazingFeature</code>)</li>
  <li>Open a Pull Request</li>
</ol>

<hr>

<h2>⚖️ License</h2>

<p>
This project is open source and available for educational purposes. The GPT-2 model weights are subject to OpenAI's license terms.
</p>

<hr>

<h2>👨‍💻 Author</h2>

<p align="center">
  <b>Prayash Ranjan Mohanty</b><br>
  Machine Learning Enthusiast | NLP Developer<br>
  📧 <a href="mailto:prayashranjanmohanty11@gmail.com">prayashranjanmohanty11@gmail.com</a>
</p>

<p align="center">
  <a href="https://github.com/PrayashRM">
    <img src="https://img.shields.io/badge/GitHub-PrayashRM-black?logo=github&style=for-the-badge" alt="GitHub">
  </a>
  <a href="https://www.linkedin.com/in/prayash-mohanty-209303382">
    <img src="https://img.shields.io/badge/LinkedIn-Prayash_Mohanty-blue?logo=linkedin&style=for-the-badge" alt="LinkedIn">
  </a>
</p>

<hr>

<h2>🙏 Acknowledgments</h2>

<ul>
  <li>📚 <b>UCI SMS Spam Collection:</b> For providing the dataset</li>
  <li>🔥 <b>OpenAI:</b> For the GPT-2 model and pre-trained weights</li>
  <li>🎓 <b>PyTorch Team:</b> For the exceptional deep learning framework</li>
  <li>💡 <b>Open Source Community:</b> For inspiration and learning resources</li>
</ul>

<hr>

<p align="center">
  <b>⭐ If you found this project helpful, please consider giving it a star! ⭐</b><br>
  <i>"The best way to predict the future is to create it." - Abraham Lincoln</i><br><br>
  <i>Made with ❤️ using PyTorch and GPT-2 | © 2024 Prayash Ranjan Mohanty</i>
</p>

</body>
</html>
