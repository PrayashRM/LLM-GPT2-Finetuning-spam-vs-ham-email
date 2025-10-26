<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-2 Finetuning for Spam Classification</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        pre code { font-family: 'Menlo', 'Monaco', 'Consolas', monospace; }
        pre::-webkit-scrollbar { height: 8px; }
        pre::-webkit-scrollbar-track { background: #1f2937; border-radius: 4px; }
        pre::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
    </style>
</head>
<body class="bg-gray-900 text-gray-200">

    <main class="max-w-4xl mx-auto p-8 bg-gray-800 shadow-2xl rounded-2xl my-8 ring-1 ring-gray-700">

        <!-- HEADER -->
        <header class="text-center mb-8">
            <h1 class="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 pb-2">
                🔢 GPT-2 Spam Classification
            </h1>
            <p class="text-lg text-gray-400 mt-2"><b>PyTorch | NLP | Transfer Learning</b></p>
            <p class="text-xl text-gray-300 mt-3">
                Finetuning GPT-2 to classify SMS messages with <b>95.67% Test Accuracy</b> 🎯
            </p>
            <div class="flex justify-center flex-wrap gap-2 my-6">
                <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&style=for-the-badge" alt="Python">
                <img src="https://img.shields.io/badge/PyTorch-LTS-orange?logo=pytorch&style=for-the-badge" alt="PyTorch">
                <img src="https://img.shields.io/badge/Model-GPT--2_124M-green?style=for-the-badge" alt="GPT-2">
                <img src="https://img.shields.io/badge/Accuracy-95.67%25-brightgreen?style=for-the-badge" alt="Accuracy">
            </div>
        </header>

        <hr class="border-gray-700 my-8">

        <!-- OVERVIEW -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3">🚀</span> Overview
            </h2>
            <p class="text-gray-300 mb-4 leading-relaxed">
                This project demonstrates transfer learning by adapting GPT-2 (124M parameters) for spam classification. 
                The model uses pre-trained weights from OpenAI and is efficiently finetuned by freezing most layers and 
                only training the final transformer block and layer normalization.
            </p>
            <p class="text-gray-300 mb-4 leading-relaxed">
                Achieves <strong>95.67% accuracy</strong> on the test set after just 5 epochs, completing training in 
                under 1 minute on a T4 GPU.
            </p>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- KEY FEATURES -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3">🎯</span> Key Features
            </h2>
            <ul class="space-y-3 text-gray-300">
                <li>✨ <strong>GPT-2 from Scratch:</strong> Complete implementation of GPT-2 architecture</li>
                <li>⚙️ <strong>Efficient Finetuning:</strong> Only ~10M parameters trained (8% of total model)</li>
                <li>📊 <strong>High Accuracy:</strong> 95.67% on test set after 5 epochs</li>
                <li>💾 <strong>Pre-trained Weights:</strong> Uses OpenAI's official 124M checkpoint</li>
                <li>🔡 <strong>Modern Tokenization:</strong> TikToken for text encoding</li>
            </ul>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- DATASET -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3">📊</span> Dataset & Splits
            </h2>
            <p class="text-gray-300 mb-4">
                Uses the <a href="https://archive.ics.uci.edu/dataset/228/sms+spam+collection" 
                class="text-blue-400 hover:underline">UCI SMS Spam Collection Dataset</a>. 
                Original imbalanced dataset (4825 ham, 747 spam) was balanced by downsampling.
            </p>
            <div class="overflow-x-auto rounded-lg ring-1 ring-gray-700">
                <table class="w-full text-left">
                    <thead class="bg-gray-700 text-gray-100">
                        <tr>
                            <th class="p-3">Split</th>
                            <th class="p-3">Percentage</th>
                            <th class="p-3">Samples</th>
                        </tr>
                    </thead>
                    <tbody class="bg-gray-800 divide-y divide-gray-700">
                        <tr><td class="p-3">Training</td><td class="p-3">70%</td><td class="p-3">1045</td></tr>
                        <tr><td class="p-3">Validation</td><td class="p-3">10%</td><td class="p-3">149</td></tr>
                        <tr><td class="p-3">Test</td><td class="p-3">20%</td><td class="p-3">300</td></tr>
                    </tbody>
                </table>
            </div>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- TRAINING CONFIG -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3">⚙️</span> Training Configuration
            </h2>
            <div class="overflow-x-auto rounded-lg ring-1 ring-gray-700">
                <table class="w-full text-left">
                    <thead class="bg-gray-700 text-gray-100">
                        <tr><th class="p-3">Parameter</th><th class="p-3">Value</th></tr>
                    </thead>
                    <tbody class="bg-gray-800 divide-y divide-gray-700">
                        <tr><td class="p-3">Optimizer</td><td class="p-3">AdamW</td></tr>
                        <tr><td class="p-3">Learning Rate</td><td class="p-3">5e-5</td></tr>
                        <tr><td class="p-3">Weight Decay</td><td class="p-3">0.1</td></tr>
                        <tr><td class="p-3">Batch Size</td><td class="p-3">8</td></tr>
                        <tr><td class="p-3">Epochs</td><td class="p-3">5</td></tr>
                        <tr><td class="p-3">Loss Function</td><td class="p-3">CrossEntropyLoss</td></tr>
                    </tbody>
                </table>
            </div>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- PERFORMANCE -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3">📈</span> Performance Results
            </h2>
            
            <h3 class="text-xl font-semibold mb-3 text-blue-300">Training Progress</h3>
            <div class="overflow-x-auto rounded-lg ring-1 ring-gray-700 mb-6">
                <table class="w-full text-left">
                    <thead class="bg-gray-700 text-gray-100">
                        <tr>
                            <th class="p-3">Epoch</th>
                            <th class="p-3">Train Acc</th>
                            <th class="p-3">Val Acc</th>
                        </tr>
                    </thead>
                    <tbody class="bg-gray-800 divide-y divide-gray-700">
                        <tr><td class="p-3">1</td><td class="p-3">70.00%</td><td class="p-3">72.50%</td></tr>
                        <tr><td class="p-3">2</td><td class="p-3">82.50%</td><td class="p-3">85.00%</td></tr>
                        <tr><td class="p-3">3</td><td class="p-3">90.00%</td><td class="p-3">90.00%</td></tr>
                        <tr><td class="p-3">4</td><td class="p-3">100.00%</td><td class="p-3">97.50%</td></tr>
                        <tr><td class="p-3">5</td><td class="p-3">100.00%</td><td class="p-3">97.50%</td></tr>
                    </tbody>
                </table>
            </div>

            <h3 class="text-xl font-semibold mb-3 text-blue-300">Final Performance</h3>
            <div class="overflow-x-auto rounded-lg ring-1 ring-gray-700 mb-6">
                <table class="w-full text-left">
                    <thead class="bg-gray-700 text-gray-100">
                        <tr><th class="p-3">Metric</th><th class="p-3">Value</th></tr>
                    </thead>
                    <tbody class="bg-gray-800 divide-y divide-gray-700">
                        <tr class="bg-green-900/30">
                            <td class="p-3 font-medium">Test Accuracy</td>
                            <td class="p-3 font-bold text-green-400">95.67%</td>
                        </tr>
                        <tr><td class="p-3">Validation Accuracy</td><td class="p-3">97.32%</td></tr>
                        <tr><td class="p-3">Training Accuracy</td><td class="p-3">97.21%</td></tr>
                        <tr><td class="p-3">Training Time</td><td class="p-3">0.99 minutes</td></tr>
                    </tbody>
                </table>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="bg-gradient-to-br from-blue-900/40 to-purple-900/40 p-5 rounded-lg ring-1 ring-blue-500/30">
                    <div class="text-3xl mb-2">⚡</div>
                    <h4 class="font-semibold text-lg mb-2 text-blue-300">Fast Training</h4>
                    <p class="text-sm text-gray-300">Under 1 minute on T4 GPU</p>
                </div>
                <div class="bg-gradient-to-br from-green-900/40 to-teal-900/40 p-5 rounded-lg ring-1 ring-green-500/30">
                    <div class="text-3xl mb-2">🎯</div>
                    <h4 class="font-semibold text-lg mb-2 text-green-300">High Accuracy</h4>
                    <p class="text-sm text-gray-300">95.67% test accuracy</p>
                </div>
            </div>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- INSTALLATION -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3">📦</span> Installation
            </h2>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mb-4"><code>git clone https://github.com/PrayashRM/GPT-2-Spam-Classification.git
cd GPT-2-Spam-Classification
pip install torch pandas numpy tiktoken matplotlib</code></pre>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- USAGE -->
        <section>
            <h2 class="text-3xl font-bold mb-4 text-white flex items-center">
                <span class="mr-3">🚀</span> Usage
            </h2>
            <h3 class="text-lg font-semibold mb-2 text-blue-300">Example: Spam Detection</h3>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700 mb-4"><code>text = "You won $1000! Click here to claim your prize"
result = classify_review(text, model, tokenizer, device)
print(result)  # Output: spam</code></pre>

            <h3 class="text-lg font-semibold mb-2 text-blue-300">Example: Ham Detection</h3>
            <pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm ring-1 ring-gray-700"><code>text = "Hey, are we still meeting for dinner tonight?"
result = classify_review(text, model, tokenizer, device)
print(result)  # Output: not spam</code></pre>
        </section>

        <hr class="border-gray-700 my-8">

        <!-- AUTHOR -->
        <footer class="text-center mt-12">
            <div class="bg-gradient-to-r from-blue-900/30 to-purple-900/30 p-8 rounded-xl ring-1 ring-blue-500/30">
                <div class="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full mx-auto mb-4 flex items-center justify-center text-3xl font-bold">
                    PM
                </div>
                <p class="text-2xl font-bold text-white mb-2">Prayash Ranjan Mohanty</p>
                <p class="text-gray-400 mb-4">ML Enthusiast | NLP Developer</p>
                
                <div class="flex justify-center gap-4 mb-6">
                    <a href="https://github.com/PrayashRM" 
                       class="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg transition">
                        GitHub
                    </a>
                    <a href="https://www.linkedin.com/in/prayash-mohanty-209303382" 
                       class="bg-blue-700 hover:bg-blue-600 px-4 py-2 rounded-lg transition">
                        LinkedIn
                    </a>
                </div>
                
                <p class="text-gray-400">
                    📧 <a href="mailto:prayashranjanmohanty11@gmail.com" class="text-blue-400 hover:underline">
                        prayashranjanmohanty11@gmail.com
                    </a>
                </p>
            </div>
            
            <div class="mt-8 pt-6 border-t border-gray-700">
                <p class="text-sm text-gray-500 italic">
                    "The best way to predict the future is to create it."
                </p>
            </div>
        </footer>

    </main>

</body>
</html>
