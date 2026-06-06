# ── CELL 2: Create conda env + install dependencies ──
# Run after runtime restarts from condacolab

!mamba create -n sep python=3.9 -y

!mamba run -n sep pip install torch==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

!mamba run -n sep pip install \
    "openai==1.12.0" "tenacity==8.2.3" "tiktoken==0.6.0" \
    "transformers==4.34.1" "pandas==2.2.0" "scikit-learn==1.4.0" \
    "bitsandbytes>=0.43.0" "datasets==2.14.7" "sentencepiece==0.1.99" \
    "peft==0.6.2" "evaluate==0.4.1" "trl==0.7.1" "protobuf==4.25.2" \
    "accelerate==0.21.0" "python-dotenv" "ta" "yfinance" \
    "httpx<0.28.0" -q

print("Environment ready.")
