.PHONY: install smoke-test data tokenizer train-router train-reconstructor package all clean help

help:
	@echo "Run these in order (or just 'make all'):"
	@echo "  make install            - install Python deps (pure pip, no Node/npm needed)"
	@echo "  make smoke-test         - confirm mermaidx renders correctly (quick sanity check)"
	@echo "  make data               - generate router + reconstructor datasets"
	@echo "  make tokenizer          - train the small BPE tokenizer on generated Mermaid sources"
	@echo "  make train-router       - train the diagram-type classifier"
	@echo "  make train-reconstructor - train the image->mermaid model"
	@echo "  make package            - zip results/ into results.zip to send back for analysis"
	@echo "  make all                - run everything above, in order"
	@echo "  make clean              - remove data/, tokenizer/, results/, results.zip"

install:
	pip install -r requirements.txt
	python3 -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
	python3 model.py

smoke-test:
	python3 generate_dataset.py --smoke-test

data: smoke-test
	python3 generate_dataset.py --out ./data --router-n-per-class 300 --reconstructor-n-per-type 400

tokenizer:
	python3 train_tokenizer.py --manifest ./data/reconstructor/manifest.jsonl --out ./tokenizer

train-router:
	python3 train_router.py --data ./data/router --epochs 15

train-reconstructor:
	python3 train_reconstructor.py --data ./data/reconstructor --tokenizer ./tokenizer --epochs 30

package:
	python3 package_results.py

all: install data tokenizer train-router train-reconstructor package
	@echo ""
	@echo "Done. Send results.zip back for analysis."

clean:
	rm -rf data tokenizer results results.zip __pycache__ common/__pycache__
