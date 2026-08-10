.PHONY: install smoke-test data data-fixed tokenizer train-router train-reconstructor train-reconstructor-fixed package all clean help

help:
	@echo "Run these in order (or just 'make all'):"
	@echo "  make install                    - install Python deps (pure pip, no Node/npm needed)"
	@echo "  make smoke-test                 - confirm mermaidx renders correctly (quick sanity check)"
	@echo "  make data                       - generate router dataset + reconstructor VAL set + tokenizer corpus"
	@echo "                                     (reconstructor TRAIN data is generated ON-THE-FLY during"
	@echo "                                     training by default -- no train images written to disk)"
	@echo "  make data-fixed                 - ALSO generate the old fixed reconstructor train+val image set,"
	@echo "                                     only needed for 'make train-reconstructor-fixed' comparisons"
	@echo "  make tokenizer                  - train the BPE tokenizer on the (text-only) tokenizer corpus"
	@echo "  make train-router               - train the diagram-type classifier"
	@echo "  make train-reconstructor        - train the image->mermaid model (on-the-fly data, default)"
	@echo "  make train-reconstructor-fixed  - same model, but reads the fixed image set from 'make data-fixed'"
	@echo "                                     instead -- for a direct before/after comparison"
	@echo "  make package                    - zip results/ into results.zip to send back for analysis"
	@echo "  make all                        - run everything above (default on-the-fly path), in order"
	@echo "  make clean                      - remove data/, tokenizer/, results/, results.zip"

install:
	pip install -r requirements.txt
	python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
	python model.py

smoke-test:
	python generate_dataset.py --smoke-test

data: smoke-test
	python generate_dataset.py --out ./data --router-n-per-class 300 \
		--reconstructor-val-n-per-type 60 --tokenizer-corpus-n-per-type 2000

data-fixed:
	python generate_dataset.py --out ./data --skip-router --skip-reconstructor-val --skip-tokenizer-corpus \
		--fixed-train-n-per-type 400

tokenizer:
	python train_tokenizer.py --manifest ./data/reconstructor/tokenizer_corpus.jsonl --out ./tokenizer

train-router:
	python train_router.py --data ./data/router --epochs 15

train-reconstructor:
	python train_reconstructor.py --data ./data/reconstructor --tokenizer ./tokenizer --epochs 30

train-reconstructor-fixed: data-fixed
	python train_reconstructor.py --data ./data/reconstructor_fixed --tokenizer ./tokenizer --epochs 30 --no-on-the-fly

package:
	python package_results.py

all: install data tokenizer train-router train-reconstructor package
	@echo ""
	@echo "Done. Send results.zip back for analysis."

clean:
	rm -rf data tokenizer results results.zip __pycache__ common/__pycache__
