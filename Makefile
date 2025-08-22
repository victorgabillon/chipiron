# Include environment variables from .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Use environment variables with fallbacks
SYZYGY_SOURCE?=https://syzygy-tables.info/download.txt?source=sesse&max-pieces=5
SYZYGY_DESTINATION?=${ROOT_DIR}/$(SYZYGY_TABLES_DIR)

STOCKFISH_VERSION?=16
STOCKFISH_URL?=https://github.com/official-stockfish/Stockfish/releases/download/sf_$(STOCKFISH_VERSION)/stockfish-ubuntu-x86-64-avx2.tar
STOCKFISH_DESTINATION?=${ROOT_DIR}/$(STOCKFISH_DIR)

DATA_SOURCE?=https://drive.google.com/drive/folders/1tvkuiaN-oXC7UAjUw-6cIl1PB0r2as7Y?usp=sharing
DATA_DESTINATION?=${ROOT_DIR}/$(EXTERNAL_DATA_DIR)

.PHONY: init init-no-syzygy lichess-pgn stockfish

init: chipiron/requirements $(EXTERNAL_DATA_DIR)/ $(SYZYGY_TABLES_DIR) stockfish

init-no-syzygy: chipiron/requirements $(EXTERNAL_DATA_DIR)/ stockfish

lichess-pgn: ${LICHESS_PGN_DIR}/lichess_db_standard_rated_2015-03.pgn

stockfish: ${STOCKFISH_DESTINATION}/stockfish/stockfish-ubuntu-x86-64-avx2
	echo "Stockfish setup complete"

${STOCKFISH_DESTINATION}/stockfish/stockfish-ubuntu-x86-64-avx2:
	echo "Downloading and setting up Stockfish..."
	mkdir -p ${STOCKFISH_DESTINATION}
	cd ${STOCKFISH_DESTINATION} && \
	curl -L ${STOCKFISH_URL} -o stockfish-ubuntu-x86-64-avx2.tar && \
	tar -xf stockfish-ubuntu-x86-64-avx2.tar && \
	rm stockfish-ubuntu-x86-64-avx2.tar && \
	chmod +x stockfish/stockfish-ubuntu-x86-64-avx2
	echo "Stockfish installed at ${STOCKFISH_DESTINATION}stockfish/stockfish-ubuntu-x86-64-avx2"

chipiron/requirements:
	pip install -e .

$(SYZYGY_TABLES_DIR):
	echo "downloading SYZYGY tables (this may take 10-20 minutes)"
	mkdir -p ${SYZYGY_DESTINATION}
	@echo "Downloading Syzygy tables sequentially to avoid server overload..."
	@curl -s ${SYZYGY_SOURCE} | head -50 | while read url; do \
		echo "Downloading $$url"; \
		wget -t 3 -T 30 -c -P ${SYZYGY_DESTINATION} "$$url" || echo "Failed to download $$url (continuing...)"; \
	done
	@echo "Syzygy table download completed (downloaded first 50 tables for basic functionality)"

$(EXTERNAL_DATA_DIR)/: chipiron/requirements
	echo "downloading Data"
	gdown --folder ${DATA_SOURCE} -O ${DATA_DESTINATION}

