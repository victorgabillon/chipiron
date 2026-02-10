# Include environment variables from .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Warn if key variables are set in the environment
ifeq ($(origin EXTERNAL_DATA_DIR), environment)
    $(warning EXTERNAL_DATA_DIR was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin LICHESS_PGN_DIR), environment)
    $(warning LICHESS_PGN_DIR was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin SYZYGY_TABLES_DIR), environment)
    $(warning SYZYGY_TABLES_DIR was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin STOCKFISH_DIR), environment)
    $(warning STOCKFISH_DIR was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin GUI_DIR), environment)
    $(warning GUI_DIR was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin LICHESS_PGN_FILE), environment)
    $(warning LICHESS_PGN_FILE was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin STOCKFISH_BINARY_PATH), environment)
    $(warning STOCKFISH_BINARY_PATH was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin STOCKFISH_VERSION), environment)
    $(warning STOCKFISH_VERSION was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin SYZYGY_SOURCE), environment)
    $(warning SYZYGY_SOURCE was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin STOCKFISH_URL), environment)
    $(warning STOCKFISH_URL was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin DATA_SOURCE), environment)
    $(warning DATA_SOURCE was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin DATA_DESTINATION), environment)
    $(warning DATA_DESTINATION was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin ML_FLOW_URI_PATH), environment)
    $(warning ML_FLOW_URI_PATH was set in the environment and will override .env and Makefile defaults)
endif
ifeq ($(origin ML_FLOW_URI_PATH_TEST), environment)
    $(warning ML_FLOW_URI_PATH_TEST was set in the environment and will override .env and Makefile defaults)
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

.PHONY: init init-no-syzygy lichess-pgn stockfish syzygy-tables

init: src/chipiron/requirements $(EXTERNAL_DATA_DIR)/ $(SYZYGY_TABLES_DIR)/.syzygy-complete stockfish

init-no-syzygy: src/chipiron/requirements $(EXTERNAL_DATA_DIR)/ stockfish

syzygy-tables: $(SYZYGY_TABLES_DIR)/.syzygy-complete

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

src/chipiron/requirements:
	pip install -e .

$(SYZYGY_TABLES_DIR)/.syzygy-complete:
	echo "downloading SYZYGY tables (this may take 10-20 minutes)"
	mkdir -p ${SYZYGY_DESTINATION}
	@echo "Downloading Syzygy tables sequentially to avoid server overload..."
	@curl -s "${SYZYGY_SOURCE}" | while read url; do \
		echo "Downloading $$url"; \
		wget -t 3 -T 30 -c -P ${SYZYGY_DESTINATION} "$$url" || echo "Failed to download $$url (continuing...)"; \
	done
	@echo "Syzygy table download completed (downloaded first 50 tables for basic functionality)"
	@if [ `find ${SYZYGY_DESTINATION} -name "*.rtb*" | wc -l` -gt 0 ]; then \
		echo "✅ Verified Syzygy files downloaded successfully"; \
		touch $(SYZYGY_TABLES_DIR)/.syzygy-complete; \
	else \
		echo "❌ No Syzygy files found after download"; \
		exit 1; \
	fi

$(EXTERNAL_DATA_DIR)/: chipiron/requirements
	echo "downloading Data"
	gdown --folder ${DATA_SOURCE} -O ${DATA_DESTINATION}

