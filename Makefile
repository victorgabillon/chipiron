ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

SYZYGY_SOURCE="https://syzygy-tables.info/download.txt?source=sesse&max-pieces=5"
SYZYGY_DESTINATION=${ROOT_DIR}/external_data/syzygy-tables/

STOCKFISH_VERSION=16
STOCKFISH_URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_$(STOCKFISH_VERSION)/stockfish-ubuntu-x86-64-avx2.tar"
STOCKFISH_DESTINATION=${ROOT_DIR}/external_data/stockfish/

DATA_SOURCE="https://drive.google.com/drive/folders/1tvkuiaN-oXC7UAjUw-6cIl1PB0r2as7Y?usp=sharing"
DATA_DESTINATION=${ROOT_DIR}/external_data/


.PHONY: init lichess-pgn stockfish

init: external_data/ external_data/syzygy-tables chipiron/requirements

lichess-pgn: ${LICHESS_PGN_DESTINATION}lichess_db_standard_rated_2015-03.pgn

stockfish: ${STOCKFISH_DESTINATION}stockfish/stockfish-ubuntu-x86-64-avx2
	echo "Stockfish setup complete"

${STOCKFISH_DESTINATION}stockfish/stockfish-ubuntu-x86-64-avx2:
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

external_data/syzygy-tables:
	echo "downloading SYZYGY"
	mkdir -p ${SYZYGY_DESTINATION}
	curl ${SYZYGY_SOURCE} | xargs wget -P ${SYZYGY_DESTINATION}

external_data/:
	echo "downloading Data"
	gdown --folder ${DATA_SOURCE} -O ${DATA_DESTINATION}

