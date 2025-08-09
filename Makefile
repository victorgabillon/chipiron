ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

SYZYGY_SOURCE="https://syzygy-tables.info/download.txt?source=sesse&max-pieces=5"
SYZYGY_DESTINATION=${ROOT_DIR}/external_data/syzygy-tables/

STOCKFISH_ZIP_FILE=stock.tar
STOCKFISH_SOURCE="https://drive.google.com/file/d/1kqmgrZ2_1RwyUjAl6BOktkJx9mcSW3xG"
STOCKFISH_DESTINATION=${ROOT_DIR}/stockfish/

DATA_SOURCE="https://drive.google.com/drive/folders/1tvkuiaN-oXC7UAjUw-6cIl1PB0r2as7Y?usp=sharing"
DATA_DESTINATION=${ROOT_DIR}/external_data/

# Lichess PGN database download
LICHESS_PGN_URL="https://database.lichess.org/standard/lichess_db_standard_rated_2015-03.pgn.zst"
LICHESS_PGN_DESTINATION=${ROOT_DIR}/external_data/lichess_pgn/

.PHONY: init lichess-pgn

init: external_data/ external_data/syzygy-tables chipiron/requirements

lichess-pgn: ${LICHESS_PGN_DESTINATION}lichess_db_standard_rated_2015-03.pgn

chipiron/requirements:
	pip install -e .

external_data/syzygy-tables:
	echo "downloading SYZYGY"
	mkdir -p ${SYZYGY_DESTINATION}
	curl ${SYZYGY_SOURCE} | xargs wget -P ${SYZYGY_DESTINATION}

external_data/:
	echo "downloading Data"
	gdown --folder ${DATA_SOURCE} -O ${DATA_DESTINATION}

${LICHESS_PGN_DESTINATION}lichess_db_standard_rated_2015-03.pgn:
	echo "downloading Lichess PGN database"
	mkdir -p ${LICHESS_PGN_DESTINATION}
	wget ${LICHESS_PGN_URL} -O ${LICHESS_PGN_DESTINATION}lichess_db_standard_rated_2015-03.pgn.zst
	zstd -d ${LICHESS_PGN_DESTINATION}lichess_db_standard_rated_2015-03.pgn.zst
	rm ${LICHESS_PGN_DESTINATION}lichess_db_standard_rated_2015-03.pgn.zst
