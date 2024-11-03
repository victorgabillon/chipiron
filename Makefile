ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

SYZYGY_SOURCE="https://syzygy-tables.info/download.txt?source=sesse&max-pieces=5"
SYZYGY_DESTINATION=${ROOT_DIR}/data/syzygy-tables/

STOCKFISH_ZIP_FILE=stock.tar
STOCKFISH_SOURCE="https://drive.google.com/file/d/1kqmgrZ2_1RwyUjAl6BOktkJx9mcSW3xG"
STOCKFISH_DESTINATION=${ROOT_DIR}/stockfish/

DATA_SOURCE="https://drive.google.com/drive/folders/1tvkuiaN-oXC7UAjUw-6cIl1PB0r2as7Y?usp=sharing"
DATA_DESTINATION=${ROOT_DIR}/data/

.PHONY: init
init:  chipiron/stockfish chipiron/data chipiron/syzygy-tables chipiron/requirements

chipiron/requirements:
	python3 -m pip install --no-cache-dir -r  requirements.txt
	pip install -e .

chipiron/syzygy-tables:
	echo "downloading SYZYGY"
	mkdir -p ${SYZYGY_DESTINATION}
	curl ${SYZYGY_SOURCE} | xargs wget -P ${SYZYGY_DESTINATION}

chipiron/stockfish:
	echo "downloading STOCKFISH"
	mkdir -p ${STOCKFISH_DESTINATION}
	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kqmgrZ2_1RwyUjAl6BOktkJx9mcSW3xG' -P ${STOCKFISH_DESTINATION} -O ${STOCKFISH_DESTINATION}${STOCKFISH_ZIP_FILE}
	tar -xf  ${STOCKFISH_DESTINATION}${STOCKFISH_ZIP_FILE} -C ${STOCKFISH_DESTINATION}
	chmod 777 stockfish/stockfish/stockfish-ubuntu-x86-64-avx2

chipiron/data:
	echo "downloading Data"
	gdown --folder ${DATA_SOURCE} -O ${DATA_DESTINATION}