ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

SYZYGY_SOURCE="https://syzygy-tables.info/download.txt?source=sesse&max-pieces=5"
SYZYGY_DESTINATION=${ROOT_DIR}/data/syzygy-tables/

STOCKFISH_ZIP_FILE="stockfish_14.1_linux_x64.zip"
STOCKFISH_SOURCE="https://stockfishchess.org/files/stockfish_14.1_linux_x64.zip"
STOCKFISH_DESTINATION=${ROOT_DIR}/stockfish/

DATASET_SOURCE="https://drive.google.com/drive/folders/1ttXfSxwd5MiWZYze9E3D3SWKvdLa35xe?usp=sharing"
DATASET_DESTINATION=${ROOT_DIR}/data/datasets/

DATA_GUI_SOURCE="https://drive.google.com/drive/folders/1Ir76d9oHj2IGKjyoSZ3Fc1qNDH7vt99z?usp=sharing"
DATA_GUI_DESTINATION=${ROOT_DIR}/data/gui/

.PHONY: init
init: chipiron/requirements chipiron/syzygy-tables chipiron/stockfish chipiron/datasets chipiron/data_gui

chipiron/requirements:
	pip install -r requirements.txt

chipiron/syzygy-tables:
	echo "downloading SYZYGY"
	mkdir -p ${SYZYGY_DESTINATION}
	curl ${SYZYGY_SOURCE} | xargs wget -P ${SYZYGY_DESTINATION}

chipiron/stockfish:
	echo "downloading STOCKFISH"
	mkdir -p ${STOCKFISH_DESTINATION}
	wget ${STOCKFISH_SOURCE} -P ${STOCKFISH_DESTINATION}
	unzip ${STOCKFISH_DESTINATION}${STOCKFISH_ZIP_FILE} -d ${STOCKFISH_DESTINATION}
	chmod 777 stockfish/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64

chipiron/datasets:
	echo "downloading DataSets"
	gdown --folder ${DATASETS_SOURCE} -O ${DATASETS_DESTINATION}

chipiron/data_gui:
	echo "downloading Data for GUI"
	gdown --folder ${DATA_GUI_SOURCE} -O ${DATA_GUI_DESTINATION}