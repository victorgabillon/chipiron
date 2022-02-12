ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

SYZYGY_SOURCE="https://syzygy-tables.info/download.txt?source=sesse&max-pieces=5"
SYZYGY_DESTINATION=${ROOT_DIR}/chipiron/syzygy-tables/

STOCKFISH_ZIP_FILE="stockfish_14.1_linux_x64.zip"
STOCKFISH_SOURCE="https://stockfishchess.org/files/stockfish_14.1_linux_x64.zip"
STOCKFISH_DESTINATION=${ROOT_DIR}/chipiron/stockfish/

DATASET_SOURCE="https://mega.nz/file/8VkWnIhK#CCFfuzMvKnWzbuvvvcsH_AezrsexEP1LociMuTyz24A"
DATASET_SOURCE2="https://mega.nz/file/8YtS3CaY#qxvht1p2Soswjad434fCtgbr5orNOkQ8IWZiTg4K21A"
DATASET_DESTINATION=${ROOT_DIR}/chipiron/data/datasets/


.PHONY: init
init: chipiron/syzygy-tables chipiron/stockfish chipiron/datasets
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
	chmod 777 chipiron/stockfish/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64

chipiron/datasets:
	echo "downloading Datasets"
	wget ${DATASET_SOURCE} -P ${DATASET_DESTINATION}
	wget ${DATASET_SOURCE2} -P ${DATASET_DESTINATION}
