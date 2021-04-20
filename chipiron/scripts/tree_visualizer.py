from src.players.treevaluebuilders.trees.move_and_value_tree import MoveAndValueTree
import pickle
from scripts.script import Script
import sys

from PyQt5 import QtCore, QtGui, QtWidgets


class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        pass
        # rect = QtCore.QRectF(self._photo.pixmap().rect())
        # if not rect.isNull():
        #     self.setSceneRect(rect)
        #     if self.hasPhoto():
        #         unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
        #         self.scale(1 / unity.width(), 1 / unity.height())
        #         viewrect = self.viewport().rect()
        #         scenerect = self.transform().mapRect(rect)
        #         factor = min(viewrect.width() / scenerect.width(),
        #                      viewrect.height() / scenerect.height())
        #         self.scale(factor, factor)
        #     self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)

    def zoomin(self):
        factor = 1.25
        self._zoom += 1
        if self._zoom > 0:
            self.scale(factor, factor)
        elif self._zoom == 0:
            self.fitInView()
        else:
            self._zoom = 0

    def zoomout(self):
        factor = 1 / 1.25
        self._zoom -= 1
        if self._zoom > 0:
            self.scale(factor, factor)
        elif self._zoom == 0:
            self.fitInView()
        else:
            self._zoom = 0


class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.viewer = PhotoViewer(self)
        # 'Load image' button
        self.btnLoad = QtWidgets.QToolButton(self)
        self.btnLoad.setText('Load image')
        self.btnLoad.clicked.connect(self.loadImage)
        # Button to change from drag/pan to getting pixel info
        self.btnPixInfo = QtWidgets.QToolButton(self)
        self.btnPixInfo.setText('Enter pixel info mode')
        self.btnPixInfo.clicked.connect(self.pixInfo)
        self.editPixInfo = QtWidgets.QLineEdit(self)
        self.editPixInfo.setReadOnly(True)
        self.viewer.photoClicked.connect(self.photoClicked)
        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        VBlayout.addWidget(self.viewer)
        HBlayout = QtWidgets.QHBoxLayout()
        HBlayout.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout.addWidget(self.btnLoad)
        HBlayout.addWidget(self.btnPixInfo)
        HBlayout.addWidget(self.editPixInfo)
        VBlayout.addLayout(HBlayout)

        self.tree = MoveAndValueTree(None, None, None)
        pic = pickle.load(open("chipiron/runs/treedisplays/TreeData_9white-#.td", "rb"))
        self.tree.descendants = pic[0]
        self.tree.root_node = pic[1]

        self.current_node = self.tree.root_node
        self.buildSubtree()
        self.displaySubtree()
        self.loadImage()

    def buildSubtree(self):
        self.index = {}
        for ind, move in enumerate(self.current_node.moves_children):
            self.index[move] = chr(33 + ind)

    def displaySubtree(self):
        dot = self.tree.display_special(self.current_node, 'jpg', self.index)
        dot.render('chipiron/runs/treedisplays/TreeVisualtemp')

    def loadImage(self):
        self.viewer.setPhoto(QtGui.QPixmap('chipiron/runs/treedisplays/TreeVisualtemp.jpg'))

    def pixInfo(self):
        self.viewer.toggleDragMode()

    def photoClicked(self, pos):
        if self.viewer.dragMode() == QtWidgets.QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))

    def keyPressEvent(self, event):
        print(event.text())
        key = event.text()
        if key in self.index.values():
            self.move_to_son(key)
        if key == '9':
            self.viewer.zoomin()
        if key == '0':
            self.viewer.zoomout()
        if key == 'z':
            self.father()

            # todo there is collision as these numbers are used for children too now...

    def father(self):
        qq = list(self.current_node.parent_nodes)
        father_node = qq[0]  # by default!
        if father_node is not None:
            self.current_node = father_node
            self.buildSubtree()
            self.displaySubtree()
            self.loadImage()

    def move_to_son(self, key):
        for move, ind in self.index.items():
            if key == ind:
                print('switching to move', move, ind, key)
                move_to_move = move

        self.current_node = self.current_node.moves_children[move_to_move]
        self.buildSubtree()
        self.displaySubtree()
        self.loadImage()


class VisualizeTreeScript(Script):

    def __init__(self):
        super().__init__()

    def run(self):
        app = QtWidgets.QApplication(sys.argv)
        window = Window()
        window.setGeometry(0, 0, 1800, 1600)
        window.show()
        sys.exit(app.exec_())
