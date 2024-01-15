from chipiron.players.move_selector.treevalue.trees.move_and_value_tree import MoveAndValueTree
import pickle
from scripts.script import Script
import sys
from chipiron.players.move_selector.treevalue.trees.tree_visualization import display_special

from PyQt5 import QtCore, QtGui, QtWidgets
import os

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

    def set_photo(self, pixmap=None):
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
        self.btnLoad.clicked.connect(self.load_image)
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

        pic = pickle.load(open("chipiron/debugTreeData_1white-#.td", "rb"))

        self.tree: MoveAndValueTree = MoveAndValueTree(root_node=pic[1],descendants=pic[0])
        self.tree.descendants = pic[0]

        self.current_node = self.tree.root_node
        self.build_subtree()
        self.display_subtree()
        self.load_image()

    def build_subtree(self):
        self.index = {}
        for ind, move in enumerate(self.current_node.moves_children):
            self.index[move] = chr(33 + ind)

    def display_subtree(self):
        dot = display_special(node=self.current_node,
                              format='jpg',
                              index=self.index)
        dot.render('chipiron/runs/treedisplays/TreeVisualtemp')

    def load_image(self):
        self.viewer.set_photo(QtGui.QPixmap('chipiron/runs/treedisplays/TreeVisualtemp.jpg'))

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
            self.build_subtree()
            self.display_subtree()
            self.load_image()

    def move_to_son(self, key):
        for move, ind in self.index.items():
            if key == ind:
                print('switching to move', move, ind, key)
                move_to_move = move

        self.current_node = self.current_node.moves_children[move_to_move]
        self.build_subtree()
        self.display_subtree()
        self.load_image()


class VisualizeTreeScript:
    base_experiment_output_folder = os.path.join(Script.base_experiment_output_folder , 'tree_visualization/outputs/')
    base_script: Script

    def __init__(self,
                 base_script: Script,
                 ):
        self.base_script = base_script
        # Calling the init of Script that takes care of a lot of stuff, especially parsing the arguments into self.args
        args_dict: dict = self.base_script.initiate(self.base_experiment_output_folder)

    def run(self):
        app = QtWidgets.QApplication(sys.argv)
        window = Window()
        window.setGeometry(0, 0, 1800, 1600)
        window.show()
        sys.exit(app.exec_())
