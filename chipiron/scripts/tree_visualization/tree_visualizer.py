"""
This module contains classes for visualizing a tree structure and interacting with it.

Classes:
- PhotoViewer: A QGraphicsView widget for displaying and interacting with an image.
- Window: A QWidget that contains a PhotoViewer and provides additional functionality for tree visualization.
- VisualizeTreeScript: A script for running the tree visualization application.

Usage:
1. Create an instance of VisualizeTreeScript.
2. Call the run() method to start the visualization application.
3. Interact with the displayed tree using mouse clicks and keyboard shortcuts.
4. Call the terminate() method to finish the script.

Note: This code requires the PySide6 library and the chipiron package to be installed.
"""

import os
import pickle
import sys
import typing

from PySide6 import QtCore, QtGui, QtWidgets

from chipiron.players.move_selector.treevalue.trees.move_and_value_tree import MoveAndValueTree
from chipiron.players.move_selector.treevalue.trees.tree_visualization import display_special
from chipiron.scripts.script import Script


@typing.no_type_check
class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.Signal(QtCore.QPoint)

    @typing.no_type_check
    def __init__(self, parent):
        # FIXME hinting problem with pyside, do we have visualisation  problems related to this?
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

    @typing.no_type_check
    def hasPhoto(self):
        return not self._empty

    @typing.no_type_check
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

    @typing.no_type_check
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

    @typing.no_type_check
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

    @typing.no_type_check
    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    @typing.no_type_check
    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)

    @typing.no_type_check
    def zoomin(self):
        factor = 1.25
        self._zoom += 1
        if self._zoom > 0:
            self.scale(factor, factor)
        elif self._zoom == 0:
            self.fitInView()
        else:
            self._zoom = 0

    @typing.no_type_check
    def zoomout(self):
        factor = 1 / 1.25
        self._zoom -= 1
        if self._zoom > 0:
            self.scale(factor, factor)
        elif self._zoom == 0:
            self.fitInView()
        else:
            self._zoom = 0


@typing.no_type_check
class Window(QtWidgets.QWidget):
    # FIXME hinting problem with pyside, do we have visualisation  problems related to this?
    @typing.no_type_check
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

        self.tree: MoveAndValueTree = MoveAndValueTree(root_node=pic[1], descendants=pic[0])
        self.tree.descendants = pic[0]

        self.current_node = self.tree.root_node
        self.build_subtree()
        self.display_subtree()
        self.load_image()

    @typing.no_type_check
    def build_subtree(self):
        self.index = {}
        for ind, move in enumerate(self.current_node.moves_children):
            self.index[move] = chr(33 + ind)

    @typing.no_type_check
    def display_subtree(self):
        dot = display_special(
            node=self.current_node,
            format='jpg',
            index=self.index
        )
        dot.render('chipiron/runs/treedisplays/TreeVisualtemp')

    @typing.no_type_check
    def load_image(self):
        self.viewer.set_photo(QtGui.QPixmap('chipiron/runs/treedisplays/TreeVisualtemp.jpg'))

    @typing.no_type_check
    def pixInfo(self):
        self.viewer.toggleDragMode()

    @typing.no_type_check
    def photoClicked(self, pos):
        if self.viewer.dragMode() == QtWidgets.QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))

    @typing.no_type_check
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

    @typing.no_type_check
    def father(self):
        qq = list(self.current_node.parent_nodes)
        father_node = qq[0]  # by default!
        if father_node is not None:
            self.current_node = father_node
            self.build_subtree()
            self.display_subtree()
            self.load_image()

    @typing.no_type_check
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
    base_experiment_output_folder = os.path.join(Script.base_experiment_output_folder, 'tree_visualization/outputs/')
    base_script: Script

    def __init__(
            self,
            base_script: Script,
    ):
        ...

    def run(self) -> None:
        app = QtWidgets.QApplication(sys.argv)
        window = Window()
        window.setGeometry(0, 0, 1800, 1600)
        window.show()
        sys.exit(app.exec_())

    def terminate(self) -> None:
        """
        Finishing the script. Profiling or timing.
        """
        pass
