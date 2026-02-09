"""Document the module contains classes for visualizing a tree structure and interacting with it.

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
from typing import Any

from anemone.trees.tree import Tree
from anemone.trees.tree_visualization import (
    display_special,
)
from PySide6 import QtCore, QtGui, QtWidgets

from chipiron.scripts.iscript import IScript
from chipiron.scripts.script import Script


@typing.no_type_check
class PhotoViewer(QtWidgets.QGraphicsView):
    """A custom QGraphicsView widget for displaying photos.

    Signals:
        - photoClicked: Signal emitted when the photo is clicked. It provides the position of the click as a QPoint.

    Methods:
        - hasPhoto(): Check if the viewer has a photo loaded.
        - fitInView(scale=True): Fit the photo within the view.
        - set_photo(pixmap=None): Set the photo to be displayed.
        - wheelEvent(event): Handle the wheel event for zooming.
        - toggleDragMode(): Toggle the drag mode between ScrollHandDrag and NoDrag.
        - mousePressEvent(event): Handle the mouse press event.
        - zoomin(): Zoom in the photo.
        - zoomout(): Zoom out the photo.

    """

    photo_clicked = QtCore.Signal(QtCore.QPoint)

    @typing.no_type_check
    def __init__(self, parent: QtWidgets.QWidget | None) -> None:
        """Initialize the PhotoViewer.

        Args:
            parent (QWidget): The parent widget.

        """
        super().__init__(parent)
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
    def has_photo(self) -> bool:
        """Check if the viewer has a photo loaded.

        Returns:
            bool: True if a photo is loaded, False otherwise.

        """
        return not self._empty

    @typing.no_type_check
    @typing.override
    def fitInView(self, scale: bool = True) -> None:
        """Fit the photo within the view.

        Args:
            scale (bool): Whether to scale the photo to fit the view. Default is True.

        """

    @typing.no_type_check
    def set_photo(self, pixmap: QtGui.QPixmap | None = None) -> None:
        """Set the photo to be displayed.

        Args:
            pixmap (QPixmap): The photo to be displayed. If None, the viewer will be empty.

        """
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
    @typing.override
    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Handle the wheel event for zooming.

        Args:
            event (QWheelEvent): The wheel event object.

        """
        if self.has_photo():
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
    def toggle_drag_mode(self) -> None:
        """Toggle the drag mode between ScrollHandDrag and NoDrag."""
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    @typing.no_type_check
    @typing.override
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle the mouse press event.

        Args:
            event (QMouseEvent): The mouse press event object.

        """
        if self._photo.isUnderMouse():
            self.photo_clicked.emit(self.mapToScene(event.pos()).toPoint())
        super().mousePressEvent(event)

    @typing.no_type_check
    def zoomin(self) -> None:
        """Zoom in the photo."""
        factor = 1.25
        self._zoom += 1
        if self._zoom > 0:
            self.scale(factor, factor)
        elif self._zoom == 0:
            self.fitInView()
        else:
            self._zoom = 0

    @typing.no_type_check
    def zoomout(self) -> None:
        """Zoom out the photo."""
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
    """A class representing a window for tree visualization.

    Attributes:
        viewer (PhotoViewer): The photo viewer widget.
        btnLoad (QtWidgets.QToolButton): The button to load an image.
        btnPixInfo (QtWidgets.QToolButton): The button to enter pixel info mode.
        editPixInfo (QtWidgets.QLineEdit): The line edit widget for pixel info.
        tree (MoveAndValueTree): The tree object for visualization.
        current_node (Node): The current node being displayed.
        index (dict): A dictionary mapping moves to indices.

    Methods:
        __init__(): Initializes the Window object.
        build_subtree(): Builds the subtree.
        display_subtree(): Displays the subtree.
        load_image(): Loads an image.
        pixInfo(): Toggles between drag/pan and pixel info mode.
        photoClicked(): Handles the photo clicked event.
        keyPressEvent(): Handles the key press event.
        father(): Moves to the parent node.
        move_to_son(): Moves to the specified child node.

    """

    @typing.no_type_check
    def __init__(self) -> None:
        """Initialize the Window class.

        This method sets up the GUI elements and initializes the necessary variables.
        It also loads the image, builds the subtree, and displays the subtree.

        Args:
            None

        Returns:
            None

        """
        super().__init__()
        self.viewer = PhotoViewer(self)
        # 'Load image' button
        self.btn_load = QtWidgets.QToolButton(self)
        self.btn_load.setText("Load image")
        self.btn_load.clicked.connect(self.load_image)  # pylint: disable=no-member
        # Button to change from drag/pan to getting pixel info
        self.btn_pix_info = QtWidgets.QToolButton(self)
        self.btn_pix_info.setText("Enter pixel info mode")
        self.btn_pix_info.clicked.connect(self.pix_info)  # pylint: disable=no-member
        self.edit_pix_info = QtWidgets.QLineEdit(self)
        self.edit_pix_info.setReadOnly(True)
        self.viewer.photo_clicked.connect(self.photo_clicked)
        # Arrange layout
        v_layout = QtWidgets.QVBoxLayout(self)
        v_layout.addWidget(self.viewer)
        h_layout = QtWidgets.QHBoxLayout()
        h_layout.setAlignment(QtCore.Qt.AlignLeft)
        h_layout.addWidget(self.btn_load)
        h_layout.addWidget(self.btn_pix_info)
        h_layout.addWidget(self.edit_pix_info)
        v_layout.addLayout(h_layout)

        with open("chipiron/debugTreeData_1white-#.td", "rb") as tree_data_file:
            pic = pickle.load(tree_data_file)

        self.tree: Tree[Any] = Tree(root_node=pic[1], descendants=pic[0])
        self.tree.descendants = pic[0]

        self.current_node = self.tree.root_node
        self.build_subtree()
        self.display_subtree()
        self.load_image()

    @typing.no_type_check
    def build_subtree(self) -> None:
        """Build the subtree of the current node.

        This method iterates over the moves of the current node and assigns an index to each move.
        The index is stored in the `index` dictionary, where the move is the key and the index is the value.

        Returns:
            None

        """
        self.index = {}
        for ind, move in enumerate(self.current_node.moves_children):
            self.index[move] = chr(33 + ind)

    @typing.no_type_check
    def display_subtree(self) -> None:
        """Display the subtree rooted at the current node.

        This method generates a visualization of the subtree rooted at the current node and saves it as a JPG image file.

        Args:
            None

        Returns:
            None

        """
        dot = display_special(
            node=self.current_node, format_str="jpg", index=self.index
        )
        dot.render("chipiron/runs/treedisplays/TreeVisualtemp")

    @typing.no_type_check
    def load_image(self) -> None:
        """Load an image and sets it as the photo for the viewer.

        The image is loaded from the file 'chipiron/runs/treedisplays/TreeVisualtemp.jpg'.

        Args:
            None

        Returns:
            None

        """
        self.viewer.set_photo(
            QtGui.QPixmap("chipiron/runs/treedisplays/TreeVisualtemp.jpg")
        )

    @typing.no_type_check
    def pix_info(self) -> None:
        """Toggles the drag mode of the viewer.

        This method is used to toggle the drag mode of the viewer. It switches between the modes of dragging and not dragging
        the image.

        Args:
            None

        Returns:
            None

        """
        self.viewer.toggle_drag_mode()

    @typing.no_type_check
    def photo_clicked(self, pos: QtCore.QPoint) -> None:
        """Handle the event when a photo is clicked.

        Args:
            pos (QPoint): The position of the click.

        Returns:
            None

        """
        if self.viewer.dragMode() == QtWidgets.QGraphicsView.NoDrag:
            self.edit_pix_info.setText(f"{pos.x()}, {pos.y()}")

    @typing.no_type_check
    @typing.override
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key press events.

        Args:
            event (QKeyEvent): The key press event.

        Returns:
            None

        """
        print(event.text())
        key = event.text()
        if key in self.index.values():
            self.move_to_son(key)
        if key == "9":
            self.viewer.zoomin()
        if key == "0":
            self.viewer.zoomout()
        if key == "z":
            self.father()

        # TODO: there is collision as these numbers are used for children too now...

    @typing.no_type_check
    def father(self) -> None:
        """Move to the parent node of the current node and perform necessary operations."""
        qq = list(self.current_node.parent_nodes.keys())
        father_node = qq[0]  # by default!
        if father_node is not None:
            self.current_node = father_node
            self.build_subtree()
            self.display_subtree()
            self.load_image()

    @typing.no_type_check
    def move_to_son(self, key: str) -> None:
        """Move to the specified son node based on the given key.

        Args:
            key: The key of the son node to move to.

        Returns:
            None

        Raises:
            None

        """
        move_to_move = None
        for move, ind in self.index.items():
            if key == ind:
                print("switching to move", move, ind, key)
                move_to_move = move

        assert move_to_move is not None
        self.current_node = self.current_node.moves_children[move_to_move]
        self.build_subtree()
        self.display_subtree()
        self.load_image()


class VisualizeTreeScript(IScript):
    """Describe the class represents a script for visualizing a tree."""

    base_experiment_output_folder = os.path.join(
        Script.base_experiment_output_folder, "tree_visualization/outputs/"
    )
    base_script: Script

    def __init__(
        self,
        base_script: Script,
    ) -> None:
        """Initialize a TreeVisualizer object.

        Args:
            base_script (Script): The base script to visualize.

        Returns:
            None

        """

    def run(self) -> None:
        """Run the tree visualizer application."""
        app = QtWidgets.QApplication(sys.argv)
        window = Window()
        window.setGeometry(0, 0, 1800, 1600)
        window.show()
        sys.exit(app.exec_())

    def terminate(self) -> None:
        """Finishing the script. Profiling or timing."""
