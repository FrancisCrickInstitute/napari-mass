# Using Annotated: https://napari.org/stable/gallery/magic_parameter_sweep.html
# magicgui: https://forum.image.sc/t/magicgui-select-folder-instead-of-file/76378
# multiple viewer: https://napari.org/stable/gallery/multiple_viewer_widget.html
# https://forum.image.sc/t/how-to-maximize-this-custom-mulitple-view-grid/84400


from pathlib import Path
from typing import Annotated
import napari
import yaml
from magicgui import magicgui
from napari.qt import QtViewer
from napari.components.viewer_model import ViewerModel
from qtpy.QtCore import Qt
from qtpy.QtWidgets import *


class QtViewerWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer
        self.setAcceptDrops(True)

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin: str = None,
        layer_type: str = None,
        **kwargs,
    ):
        """for drag and drop open files"""
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )

    def dropEvent(self, event):
        """Triggered when the user moves & drops one of the items in the list.

        Parameters
        ----------
        event : QEvent
            The event that triggered the dropEvent.
        """
        self.main_viewer.window._qt_viewer.dropEvent(event)


# TODO: use magicgui to create initial path selection (without auto-generate) and all other controls (https://pyapp-kit.github.io/magicgui/)
class MassWidget(QWidget):
    DEFAULT_PARAMS_NAME = 'parameters.yml'
    DEFAULT_PARAMS_NAME = 'D:/slides/EM04613/output/parameters.yml'

    def __init__(self, napari_viewer: napari.Viewer,
                 working_path: Annotated[Path, {'mode': 'w'}] = DEFAULT_PARAMS_NAME
                 ):
        super().__init__()
        self.viewer = napari_viewer

        viewer_model = ViewerModel()
        detail_viewer = QtViewerWrap(self.viewer, viewer_model)

        tab_widget = self.create_widgets_from_config(str(working_path))
        splitter = QSplitter()
        splitter.setOrientation(Qt.Vertical)
        splitter.addWidget(tab_widget)
        splitter.addWidget(detail_viewer)

        self.viewer.window.add_dock_widget(splitter)

    def create_widgets_from_config(self, path):
        with open(str(path), 'r') as infile:
            param_dict = yaml.load(infile, Loader=yaml.Loader)
        tab_widget = QTabWidget()
        for label, section_dict in param_dict.items():
            widget = create_params_widget(section_dict)
            tab_widget.addTab(widget, label)
        return tab_widget


def create_params_widget(param_dict):
    """
    Creates the widget for a specific model's parameters to swap in and out
    """
    widget = QWidget()
    layout = QGridLayout()
    layout.setAlignment(Qt.AlignTop)

    for i, (label, value) in enumerate(param_dict.items()):
        param_label = QLabel(label.replace('_', ' '))
        #param_label.setToolTip(value.tooltip)
        layout.addWidget(param_label, i, 0)
        if isinstance(value, dict):
            sub_items = create_params_widget(value)
            param_widget = QGroupBox()
            param_widget.setLayout(sub_items.layout())
        elif isinstance(value, bool):
            param_widget = QCheckBox()
            if value:
                param_widget.setChecked(True)
            else:
                param_widget.setChecked(False)
        else:
            param_widget = QLineEdit()
            param_widget.setText(str(value))
        layout.addWidget(param_widget, i, 1)
    # Tighten up margins and set layout
    layout.setContentsMargins(0, 0, 0, 0)
    widget.setLayout(layout)
    return widget
