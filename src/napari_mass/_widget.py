# Using Annotated: https://napari.org/stable/gallery/magic_parameter_sweep.html
# magicgui: https://forum.image.sc/t/magicgui-select-folder-instead-of-file/76378

import json
from pathlib import Path
from typing import Annotated

from qtpy.QtWidgets import *


def init(
        working_path: Annotated[Path, {'mode': 'w'}] = 'array.json'
):
    # now load files from working_folder
    param_dict = json.load(working_path)

    #widget = create_params_widget(param_dict)
    # viewer.window.add_dock_widget(widget, name='Section tool', area='left')


class MassWidget(QWidget):
    def __init__(self, viewer):
        self.viewer = viewer
        super().__init__()

    def create_params_widget(self, param_dict):
        """
        Creates the widget for a specific model's parameters to swap in and out
        """
        # Create a widget for the model parameters
        widget = QWidget()
        layout = QGridLayout()
        # Add the default model parameters
        for i, (label, model_param) in enumerate(param_dict.items()):
            # Create labels for each of the model parameters
            param_label = QLabel(f"{label}:")
            param_label.setToolTip(model_param.tooltip)
            layout.addWidget(param_label, i, 0)
            # Add the model parameter(s)
            param_values = model_param.values
            # Widget added depends on the input
            # True/False -> Checkbox
            if param_values is True or param_values is False:
                param_val_widget = QCheckBox()
                if param_values:
                    param_val_widget.setChecked(True)
                else:
                    param_val_widget.setChecked(False)
            # List -> ComboBox
            elif isinstance(param_values, list):
                param_val_widget = QComboBox()
                param_val_widget.addItems([str(i) for i in param_values])
            # Int/float -> LineEdit
            elif isinstance(param_values, (int, float, str)):
                param_val_widget = QLineEdit()
                param_val_widget.setText(str(param_values))
            else:
                raise ValueError(
                    f"Model parameter {label} has invalid type {type(param_values)}"
                )
            layout.addWidget(param_val_widget, i, 1)
        # Tighten up margins and set layout
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        return widget
