# Using Annotated: https://napari.org/stable/gallery/magic_parameter_sweep.html
# magicgui: https://forum.image.sc/t/magicgui-select-folder-instead-of-file/76378
# multiple viewer: https://napari.org/stable/gallery/multiple_viewer_widget.html
# https://forum.image.sc/t/how-to-maximize-this-custom-mulitple-view-grid/84400


import napari
import yaml
from napari.qt import QtViewer
from napari.components.viewer_model import ViewerModel
from qtpy.QtCore import Qt
from qtpy.QtWidgets import *


class QtViewerModelWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer


class MassWidget(QSplitter):
    PROJECT_TEMPLATE = 'project_template.yml'

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.params = {}
        self.viewer = napari_viewer

        self.viewer_model = ViewerModel()
        self.detail_viewer = QtViewerModelWrap(self.viewer, self.viewer_model)

        #self.viewer_model.add_points([(0,0), (0,100), (100,0), (100,100)], name='test_layer')

        self.params_widget = self.create_widgets_from_template(self.PROJECT_TEMPLATE)
        self.setOrientation(Qt.Vertical)
        self.addWidget(self.params_widget)
        self.addWidget(self.detail_viewer)
        self.setMaximumWidth(300)   # only way found to limit detail_viewer initial width

    def set_param(self, param, value):
        params = self.params
        for key in param.split('.'):
            if key not in params:
                params[key] = {}
                params = params[key]
        params[key] = value
        pass

    def create_widgets_from_template(self, path):
        with open(str(path), 'r') as infile:
            param_dict = yaml.load(infile, Loader=yaml.Loader)
        tab_widget = QTabWidget()
        for label0, section_dict in param_dict.items():
            tip = None
            label = label0.replace('_', ' ').capitalize()
            if 'label' in section_dict:
                label = section_dict.pop('label')
            if 'tip' in section_dict:
                tip = section_dict.pop('tip')

            widget = self.create_params_widget(section_dict)
            if tip is not None:
                widget.settabText(tip)
            tab_widget.addTab(widget, label)
        return tab_widget

    def create_params_widget(self, template_dict):
        """
        Creates the widget for a specific model's parameters to swap in and out
        """
        widget = QWidget()
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignTop)

        for i, (label0, template) in enumerate(template_dict.items()):
            label = label0.replace('_', ' ').capitalize()
            if 'label' in template:
                label = template['label']

            value = template.get('value')

            function = None
            if 'function' in template:
                function = getattr(self, template['function'])

            value_type = None
            if 'type' in template:
                value_type = template['type'].lower()
            elif 'value' in template:
                value_type = type(template['value']).__name__.lower()
            elif isinstance(template, dict):
                sub_items = self.create_params_widget(template)
                var_widget = QGroupBox()
                var_widget.setLayout(sub_items.layout())

            label_widget = QLabel(label)

            if value_type == 'button':
                var_widget = QPushButton(label)
                var_widget.clicked.connect(function)
                label_widget = None
            elif value_type == 'bool':
                var_widget = QCheckBox()
                if value:
                    var_widget.setChecked(True)
                else:
                    var_widget.setChecked(False)
            elif value_type == 'float':
                var_widget = QDoubleSpinBox()
                if value is not None:
                    var_widget.setValue(value)
            elif value_type == 'int':
                var_widget = QSpinBox()
                if value is not None:
                    var_widget.setValue(value)
            elif value_type is not None:
                var_widget = QLineEdit()
                if value is not None:
                    var_widget.setText(value)

            if 'tip' in template:
                if label_widget is not None:
                    label_widget.setToolTip(template['tip'])
                var_widget.setToolTip(template['tip'])

            if label_widget is not None:
                layout.addWidget(label_widget, i, 0)
                layout.addWidget(var_widget, i, 1)
            else:
                layout.addWidget(var_widget, i, 0, 1, 2)

            if value_type is not None and value_type.startswith('path'):
                button = QPushButton('...')
                button.clicked.connect(function)
                layout.addWidget(button, i, 2)

        #layout.setContentsMargins(0, 0, 0, 0)  # Tighten up margins
        widget.setLayout(layout)
        return widget

    def file_dialog(self, dialog_type, file_type, tip=None):
        result = QFileDialog.getOpenFileName(
            self, caption=tip
        )
        if result[0]:
            return result[0]
        return None

    def dialog_project_file(self):
        result = self.file_dialog('file.open', '*.json|*.yml|*.yaml', tip='Select project file')
        if result is not None:
            print(result)

    def dialog_source(self):
        result = self.file_dialog('file.open', '*.tif|*.tiff|*.zarr')
        if result is not None:
            print(result)

    def dialog_bf(self):
        pass

    def dialog_fluor(self):
        pass

    def dialog_fluor_hq(self):
        pass

    def dialog_fluor_hq_metadata(self):
        pass

    def dialog_fluor_hq_flatfield(self):
        pass

    def dialog_output_folder(self):
        pass

    def dialog_output_datafile(self):
        pass



    def detect_magnets(self):
        pass

    def order_magnets(self):
        pass

    def detect_sample(self):
        pass

    def order_sample(self):
        pass

    def copy_sample(self):
        pass

    def propagate_sample(self):
        pass

    def propagate_roi(self):
        pass

    def propagate_focus(self):
        pass
