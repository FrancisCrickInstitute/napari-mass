# Using Annotated: https://napari.org/stable/gallery/magic_parameter_sweep.html
# magicgui: https://forum.image.sc/t/magicgui-select-folder-instead-of-file/76378
# multiple viewer: https://napari.org/stable/gallery/multiple_viewer_widget.html
# https://forum.image.sc/t/how-to-maximize-this-custom-mulitple-view-grid/84400


import napari
import os
import yaml
from napari.qt import QtViewer
from napari.components.viewer_model import ViewerModel
from qtpy.QtCore import Qt
from qtpy.QtWidgets import *


class QtViewerModelWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer


class ParamControl:
    def __init__(self, main_widget, event, param_dict, param_label):
        self.main_widget = main_widget
        self.param_dict = param_dict
        self.param_label = param_label
        event.connect(self.changed)

    def changed(self, value):
        param_dict = self.param_dict
        for key in self.param_label.split('.'):
            param_dict = param_dict[key]
        param_dict['value'] = value
        self.main_widget.save_params()


class PathControl:
    def __init__(self, template, path_widget, function=None):
        self.path_widget = path_widget
        self.template = template
        self.path_type = template['type']
        self.function = function
        self.path_button = QPushButton('...')
        self.path_button.clicked.connect(self.show_dialog)

    def get_button_widget(self):
        return self.path_button

    def show_dialog(self):
        value = self.path_widget.text()
        result = path_dialog(self.path_type, value, caption=self.template.get('tip'))
        if result is not None:
            self.path_widget.setText(result)
            if self.function is not None:
                self.function(result)


class MassWidget(QSplitter):
    PROJECT_TEMPLATE = 'project_template.yml'

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.clear_controls()
        self.viewer = napari_viewer
        self.set_project = False

        self.viewer_model = ViewerModel()
        self.detail_viewer = QtViewerModelWrap(self.viewer, self.viewer_model)

        #self.viewer_model.add_points([(0,0), (0,100), (100,0), (100,100)], name='test_layer')

        self.params_widget = self.create_widgets_from_template(self.PROJECT_TEMPLATE)
        self.setOrientation(Qt.Vertical)
        self.addWidget(self.params_widget)
        self.addWidget(self.detail_viewer)
        self.setMaximumWidth(300)   # only way found to limit detail_viewer initial width

    def clear_controls(self):
        self.all_widgets = {}
        self.param_controls = {}
        self.path_controls = {}

    def create_widgets_from_template(self, path):
        with open(path, 'r') as infile:
            self.param_dict = yaml.load(infile, Loader=yaml.Loader)
        tab_widget = QTabWidget()
        for label0, section_dict in self.param_dict.items():
            tip = None
            label = label0.replace('_', ' ').capitalize()
            if 'label' in section_dict:
                label = section_dict.pop('label')
            if 'tip' in section_dict:
                tip = section_dict.pop('tip')

            widget = self.create_params_widget(label0, section_dict)
            if tip is not None:
                widget.settabText(tip)
            tab_widget.addTab(widget, label)
        return tab_widget

    def create_params_widget(self, param_prefix, template_dict):
        widget = QWidget()
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignTop)

        for i, (label0, template) in enumerate(template_dict.items()):
            param_label = param_prefix + '.' + label0
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
                sub_items = self.create_params_widget(param_label, template)
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
                self.param_controls[param_label] = ParamControl(self, var_widget.clicked, self.param_dict, param_label)
            elif value_type == 'float':
                var_widget = QDoubleSpinBox()
                if value is not None:
                    var_widget.setValue(value)
                self.param_controls[param_label] = ParamControl(self, var_widget.valueChanged, self.param_dict, param_label)
            elif value_type == 'int':
                var_widget = QSpinBox()
                if value is not None:
                    var_widget.setValue(value)
                self.param_controls[param_label] = ParamControl(self, var_widget.valueChanged, self.param_dict, param_label)
            elif value_type is not None:
                var_widget = QLineEdit()
                if value is not None:
                    var_widget.setText(value)
                self.param_controls[param_label] = ParamControl(self, var_widget.textChanged, self.param_dict, param_label)

            var_widget.setObjectName(param_label)
            self.all_widgets[param_label] = var_widget

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
                path_control = PathControl(template, var_widget, function=function)
                layout.addWidget(path_control.get_button_widget(), i, 2)
                self.path_controls[param_label] = path_control

        #layout.setContentsMargins(0, 0, 0, 0)  # Tighten up margins
        widget.setLayout(layout)
        return widget

    def save_params(self):
        if self.set_project:
            path = self.param_dict['project']['filename']['value']
            with open(path, 'w') as outfile:
                yaml.dump(self.param_dict, outfile, default_flow_style=None, sort_keys=False)

    def dialog_project_file(self, path):
        self.set_project = True
        if os.path.exists(path):
            self.clear_controls()
            self.replaceWidget(0, self.create_widgets_from_template(path))
            self.all_widgets['project.filename'].setText(path)
        else:
            self.save_params()

    def dialog_output_folder(self, path):
        print(path)

    def dialog_output_datafile(self, path):
        print(path)



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


def path_dialog(dialog_type, value, caption=None):
    types = dialog_type.split('.')[1:]
    dialog_type = types[0].lower()

    if caption is None:
        caption = ' '.join(types).capitalize()

    filter = ''
    if len(types) > 0:
        file_type = types[1]
        if file_type.startswith('image'):
            filter += 'Images (*.tif *.tiff *.zarr);;'
        if 'json' in file_type:
            filter += 'JSON files (*.json);;'
        if 'yml' in file_type or 'yaml' in file_type:
            filter += 'YAML files (*.yml *.yaml);;'
        if 'xml' in file_type:
            filter += 'XML files (*.xml);;'

    if dialog_type in ['folder', 'dir', 'directory']:
        result = QFileDialog.getExistingDirectory(
            caption=caption, directory=value
        )
    elif dialog_type in ['save', 'set']:
        result = QFileDialog.getSaveFileName(
            caption=caption, directory=value, filter=filter
        )
    else:
        result = QFileDialog.getOpenFileName(
            caption=caption, directory=value, filter=filter
        )
    if result[0]:
        return result[0]
    return None
