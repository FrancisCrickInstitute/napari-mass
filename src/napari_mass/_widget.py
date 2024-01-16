# Using Annotated: https://napari.org/stable/gallery/magic_parameter_sweep.html
# magicgui: https://forum.image.sc/t/magicgui-select-folder-instead-of-file/76378
# multiple viewer: https://napari.org/stable/gallery/multiple_viewer_widget.html
# https://forum.image.sc/t/how-to-maximize-this-custom-mulitple-view-grid/84400


import napari
from napari.layers import Layer
from napari.qt import QtViewer
from napari.components.viewer_model import ViewerModel
import os
from qtpy.QtCore import Qt
from qtpy.QtWidgets import *
import yaml

from napari_mass.parameters import *


def get_reader(path):
    if isinstance(path, list):
        path = path[0]

    ext = os.path.splitext(path)[1]

    if ext in READER_SUPPORTED_TYPES:
        return widget.image_dropped
    return None


class QtViewerModelWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer
        self.setAcceptDrops(False)  # do not accept file drop


class ParamControl:
    def __init__(self, main_widget, event, params, param_label):
        self.main_widget = main_widget
        self.params = params
        self.param_label = param_label
        event.connect(self.changed)

    def changed(self, value):
        params = self.params
        for key in self.param_label.split('.'):
            params = params[key]
        params['value'] = value
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
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        global widget
        widget = self

        self.clear_controls()
        self.viewer = napari_viewer

        self.viewer_model = ViewerModel()
        self.detail_viewer = QtViewerModelWrap(self.viewer, self.viewer_model)

        self.params_widget = self.create_widgets_from_template(PROJECT_TEMPLATE)
        self.setOrientation(Qt.Vertical)
        self.addWidget(self.params_widget)
        self.addWidget(self.detail_viewer)
        self.setMaximumWidth(300)   # only way found to limit detail_viewer initial width

        self.set_project(False, 0)
        self.init_imaging()

    def set_project(self, set, tabs_enable=None):
        self.project_set = set
        for index in range(self.params_widget.count()):
            if tabs_enable is not None:
                set0 = (index <= tabs_enable)
            else:
                set0 = set
            self.params_widget.setTabEnabled(index, set0)

    def init_imaging(self):
        from napari_mass.Imaging import Imaging
        self.imaging = Imaging(self.params)

    def image_dropped(self, path):
        self.all_widgets['input.source.filename'].setText(path)
        self.viewer.layers.clear()
        return self.imaging.init_layers()

    def clear_controls(self):
        self.all_widgets = {}
        self.param_controls = {}
        self.path_controls = {}
        self.tab_index = 0

    def create_widgets_from_template(self, path):
        with open(path, 'r') as infile:
            self.params = yaml.load(infile, Loader=yaml.Loader)
        self.tab_names = []
        tab_widget = QTabWidget()
        for label0, section_dict in self.params.items():
            tip = None
            self.tab_names.append(label0)
            label = label0.replace('_', ' ').capitalize()
            if 'label' in section_dict:
                label = section_dict.pop('label')
            if 'tip' in section_dict:
                tip = section_dict.pop('tip')

            widget = self.create_params_widget(label0, section_dict)
            if tip is not None:
                widget.settabText(tip)
            tab_widget.addTab(widget, label)
            tab_widget.currentChanged.connect(self.tab_changed)
        return tab_widget

    def tab_changed(self, new_index):
        if new_index != self.tab_index:
            if self.tab_index == 0:
                self.imaging.init_output()
            if self.viewer.layers:
                current_tab = self.tab_names[new_index]
                if current_tab in self.layer_names:
                    index = self.layer_names.index(current_tab)
                    self.viewer.layers.selection.active = self.viewer.layers[index]
            self.tab_index = new_index

    def layer_changed(self, event):
        new_selection = list(event.added)
        if len(new_selection) > 0:
            layer_name = new_selection[0].name
            if layer_name in self.tab_names:
                index = self.tab_names.index(layer_name)
                self.tab_index = index  # prevent event loop
                self.params_widget.setCurrentIndex(index)   # does not seem to redraw tab

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
                self.param_controls[param_label] = ParamControl(self, var_widget.clicked, self.params, param_label)
            elif value_type == 'float':
                var_widget = QDoubleSpinBox()
                if value is not None:
                    var_widget.setValue(value)
                self.param_controls[param_label] = ParamControl(self, var_widget.valueChanged, self.params, param_label)
            elif value_type == 'int':
                var_widget = QSpinBox()
                if value is not None:
                    var_widget.setValue(value)
                self.param_controls[param_label] = ParamControl(self, var_widget.valueChanged, self.params, param_label)
            elif value_type is not None:
                var_widget = QLineEdit()
                if value is not None:
                    var_widget.setText(value)
                self.param_controls[param_label] = ParamControl(self, var_widget.textChanged, self.params, param_label)

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

            if 'edit' in template and not template['edit']:
                var_widget.setReadOnly(True)

        #layout.setContentsMargins(0, 0, 0, 0)  # Tighten up margins
        widget.setLayout(layout)
        return widget

    def enable_tabs(self):
        for i in range(self.params_widget.count())[1:]:
            self.params_widget.setTabEnabled(i, True)

    def save_params(self):
        if self.project_set:
            path = self.params['project']['filename']['value']
            with open(path, 'w') as outfile:
                yaml.dump(self.params, outfile, default_flow_style=None, sort_keys=False)

    def dialog_project_file(self, path):
        self.set_project(True, 1)   # TODO: fix: should only enable tab 0 and 1!
        if os.path.exists(path):
            self.clear_controls()
            self.replaceWidget(0, self.create_widgets_from_template(path))
            self.all_widgets['project.filename'].setText(path)
        else:
            self.save_params()
        self.imaging.set_params(self.params)

    def load_images(self):
        self.viewer.layers.clear()
        layer_infos = self.imaging.init_layers()
        self.layer_names = []
        for layer_info in layer_infos:
            layer = Layer.create(*layer_info)
            self.viewer.layers.selection.events.changed.connect(self.layer_changed)
            self.viewer.add_layer(layer)
            self.layer_names.append(layer_info[1]['name'])
        self.set_project(True)

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
    if len(types) > 1:
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
        result = result[0]
    else:
        result = QFileDialog.getOpenFileName(
            caption=caption, directory=value, filter=filter
        )
        result = result[0]
    if result == '':
        result = None
    return result


widget: MassWidget
