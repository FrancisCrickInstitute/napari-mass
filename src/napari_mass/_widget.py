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
from napari_mass.util import get_dict


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
    FOLDER_PLACEHOLDER = '[this folder]'

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
        types = self.path_type.split('.')[1:]
        dialog_type = types[0].lower()
        caption = self.template.get('tip')

        if caption is None:
            caption = ' '.join(types).capitalize()

        filter = ''
        default_ext = None
        if len(types) > 1:
            file_type = types[1]
            if file_type.startswith('image'):
                filter += 'Images (*.tif *.tiff *.zarr);;'
                default_ext = '.tif'
            if 'json' in file_type:
                filter += 'JSON files (*.json);;'
                if default_ext is None:
                    default_ext = '.json'
            if 'yml' in file_type or 'yaml' in file_type:
                filter += 'YAML files (*.yml *.yaml);;'
                if default_ext is None:
                    default_ext = '.yml'
            if 'xml' in file_type:
                filter += 'XML files (*.xml);;'
                if default_ext is None:
                    default_ext = '.xml'
        self.default_ext = default_ext

        self.is_folder = False
        if dialog_type in ['folder', 'dir', 'directory']:
            self.is_folder = True
            result = QFileDialog.getExistingDirectory(
                caption=caption, directory=value
            )
            self.finish_result(result)
        elif dialog_type == 'save':
            result = QFileDialog.getSaveFileName(
                caption=caption, directory=value, filter=filter
            )
            self.finish_result(result[0])
        else:
            if 'zarr' in filter:
                value = os.path.join(value, self.FOLDER_PLACEHOLDER)
            self.dialog = QFileDialog(caption=caption, directory=value, filter=filter)
            self.dialog.accepted.connect(self.accept)
            self.dialog.exec()

    def accept(self):
        files = self.dialog.selectedFiles()
        if files:
            filename = files[0]
        else:
            filename = None
        if filename.endswith(self.FOLDER_PLACEHOLDER):
            filename = filename.replace('[this folder]', '')
            filename = filename.rstrip('/\\')
            self.is_folder = True
        self.finish_result(filename)

    def finish_result(self, result):
        if result is not None:
            if not self.is_folder and os.path.splitext(result)[1] == '':
                result += self.default_ext
            self.path_widget.setText(result)
            if self.function is not None:
                self.function(result)


class MassWidget(QSplitter):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        global widget
        widget = self

        self.project_set = False
        self.clear_controls()
        self.load_params(PROJECT_TEMPLATE)
        self.init_model()
        self.viewer = napari_viewer

        self.viewer_model = ViewerModel()
        self.detail_viewer = QtViewerModelWrap(self.viewer, self.viewer_model)

        self.params_widget = self.create_widgets()
        self.setOrientation(Qt.Vertical)
        self.addWidget(self.params_widget)
        self.addWidget(self.detail_viewer)
        self.setMaximumWidth(300)   # only way found to limit detail_viewer initial width

        self.enable_tabs(False, 1)

    def init_model(self):
        from napari_mass.DataModel import DataModel
        self.model = DataModel(self.params)

    def image_dropped(self, path):
        self.all_widgets['input.source.filename'].setText(path)
        return self.init_layers()

    def init_layers(self):
        self.viewer.layers.clear()
        layer_infos = self.model.init_layers()
        self.layer_names = []
        if layer_infos:
            for layer_info in layer_infos:
                self.layer_names.append(layer_info[1]['name'])
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.events.changed.connect(self.layer_changed)
            self.viewer.layers.events.inserted.connect(self.layer_added)
        self.enable_tabs()
        self.check_enabled_layers()
        return layer_infos

    def clear_controls(self):
        self.all_widgets = {}
        self.param_controls = {}
        self.path_controls = {}
        self.tab_index = 0

    def load_params(self, path):
        with open(path, 'r') as infile:
            self.params = yaml.load(infile, Loader=yaml.Loader)

    def create_widgets(self):
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

    def layer_added(self, event):
        data_layer_names = get_dict(self.params, 'input.layers')
        layer = event.value
        if layer.name in data_layer_names:
            layer.events.data.connect(self.layer_data_changed)

    def layer_data_changed(self, event):
        #print(event.source.name, event.action, event.data_indices, event.value)
        self.model.data_changed(event.source.name, event.action, event.data_indices, event.value)

    def tab_changed(self, new_index):
        if new_index != self.tab_index:
            if self.tab_index == 0:
                self.model.init_output()
            if self.tab_index == 1:
                self.check_enabled_layers()
            if self.viewer.layers:
                current_tab = self.tab_names[new_index]
                if current_tab in self.layer_names:
                    index = self.layer_names.index(current_tab)
                else:
                    index = 0
                self.viewer.layers.selection.active = self.viewer.layers[index]
            self.tab_index = new_index

    def layer_changed(self, event):
        # also triggered when new layer added (and auto-selected)
        index = 1
        new_selection = list(event.added)
        if len(new_selection) > 0:
            layer_name = new_selection[0].name
            if layer_name in self.tab_names:
                index = self.tab_names.index(layer_name)
            self.tab_index = index  # prevent event loop
            self.params_widget.setCurrentIndex(index)

    def check_enabled_layers(self):
        for index, name in enumerate(self.tab_names):
            if index > 1:
                enabled = (name in get_dict(self.params, 'input.layers'))
                self.params_widget.setTabEnabled(index, enabled)

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

    def enable_tabs(self, set=True, tab_index=-1):
        for index in range(self.params_widget.count()):
            if (set and (tab_index < 0 or index <= tab_index)) or (not set and index >= tab_index):
                self.params_widget.setTabEnabled(index, set)

    def save_params(self):
        if self.project_set:
            path = self.params['project']['filename']['value']
            with open(path, 'w') as outfile:
                yaml.dump(self.params, outfile, default_flow_style=None, sort_keys=False)

    def dialog_project_file(self, path):
        self.project_set = True
        if os.path.exists(path):
            self.clear_controls()
            self.load_params(path)
            self.params_widget = self.create_widgets()
            self.replaceWidget(0, self.params_widget)
            self.enable_tabs(False, 2)
            self.all_widgets['project.filename'].setText(path)
        else:
            self.save_params()
        self.enable_tabs(tab_index=1)
        self.model.set_params(self.params)

    def create_layers(self):
        layer_infos = self.init_layers()
        if layer_infos:
            for layer_info in layer_infos:
                layer = Layer.create(*layer_info)
                self.viewer.add_layer(layer)
        else:
            QMessageBox.warning(self, 'Source input', 'No source image loaded')

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


widget: MassWidget
