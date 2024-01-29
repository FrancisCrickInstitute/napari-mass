# Using Annotated: https://napari.org/stable/gallery/magic_parameter_sweep.html
# magicgui: https://forum.image.sc/t/magicgui-select-folder-instead-of-file/76378
# multiple viewer: https://napari.org/stable/gallery/multiple_viewer_widget.html
# https://forum.image.sc/t/how-to-maximize-this-custom-mulitple-view-grid/84400


import napari
from napari.layers import Layer
from napari.components.viewer_model import ViewerModel
import os
from qtpy.QtCore import Qt
from qtpy.QtWidgets import *
import yaml

from napari_mass.PathControl import PathControl
from napari_mass.ParamControl import ParamControl
from napari_mass.QtViewerModelWrap import QtViewerModelWrap
from napari_mass.util import get_dict, translate
from napari_mass.parameters import *


def get_reader(path):
    if isinstance(path, list):
        path = path[0]

    ext = os.path.splitext(path)[1]

    if ext in READER_SUPPORTED_TYPES:
        return widget.image_dropped
    return None


class MassWidget(QSplitter):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        global widget
        widget = self

        self.project_set = False
        self.clear_controls()
        self.load_params(PROJECT_TEMPLATE)
        self.create_model()
        self.viewer = napari_viewer
        self.copied_shape = None
        self.shape_copy_mode = False
        self.shape_copy_layer = None
        self.shape_snap_edges = False
        self.section_order_mode = False
        self.section_ordering = False
        self.section_order = []

        self.viewer_model = ViewerModel()
        self.detail_viewer = self.create_view_widget()
        self.params_widget = self.create_params_tab_widget()
        self.setOrientation(Qt.Vertical)
        self.addWidget(self.params_widget)
        self.addWidget(self.detail_viewer)
        self.setMaximumWidth(300)   # only way found to limit detail_viewer initial width

        self.enable_tabs(False, 1)

    def create_model(self):
        # TODO: try to move import to top - reason this work-around works is unclear
        from napari_mass.DataModel import DataModel
        self.model = DataModel(self.params)

    def set_model_params(self):
        self.model.set_params(self.params)

    def init_project(self):
        if self.model.params_init_done:
            self.model.init()
            self.enable_tabs(tab_index=1)
            self.select_tab(1)
        else:
            QMessageBox.warning(self, 'MASS', 'MASS project needs to be selected first')

    def image_dropped(self, path):
        if self.model.init_done:
            self.all_widgets['input.source.filename'].setText(path)
            return self.init_layers()
        else:
            QMessageBox.warning(self, 'MASS image loader', 'MASS project needs to be initialised first')
            return None

    def init_layers(self):
        self.viewer.layers.clear()
        layer_infos = self.model.init_layers()
        self.layer_names = []
        if layer_infos:
            for layer_info in layer_infos:
                self.layer_names.append(layer_info[1]['name'])
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.events.changed.connect(self.on_layer_selection_changed)
            self.viewer.layers.events.inserted.connect(self.on_layer_added)
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

    def create_view_widget(self):
        widget = QWidget()
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignTop)
        viewer_widget = QtViewerModelWrap(self.viewer, self.viewer_model)
        layout.addWidget(viewer_widget, 0, 0, 1, -1)
        populate_detail_button = QPushButton('Populate')
        populate_detail_button.clicked.connect(self.on_populate_detail_clicked)
        propagate_detail_button = QPushButton('Propagate')
        populate_detail_button.clicked.connect(self.on_propagate_detail_clicked)
        layout.addWidget(populate_detail_button, 1, 0)
        layout.addWidget(propagate_detail_button, 1, 1)
        widget.setLayout(layout)
        return widget

    def create_params_tab_widget(self):
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
            tab_widget.currentChanged.connect(self.on_tab_selection_changed)
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
                self.param_controls[param_label] = ParamControl(self, var_widget.clicked, self.params, param_label,
                                                                function=function)
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
                layout.addWidget(var_widget, i, 0, 1, -1)

            if value_type is not None and value_type.startswith('path'):
                path_control = PathControl(template, var_widget, self.params, function=function)
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

    def on_layer_added(self, event):
        data_layer_names = get_dict(self.params, 'input.layers')
        layer = event.value
        if layer.name in data_layer_names:
            layer.events.data.connect(self.on_layer_data_changed)

            @layer.mouse_drag_callbacks.append
            def click_drag(layer, event):
                self.on_layer_pressed(layer, event)
                yield
                while event.type == 'mouse_move':
                    yield
                self.on_layer_released(layer, event)

    def on_layer_pressed(self, layer, event):
        if self.shape_copy_mode and layer.name == self.shape_copy_layer:
            value = translate(self.copied_shape, event.position)
            #if self.shape_snap_edges:
            #    value = np.flip(self.on_paste_function(np.flip(value)))
            layer.add(value)
            self.shape_copy_mode = False

    def on_layer_released(self, layer, event):
        if self.section_order_mode:
            if layer.selected_data:
                item = list(layer.selected_data)[0]
                if not self.section_ordering:
                    self.section_order = []
                    self.section_ordering = True
                if item not in self.section_order:
                    self.section_order.append(item)
                    self.update_order_labels(layer)

    def on_layer_data_changed(self, event):
        #print(event.source.name, event.action, event.data_indices, event.value)
        self.model.data_changed(event.source.name, event.action, event.data_indices, event.value)

    def on_tab_selection_changed(self, new_index):
        if new_index != self.tab_index:
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

    def on_layer_selection_changed(self, event):
        # also triggered when new layer added (and auto-selected)
        new_selection = list(event.added)
        if len(new_selection) > 0:
            layer_name = new_selection[0].name
            if layer_name in self.tab_names:
                self.select_tab(self.tab_names.index(layer_name))
        self.shape_copy_mode = False

    def select_tab(self, index):
        self.tab_index = index  # prevent event loop
        self.params_widget.setCurrentIndex(index)

    def check_enabled_layers(self):
        for index, name in enumerate(self.tab_names):
            if index > 1:
                enabled = (name in get_dict(self.params, 'input.layers'))
                self.params_widget.setTabEnabled(index, enabled)

    def save_params(self):
        if self.project_set:
            path = get_dict(self.params, 'project.filename')
            with open(path, 'w') as outfile:
                yaml.dump(self.params, outfile, default_flow_style=None, sort_keys=False)

    def dialog_project_file(self, path):
        self.project_set = True
        if os.path.exists(path):
            self.clear_controls()
            self.load_params(path)
            self.params_widget = self.create_params_tab_widget()
            self.replaceWidget(0, self.params_widget)
            self.enable_tabs(False, 1)
        else:
            self.save_params()
        self.set_model_params()

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

    def set_order_mode(self, value):
        layer = self.viewer.layers.selection.active
        if layer:
            self.section_order_mode = value
            if self.section_order_mode:
                layer.mode = 'direct'
                self.section_order = list(range(len(layer.data)))
            else:
                layer.mode = 'pan_zoom'
                if self.section_ordering and len(self.section_order) == len(layer.data):
                    self.update_section_order()
            self.section_ordering = False
            self.update_order_labels(layer)

    def update_order_labels(self, layer):
        if not self.section_order_mode:
            layer.text = {}
            return

        order_labels = []
        for index in range(len(layer.data)):
            if index in self.section_order:
                order_labels.append(self.section_order.index(index))
            else:
                order_labels.append('')
        layer.features['order'] = order_labels
        layer.text = {'string': '{order}', 'size': 20}

    def update_section_order(self):
        self.model.update_section_order(self.section_order)
        data_layers = self.model.init_data_layers()
        for layer_name, layer_info in data_layers.items():
            layer_index = self.layer_names.index(layer_name)
            self.viewer.layers[layer_index].data = layer_info[0]

    def detect_sample(self):
        pass

    def order_sample(self):
        pass

    def copy_sample(self):
        shape_copy_layer = 'sample'
        if shape_copy_layer in self.layer_names:
            layer_index = self.layer_names.index(shape_copy_layer)
            layer = self.viewer.layers[layer_index]
            selection = list(layer.selected_data)
            if len(selection) > 0:
                self.copied_shape = layer.data[selection[0]]
                self.shape_copy_mode = True
                self.shape_copy_layer = shape_copy_layer

    def on_populate_detail_clicked(self):
        # TODO: get current layer polygons from image, as image stack, assign to viewer
        # add empty layers with same config as main viewer
        pass

    def on_propagate_detail_clicked(self):
        # TODO: get drawn shape from current layer and propagate to corresponding layer -> self.model.data
        pass


    def propagate_sample(self):
        pass

    def propagate_roi(self):
        pass

    def propagate_focus(self):
        pass


widget: MassWidget
