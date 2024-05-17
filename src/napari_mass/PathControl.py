import os
from qtpy.QtWidgets import QPushButton, QFileDialog, QStyle

from napari_mass.util import get_dict_value


class PathControl:
    def __init__(self, template, path_widget, params, param_label, function=None):
        self.template = template
        self.path_widget = path_widget
        self.params = params
        self.param_label = param_label
        self.function = function
        self.path_type = template['type']
        icon = path_widget.style().standardIcon(QStyle.SP_FileIcon)
        path_button = QPushButton(icon, '')
        path_button.clicked.connect(self.show_dialog)
        self.path_buttons = [path_button]
        if 'image' in self.path_type:
            icon = path_widget.style().standardIcon(QStyle.SP_DirIcon)
            dir_button = QPushButton(icon, '')
            dir_button.clicked.connect(lambda: self.show_dialog('dir'))
            self.path_buttons.append(dir_button)

    def get_button_widgets(self):
        return self.path_buttons

    def show_dialog(self, dialog_type=None):
        value = self.path_widget.text()
        value0 = get_dict_value(self.params, self.param_label)
        if not value and value0:
            value = value0
        types = self.path_type.split('.')[1:]
        if dialog_type is None:
            dialog_type = types[0].lower()
        caption = self.template.get('tip')

        if caption is None:
            caption = ' '.join(types).capitalize()

        filter = ''
        default_ext = None
        if len(types) > 1:
            file_type = types[1]
            if file_type.startswith('image'):
                filter += 'Images (*.tif *.tiff .zattrs);;'
                default_ext = '.tif'
            elif file_type.endswith('massproject'):
                filter += 'MASS project files (*.massproject.*);;'
                if default_ext is None:
                    default_ext = '.massproject.yml'
            elif file_type.endswith('mass'):
                filter += 'MASS project files (*.mass.*);;'
                if default_ext is None:
                    default_ext = '.mass.json'
            elif file_type.endswith('json'):
                filter += 'JSON files (*.json);;'
                if default_ext is None:
                    default_ext = '.json'
            elif file_type.endswith('yml') or file_type.endswith('yaml'):
                filter += 'YAML files (*.yml *.yaml);;'
                if default_ext is None:
                    default_ext = '.yml'
            elif file_type.endswith('xml'):
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
            self.process_result(result)
        elif dialog_type in ['save', 'set']:
            if dialog_type == 'set':
                options = QFileDialog.DontConfirmOverwrite  # only works on Windows?
            else:
                options = None
            result = QFileDialog.getSaveFileName(
                caption=caption, directory=value, filter=filter, options=options,
            )
            self.process_result(result[0])
        else:
            # open file
            result = QFileDialog.getOpenFileName(
                caption=caption, directory=value, filter=filter
            )
            # for zarr take parent path
            filepath = result[0]
            filename = os.path.basename(filepath)
            if filename in ['.zattrs', '.zgroup']:
                filepath = os.path.dirname(filepath)
            self.process_result(filepath)

    def process_result(self, filepath):
        if filepath:
            if not self.is_folder and os.path.splitext(filepath)[1] == '':
                filepath += self.default_ext
            self.path_widget.setText(filepath)
            if self.function is not None:
                self.function(filepath)
