import os
from PyQt5.QtWidgets import QPushButton, QFileDialog

from napari_mass.util import get_dict


class PathControl:
    FOLDER_PLACEHOLDER = '[this folder]'

    def __init__(self, template, path_widget, params, function=None):
        self.template = template
        self.path_widget = path_widget
        self.params = params
        self.function = function
        self.path_type = template['type']
        self.path_button = QPushButton('...')
        self.path_button.clicked.connect(self.show_dialog)

    def get_button_widget(self):
        return self.path_button

    def show_dialog(self):
        value = self.path_widget.text()
        if not value:
            project_path = get_dict(self.params, 'project.filename')
            if project_path is not None:
                base_path = os.path.dirname(project_path)
                value = base_path
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
