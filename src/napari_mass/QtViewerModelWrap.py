from napari.qt import QtViewer


class QtViewerModelWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer
        self.setAcceptDrops(False)  # do not accept file drop
