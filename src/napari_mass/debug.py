from napari import Viewer, run

viewer = Viewer()

dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget('napari-mass')
#dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget('napari-mass', 'Microscopy Array Section Setup')

run()
