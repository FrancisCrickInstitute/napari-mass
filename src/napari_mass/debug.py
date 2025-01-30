from napari import Viewer, run

#from napari_mass import MassWidget


viewer = Viewer()

dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget('napari-mass')
#dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget('napari-mass', 'Microscopy Array Section Setup')
run()

#widget = MassWidget(viewer)
#viewer.show(block=True)
