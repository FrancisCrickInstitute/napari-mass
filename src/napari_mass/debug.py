from napari import Viewer, run

viewer = Viewer()

dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget('napari-mass')
#dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget('napari-mass', 'Start Microscopy Array Section Setup')

# Click some buttons to save time
#plugin_widget.subwidgets["task"].task_buttons["mito"].click()
#plugin_widget.subwidgets["nxf"].nxf_profile_box.setCurrentIndex(1)
#plugin_widget.subwidgets["nxf"].overwrite_btn.click()

run()
