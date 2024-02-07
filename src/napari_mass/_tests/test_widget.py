from napari_mass._widget import MassWidget


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment
def test_mass_widget(make_napari_viewer):
    viewer = make_napari_viewer()

    dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget('napari-mass')

    assert dock_widget is not None
