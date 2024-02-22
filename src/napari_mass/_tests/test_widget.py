# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment
def test_mass_widget(make_napari_viewer):
    viewer = make_napari_viewer()

    from napari_mass._widget import MassWidget
    mass_widget = MassWidget(viewer)

    assert mass_widget is not None
