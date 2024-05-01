#from pathlib import Path
#import pkgutil
import importlib.resources
import yaml

from napari_mass.parameters import PROJECT_TEMPLATE


def get_project_template():
    # method 1: local path
    #RESOURCE_DIR = Path(__file__).parent.parent.parent / 'resources'
    #template_path = RESOURCE_DIR / PROJECT_TEMPLATE
    #file = open(template_path, 'r'):

    # method 2: pkgutil (old)
    #file = pkgutil.get_data('napari_mass', '../../resources/' + PROJECT_TEMPLATE)

    # method 3: importlib.resources (new)
    project_template_res = importlib.resources.files('napari_mass')
    resource_path = 'resources/'
    # if 'src' in path, move up first
    if 'src' in str(project_template_res):
        paths = str(project_template_res).replace('\\', '/').split('/')
        nback = len(paths) - paths.index('src')
        resource_path = '../' * nback + resource_path
    project_template_file_res = project_template_res.joinpath(resource_path + PROJECT_TEMPLATE)
    file = project_template_file_res.read_text()

    # load file content
    template = yaml.load(file, Loader=yaml.Loader)

    return template
