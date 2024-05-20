

class ParamControl:
    def __init__(self, main_widget, event, params, param_label, function=None):
        self.main_widget = main_widget
        self.params = params
        self.param_label = param_label
        self.function = function
        if self.function is not None:
            event.connect(self.function)
        else:
            event.connect(self.changed)

    def changed(self, value):
        params = self.params
        keys = self.param_label.split('.')
        for key in keys[:-1]:
            if key not in params:
                params[key] = {}
            params = params[key]
        params[keys[-1]] = value
        if self.param_label != 'project.filename':
            self.main_widget.save_params()
