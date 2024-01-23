

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
        for key in self.param_label.split('.'):
            params = params[key]
        params['value'] = value
        self.main_widget.save_params()
