class SingletonMetaClass(type):
    def __init__(cls, *args, **kwargs):
        super(SingletonMetaClass, cls).__init__(*args, **kwargs)
        cls.__instance = None

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(SingletonMetaClass, cls).__call__(*args, **kwargs)
        return cls.__instance


class Config(metaclass=SingletonMetaClass):

    def __init__(self):
        self.container = {}
        self.mapping = {}

    def register(self, cls, model_type: str, *args: str, **kwargs: str):
        self.container[cls] = list(args) + list(kwargs.items())
        self.mapping[model_type] = cls

    def load_config(self, args, model_type=None, **kwargs):
        if model_type is None:
            model_type = args.model_type
        if model_type not in self.mapping:
            raise KeyError(f"Wrong model type: {model_type}")

        cls = self.mapping[model_type]
        params = {}
        for item in self.container[cls]:
            if isinstance(item, tuple):
                key, val = item
            else:
                key, val = item, item

            if val in kwargs:
                params[key] = kwargs[val]
            elif val in args:
                params[key] = getattr(args, val)
            else:
                pass

        return cls, params

    def get_model(self, args, model_type=None, **kwargs):
        cls, params = self.load_config(args, model_type, **kwargs)
        return cls(**params)


def register(model_type, *args, **kwargs):
    config = Config()

    def wrapper(cls):
        config.register(cls, model_type, *args, **kwargs)

        return cls

    return wrapper
