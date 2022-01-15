# TODO Refactor to use more elegant means of obtaining user settings (e.g. Django settings)

settings = {
    "models" : []
}

# TODO Use more sophisticated method of updating settings (updating included fields instead of replacing)
def configure(user_settings):
    global settings
    settings = user_settings

# TODO Make it easier for users to add custom classes
def register_model(model, model_manager):
    global settings
    settings.models.append({"type" : model, "manager" : model_manager})
