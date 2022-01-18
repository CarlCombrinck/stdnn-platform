def dictionary_update_deep(dictionary, key, value):
    """
    Updates dictionary recursively (including nested dicts) with key, value pair

    Parameters
    ----------
    dictionary : dict
        The dictionary to update
    key : any
        The key whose value requires updating
    value : any
        The updated value
    """
    for k, v in dictionary.items():
        if k == key:
            dictionary[key] = value
        elif isinstance(v, dict):
            dictionary_update_deep(v, key, value)