# Contributor guide

## Repository organization

This repo is divided into modules. Each module is in a subfolder. To add functionality to geniml, you could add it to an existing module. If there's no existing module that fits, you could add your own module.

## Adding a new module

### Creating your module

Each module should be written in a way that it provides utility as a Python library. It should contain at least these files:

- `README.md` - describes how to use the code
- `<module>.py`, and other `.py` files - functions that provide utility for this module.

*All* the functions should be written to be useful via import, calling with `geniml.<module>.<function>`. For example:

```
import geniml

geniml.hmm.function()
```

### Adding your module to geniml

1. Put your module in a subfolder
2. Make sure to include a `__init__.py` so it's importable.
3. Add it to list of packages in `setup.py` 

### Shared code

Any variables, functions, or other code that is shared across modules should be placed in the parent module, which is held in the [geniml](geniml) folder.

