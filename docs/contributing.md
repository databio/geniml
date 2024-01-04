# Contributor guide

## Repository organization

This repo is divided into modules that provide related functions, each in a subfolder. The parent `geniml/` folder holds functionality that spans across modules, such as the general (top-level) argument parser (in `cli.py`), constants (`const.py`), and any other utility or general-purpose code that is used across modules.

## Using modules from Python

This repo is divided into modules. Each module should be written in a way that it provides utility as a Python library. For example, you can call functions in the `hmm` module like this:

```
import geniml

geniml.hmm.function()
```

## Command-line interfaces

In addition to being importable from Python, *some* modules also provide a CLI. For these, developers provide a subcommand for CLI use. The root `geniml` package provides a generalized command-line interface with the command `geniml`. The modules that provide CLIs then correspond to CLI commands, *e.g* `geniml hmm` or `geniml likelihood`, with the corresponding code contained within a sub-folder named after the model:

```
geniml <module> ...
```

This is implemented within each module folder with:

- `geniml/<module>/cli.py` - defines the command-line interface and provides a subparser for this module's CLI command.


## Adding a new module

To add functionality to geniml, you could add it to an existing module. Or, if no existing module fit, you could add a new module.

### Creating your module

Each module should be written in a way that it provides utility as a Python library. Organize your module with these files:

- `/docs/tutorials/<module>.md` - describes how to use the code
- `/geniml/<module>/main.py`, and other `.py` files - functions that provide utility for this module.

*All* the functions should be written to be useful via import, calling with `geniml.<module>.<function>`. For example:

```
import geniml

geniml.hmm.function()
```

### Adding your module to geniml

1. Put your module in a subfolder.
2. Make sure to include a `__init__.py` so it's importable.
3. Add it to list of packages in `setup.py`.
4. If it makes sense to have a CLI for this module, implement it in `geniml/<module_name>/cli.py`. Link this into the main cli by putting it under an appropriate command name following the pattern for other modules in `geniml/cli.py`.


### Shared code

Any variables, functions, or other code that is shared across modules should be placed in the parent module, which is held in the [geniml](geniml) folder.

