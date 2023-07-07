# gitk developer notes

## Repository architecture

gitk is organized into modules that provide related functions, each in a subfolder. The parent `gitk/` folder holds functionality that spans across modules, such as the general (top-level) argument parser (in `cli.py`), constants (`const.py`), and any other utility or general-purpose code that is used across modules.

## Using modules from Python

This repo is divided into modules. Each module should be written in a way that it provides utility as a Python library. For example, you can call functions in the `hmm` module like this:

```
import gitk

gitk.hmm.function()
```


## Command-line interfaces

In addition to being importable from Python, *some* modules also provide a CLI. For these, developers provide a subcommand for CLI use. The root `gitk` package provides a generalized command-line interface with the command `gitk`. The modules that provide CLIs then correspond to CLI commands, *e.g* `gitk hmm` or `gitk likelihood`, with the corresponding code contained within a sub-folder named after the model:

```
gitk <module> ...
```

This is implemented within each module folder with:

- `gitk/<module>/cli.py` - defines the command-line interface and provides a subparser for this module's CLI command.