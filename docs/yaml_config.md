# Use a YAML File to Specify Perturbations

Sometimes the default settings of bedshift does not allow enough control over perturbations. For example, the order of perturbations is fixed as shift, add, cut, merge, drop, so if you wanted to change the order you would have to specify multiple commands.
The same problem arises when you want to run multiple "add from file" commands - there is just no way to do it using a single command.

This is why we created the YAML config file perturbation option. In the YAML file, users can specify as many perturbations as they want, along with the parameters specific to each perturbation. An example of a YAML config file follows:

```
bedshift_operations:
  - add_from_file:
    file: exons.bed
    rate: 0.2
  - add_from_file:
    file: snp.bed
    rate: 0.05
  - shift_from_file:
    file: exons.bed
    rate: 0.4
    mean: 100
    stdev: 85
  - shift_from_file:
    file: snp.bed
    rate: 0.4
    mean: 2
    stdev: 1
  - merge:
    rate: 0.15
```

The order of perturbations is run in the same order they are specified. So in this example, we add from two different files, then also shift those regions that were just added. Finally we perform a merge at 15% rate.
