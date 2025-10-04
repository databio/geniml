PKG_NAME = "bedshift"

param_msg = """Params:
  chrom.sizes file: {chromsizes}
  shift:
    shift rate: {shiftrate}
    shift mean distance: {shiftmean}
    shift stdev: {shiftstdev}
    shift regions from file: {shiftfile}
  add:
    rate: {addrate}
    add mean length: {addmean}
    add stdev: {addstdev}
    add regions from file: {addfile}
    valid regions: {valid_regions}
  cut rate: {cutrate}
  drop rate: {droprate}
    drop regions from file: {dropfile}
  merge rate: {mergerate}
  outputfile: {outputfile}
  repeat: {repeat}
  yaml_config: {yaml_config}
  seed: {seed}
"""
