# Geniml Documentation Bugs and Issues Found

This document lists all bugs and issues found during evaluation of the geniml documentation hosted at https://docs.bedbase.org/geniml/

## Critical Bugs

### 1. **bedspace.md** - Incorrect search type flags (Lines 104, 113)
**Location:** `/tmp/bedbase/docs/geniml/tutorials/bedspace.md`

**Issue:** The search type flags are incorrectly documented with typos.

**Current (incorrect):**
```console
# Line 104
geniml bedspace search \
    -t lr2 \
    -d <path to distances> \
    ...

# Line 113
geniml bedspace search \
    -t rl2 \
    -d <path to distances> \
    ...
```

**Should be:**
```console
# Line 104 (region-to-label)
geniml bedspace search \
    -t r2l \
    -d <path to distances> \
    ...

# Line 113 (label-to-region)
geniml bedspace search \
    -t l2r \
    -d <path to distances> \
    ...
```

**Evidence:** The valid search types are defined in `geniml/bedspace/const.py`:
```python
class SearchType(Enum):
    l2r = "l2r"  # Label2Region
    r2l = "r2l"  # Region2Label
    r2r = "r2r"  # Region2Region
```

---

### 2. **pre-tokenization.md** - Wrong module import (Line 13)
**Location:** `/tmp/bedbase/docs/geniml/tutorials/pre-tokenization.md`

**Issue:** The import statement references a non-existent module `genimtools` instead of the correct module.

**Current (incorrect):**
```python
from genimtools.utils import write_tokens_to_gtok
```

**Should be:**
```python
from gtars.utils import write_tokens_to_gtok
```

**Evidence:** The function is imported from `gtars.utils` in `geniml/region2vec/utils.py:20`:
```python
from gtars.utils import read_tokens_from_gtok
```

**Note:** Need to verify if `write_tokens_to_gtok` exists in gtars or if a different function should be used.

---

### 3. **create-consensus-peaks.md** - Wrong CLI command (Line 60)
**Location:** `/tmp/bedbase/docs/geniml/tutorials/create-consensus-peaks.md`

**Issue:** The command uses an incorrect subcommand `build_model` that doesn't exist.

**Current (incorrect):**
```bash
geniml lh build_model --model-file model.tar \
                      --coverage-folder coverage/ \
                      --file-no `wc -l file_list.txt`
```

**Should be:**
```bash
geniml lh --model-file model.tar \
          --coverage-folder coverage/ \
          --file-no 4
```

**Evidence:** The CLI implementation in `geniml/cli.py:171` shows that `lh` doesn't have subcommands:
```python
if args.command == "lh":
    from .likelihood.build_model import main
    main(args.model_file, args.coverage_folder, ...)
```

**Additional note:** The backtick command `wc -l file_list.txt` should be replaced with the actual number.

---

## Minor Issues

### 4. **create-consensus-peaks.md** - Typo (Line 44)
**Location:** `/tmp/bedbase/docs/geniml/tutorials/create-consensus-peaks.md:44`

**Issue:** Spelling/grammar error.

**Current:** "beloved witch"
**Should be:** "below which"

**Full context:**
```
Depending on the task the universe can be smooth by setting `--merge`
flag with the distance beloved witch peaks should be merged together...
```

---

### 5. **create-consensus-peaks.md** - Missing download link placeholder
**Location:** `/tmp/bedbase/docs/geniml/tutorials/create-consensus-peaks.md` (Lines 4 and 84)

**Issue:** The tutorial references downloading files from "XXX" which is a placeholder.

**Current:**
```
...files, which can be downloaded from XXX. In there you will find...
```

**Action needed:** Replace "XXX" with actual download link for the example files.

---

### 6. **train-scembed-model.md** - Outdated TODO comment
**Location:** `/tmp/bedbase/docs/geniml/tutorials/train-scembed-model.md:25`

**Issue:** There's a TODO comment that appears to be already resolved.

**Current:**
```bash
# TODO: This needs to be the filtered_peak_bc_matrix, not the raw_peak_bc_matrix

wget https://cf.10xgenomics.com/samples/cell-atac/2.1.0/10k_pbmc_ATACv2_nextgem_Chromium_Controller/10k_pbmc_ATACv2_nextgem_Chromium_Controller_filtered_peak_bc_matrix.tar.gz
```

**Note:** The URL already uses `filtered_peak_bc_matrix`, so the TODO appears to be done. Should remove the TODO comment.

---

### 7. **tokenization.md** - File doesn't exist
**Location:** Referenced in mkdocs.yml but file doesn't exist at `/tmp/bedbase/docs/geniml/tutorials/tokenization.md`

**Issue:** The mkdocs.yml references a tutorial file that doesn't exist:
```yaml
- Tokenization: tutorials/tokenization.md
```

**Action needed:** Either create the missing file or remove the reference from mkdocs.yml.

---

## Documentation Location Note

The geniml documentation is hosted in the **bedbase repository** at:
- GitHub: https://github.com/databio/bedbase
- Location: `/docs/geniml/`
- Rendered at: https://docs.bedbase.org/geniml/

Changes need to be made in the bedbase repository, not the geniml repository.

---

---

## Bugs Fixed in geniml Repository

### 8. **geniml/likelihood/README.md** - Incorrect CLI commands (Lines 11, 17-18, 23)
**Location:** `/home/user/geniml/geniml/likelihood/README.md`

**Issue:** The README contained incorrect CLI commands that don't match the actual implementation.

**Status:** ✅ **FIXED**

**Changes made:**
- Changed `geniml lh build_model` to `geniml lh`
- Removed incorrect `geniml lh universe_hard` and `geniml lh universe_flexible` commands
- Added correct `geniml build-universe ml` command
- Updated parameter names to match actual CLI (--model-file, --coverage-folder, --file-no)

---

## Summary

### Issues in bedbase repository documentation
- **Critical bugs:** 3 (incorrect CLI commands/flags that will cause errors)
- **Minor issues:** 4 (typos, placeholders, missing files)
- **Total:** 7 issues

All these issues are in the documentation hosted in the bedbase repository under `/docs/geniml/tutorials/`.

### Issues in geniml repository
- **Fixed:** 1 (incorrect CLI commands in likelihood/README.md)

### Overall Summary
- **Total issues found:** 8
- **Fixed:** 1 (in geniml repository)
- **Requires bedbase repository changes:** 7

---

## Recommendations

1. **For bedbase repository:** Create a pull request to fix the 7 issues in the tutorial documentation
2. **For geniml repository:** The likelihood README fix has been applied and should be committed
3. **Testing:** After fixing the bedbase documentation, run through the tutorials to verify all commands work correctly
4. **Future:** Consider adding automated documentation testing to catch CLI mismatches
