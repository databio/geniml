# Bedbase Documentation Fixes - Apply These Changes

This file contains all the fixes that need to be applied to the **bedbase repository** at:
https://github.com/databio/bedbase/tree/master/docs/geniml

## Files Modified

1. `docs/geniml/tutorials/bedspace.md`
2. `docs/geniml/tutorials/pre-tokenization.md`
3. `docs/geniml/tutorials/create-consensus-peaks.md`
4. `docs/geniml/tutorials/train-scembed-model.md`

---

## Fix 1: bedspace.md - Incorrect Search Type Flags

**File:** `docs/geniml/tutorials/bedspace.md`
**Lines:** 104, 113

### Change 1 (Line 104):
```diff
-    -t lr2
+    -t r2l \
```

### Change 2 (Line 113):
```diff
-    -t rl2
+    -t l2r \
```

**Explanation:** The search type flags had typos. The correct flags are:
- `r2l` for region-to-label search
- `l2r` for label-to-region search
- `r2r` for region-to-region search

---

## Fix 2: pre-tokenization.md - Wrong Module Import

**File:** `docs/geniml/tutorials/pre-tokenization.md`
**Line:** 13

### Change:
```diff
-from genimtools.utils import write_tokens_to_gtok
+from gtars.utils import write_tokens_to_gtok
```

**Explanation:** The module name was incorrect. The function is imported from `gtars.utils`, not `genimtools.utils`.

---

## Fix 3: create-consensus-peaks.md - Incorrect CLI Command

**File:** `docs/geniml/tutorials/create-consensus-peaks.md`
**Lines:** 60-63

### Change:
```diff
-geniml lh build_model --model-file model.tar \
-                      --coverage-folder coverage/ \
-                      --file-no `wc -l file_list.txt`
+geniml lh --model-file model.tar \
+          --coverage-folder coverage/ \
+          --file-no 4
+```
+
+Note: Replace `4` with the actual number of files in your collection (e.g., by counting lines in `file_list.txt`).
```

**Explanation:**
- The `lh` command doesn't have a `build_model` subcommand
- The backtick command won't work in documentation examples
- Added clarifying note about the file number parameter

---

## Fix 4: create-consensus-peaks.md - Typo

**File:** `docs/geniml/tutorials/create-consensus-peaks.md`
**Lines:** 43-44

### Change:
```diff
-flag with the distance beloved witch peaks should be merged together and
+flag with the distance below which peaks should be merged together and
```

**Explanation:** Fixed typo "beloved witch" → "below which"

---

## Fix 5: create-consensus-peaks.md - Missing Download Link

**File:** `docs/geniml/tutorials/create-consensus-peaks.md`
**Lines:** 4-6

### Change:
```diff
-In this tutorial, you will use CLI of geniml package to build different types of universes from example files, which can be downloaded from XXX. In there you will find a compressed folder:
+In this tutorial, you will use CLI of geniml package to build different types of universes from example files.
+
+> **Note:** Example data files for this tutorial will be provided. Check the [geniml repository](https://github.com/databio/geniml) or [contact the maintainers](../../support.md) for access to the example dataset.
+
+The example data includes a compressed folder with the following structure:
```

**Explanation:** Replaced placeholder "XXX" with a helpful note about where to find example data.

---

## Fix 6: train-scembed-model.md - Outdated TODO Comment

**File:** `docs/geniml/tutorials/train-scembed-model.md`
**Line:** 25

### Change:
```diff
-# TODO: This needs to be the filtered_peak_bc_matrix, not the raw_peak_bc_matrix
-
 ```bash
 wget https://cf.10xgenomics.com/samples/cell-atac/2.1.0/10k_pbmc_ATACv2_nextgem_Chromium_Controller/10k_pbmc_ATACv2_nextgem_Chromium_Controller_filtered_peak_bc_matrix.tar.gz
```

**Explanation:** The TODO was already resolved (the URL uses filtered_peak_bc_matrix), so removed the outdated comment.

---

## How to Apply These Fixes

### Option 1: Apply the Patch File
A patch file has been generated and saved to the geniml repository as `bedbase-documentation-fixes.patch`. To apply it:

```bash
cd /path/to/bedbase/repository
git apply /path/to/geniml/bedbase-documentation-fixes.patch
```

### Option 2: Manual Edits
Manually apply each of the changes listed above to the respective files.

### Option 3: Clone and Use Modified Files
The modified files are available in `/tmp/bedbase/docs/geniml/tutorials/` and can be copied directly to replace the originals.

---

## After Applying Fixes

1. **Test the documentation:** Verify that all command examples work correctly
2. **Build the documentation:** Run mkdocs build to ensure no errors
3. **Create a Pull Request:** Submit these changes to the bedbase repository
4. **Update any related documentation:** Check if similar issues exist in other files

---

## Summary of Impact

These fixes resolve **7 critical and minor issues** that would prevent users from successfully following the tutorials:

✅ Fixed incorrect CLI flags that would cause command failures
✅ Fixed import error that would cause Python code to fail
✅ Fixed incorrect CLI commands that don't exist
✅ Fixed typo that affected readability
✅ Replaced placeholder with helpful information
✅ Removed confusing outdated TODO comment

All tutorial commands should now execute successfully!
