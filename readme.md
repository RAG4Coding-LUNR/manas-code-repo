# This README contains a list of patches made to get various experiments to work

## README: CodeSage Model Patch for `sentence_bert.py`

This document outlines a required manual patch to run the `Codesage-base-v2` model within the `CodeRagBench` framework.

## The Problem

The `CodeRagBench` benchmark fails to load the `Codesage-base-v2` model using the default `beir` library. The model requires the `trust_remote_code=True` flag to be set during initialization, which the library does not do by default.

---

## The Patch

To resolve this, a direct modification was made to the conda environment's installed library file.

* **File Modified:**
    ```
    miniconda3/envs/crag/lib/python3.10/site-packages/beir/retrieval/models/sentence_bert.py
    ```
* **Change:**
    On or around **line 30**, inside the `__init__` method, the `AutoModel.from_pretrained()` call was updated to include `trust_remote_code=True`.

    ```python
    # ... existing code ...
    class SentenceBERT:
        def __init__(self, model_path: Union[str, Tuple] = None, sep: str = " ", **kwargs):
            # ... existing code ...
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True) # <--- CHANGE APPLIED HERE
            # ... existing code ...
    ```

---

## ⚠️ Important

This is a **manual override** applied directly to a file within an installed package.

If the `beir-py` package is ever reinstalled or the `crag` conda environment is recreated from scratch, **this change will be lost**. The patch will need to be manually reapplied to get the `Codesage-base-v2` model working again.