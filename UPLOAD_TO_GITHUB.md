# GitHub web upload — ready-to-upload

Target repository:

`https://github.com/kamaltelyu/A-BLAST-water-demand`

1. Unzip `A-BLAST-water-demand_GITHUB_READY.zip`.
2. Open the extracted folder.
3. On GitHub, open the repository and choose **Add file → Upload files**.
4. Drag **all files and folders inside the extracted folder** to the upload area. This preserves the prepared repository structure.
5. Commit with:

   `Add final MethodsX v3.2 rolling-origin reproducibility package`

The package includes updated files at the same paths as the earlier repository (README, legacy result TSV aliases, legacy figure aliases, and compatibility Python modules), so those older paths are overwritten rather than left stale.

After the commit, create a release:

- Tag: `v1.0.0-methodsx`
- Title: `A-BLAST MethodsX Reproducibility Package v1.0.0`

Use `RELEASE_NOTES_v1.0.0-methodsx.md` as the release description.

Do not upload raw Chennai or Melbourne CSV files; acquisition code and checksums are already included.
