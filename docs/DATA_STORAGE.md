# Dataset storage and inventory

Source code, experiment records, and the dataset inventory are versioned in Git.
Model weights, runtime environments, private machine settings, and dataset bodies remain outside Git.

`data_inventory.json` lists 8,500 dataset files totaling 215,708,659,237 bytes
(200.894 GiB) under `ft_data/` and `eval_data/`.
These directories remain on the existing shared storage. Resolve manifest paths relative to the project data root.
The inventory omits processed caches and common cache directories.
Archive parts and extracted files are listed independently, so totals describe stored bytes rather than unique content.

The inventory records filesystem paths and byte sizes. File contents were not read or hashed;
it does not establish integrity, completeness of a benchmark, or acceptance of any experiment as paper evidence.
Preserved experiment identifiers and source snapshots retain their original names.

To reproduce experiments, use the shared dataset directories together with the corresponding
experiment configuration and source snapshot. A code checkout alone does not download these dataset bodies.
