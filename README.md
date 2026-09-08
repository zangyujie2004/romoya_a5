# romoya_a5

Robotics data preprocessing, policy training, and deployment.

This repository uses three component branches:
- `preprocess`: dataset preprocessing.
- `policy`: policy implementation and training.
- `deploy`: policy deployment and inference.

On wujie-2, each component has its own Git working directory under
`/mnt/workspace/zyj/romoya_a5/`. The sibling `data/` directory stores datasets
outside these repositories. Shared policy changes must be explicitly integrated
across branches; updating one branch does not update the others.
