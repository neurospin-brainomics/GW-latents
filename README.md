$$ \color{red}\boxed{\mathbf{\Large \text{THIS FILE MUST BE REPLACED WITH YOUR OWN PROJECT DESCRIPTION}}}$$

# USAGE

All neurospin-brainomics repository should be created using this
template. Each branch of this repository correspond to a template for different task. Just select the branch that fit your needs, and use the `USE THIS TEMPLATE` green button on the top-right. Available templates are:

- **[main](https://github.com/neurospin-brainomics/Template)**: the default template with only git-hooks
- **[docker/pixi-single](https://github.com/neurospin-brainomics/Template/tree/docker/pixi-single)**: docker single image of a `pixi/python` based projec (*)
- **[docker/uv-single](https://github.com/neurospin-brainomics/Template/tree/docker/uv-single)**: docker single image of a `uv/python` based projec (*)
- **[docker/pixi-multiple](https://github.com/neurospin-brainomics/Template/tree/docker/pixi-multiple)**: docker multiple images of a `pixi/python` based project (*)

(*): with instructions to deploy images on TGCC

## Initialisation after `GIT CLONE` 

**This has to be done once each time you clone a repository**

To activate neurospin-brainomics hooks use the following command:

```shell
~> ./.activate.sh
```

These hooks provide protections against unwanted actions, such as commit
on the main branch.

## Contribution guidelines

**Nine rules sofar:**

1. Always create a new branch for each feature, fix, or experiment – never work directly on main.
1. Use clear, descriptive branch names (e.g., feature/login, bugfix/api-timeout).
1. Keep main branch stable, deployable, and protected.
1. Commit often with meaningful messages that explain why, not just what.
1. Regularly pull/rebase from main to keep your branch up to date.
1. Open a Pull Request (PR) to merge into main; never push directly.
1. Ask for code review and address feedback before merging.
1. Squash or rebase commits to keep the git history clean.
1. Delete merged branches to avoid clutter.

