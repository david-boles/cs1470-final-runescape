# cs1470-final-runescape

# Git LFS setup:
1. Install git either following the [website](https://git-lfs.github.com/) or, on a Debian-based distro, `sudo apt install git-lfs`.
2. Enable it: `git lfs install`
3. Clone the repository! That's it.
4. If you need to enable LFS for other types of files, see the above link. Before pushing, run `git lfs ls-files` to make sure you set it up correctly.

# Python Virtual Environment Setup
- Run `source enter_venv.sh` to install and/or activate the virtual environment. `deactivate` exits the virtual environment as normal.
- To add requirements, look up the latest version of the library at `https://pypi.org/project/<package name>/` and edit `requirements.txt`. Then run `source enter_venv.sh reinstall`.