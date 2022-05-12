$Env:CONDA_EXE = "/Users/emmanuelpeters/cs1470-final-runescape/cs1470-final-runescape-1/yes/bin/conda"
$Env:_CE_M = ""
$Env:_CE_CONDA = ""
$Env:_CONDA_ROOT = "/Users/emmanuelpeters/cs1470-final-runescape/cs1470-final-runescape-1/yes"
$Env:_CONDA_EXE = "/Users/emmanuelpeters/cs1470-final-runescape/cs1470-final-runescape-1/yes/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $True}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs