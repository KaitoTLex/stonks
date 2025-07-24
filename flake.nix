{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  name = "stonks";

  buildInputs = [
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.numpy
    pkgs.python310Packages.pandas
    pkgs.python310Packages.matplotlib
    pkgs.python310Packages.scipy
    pkgs.python310Packages.tensorboard
    pkgs.python310Packages.requests
    pkgs.python310Packages.setuptools
    pkgs.python310Packages.wheel
  ];

  shellHook = ''
    export PYTHONPATH=$(pwd)
  '';

  # We will use pip for these:
  shellCommands = ''
    python3 -m pip install --upgrade pip
    python3 -m pip install yfinance gymnasium torch wandb
  '';
}
