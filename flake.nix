{
  description = "Stock RL project with tests and CI, multi-arch compatible";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages.default = pkgs.mkShell {
          name = "stock-rl-shell";

          buildInputs = with pkgs; [
            python310
            python310Packages.pip
            python310Packages.numpy
            python310Packages.pandas
            python310Packages.matplotlib
            python310Packages.tensorboard
            python310Packages.wandb
            python310Packages.requests
            python310Packages.setuptools
            python310Packages.wheel
            python310Packages.scipy
            python310Packages.gcc # for compiling packages if needed
          ];

          shellHook = ''
            export PYTHONPATH=$(pwd)
            # Upgrade pip and install python deps if needed
            python3 -m pip install --upgrade pip
            python3 -m pip install -r requirements.txt
          '';
        };

        devShell = self.packages.${system};

        checks = pkgs.stdenv.mkDerivation {
          pname = "stock-rl-tests";
          version = "0.1";

          nativeBuildInputs = [
            pkgs.python310
            pkgs.python310Packages.pytest
          ];

          src = self;

          buildPhase = ''
            python3 -m pip install --upgrade pip
            python3 -m pip install -r requirements.txt
          '';

          checkPhase = ''
            pytest tests/
          '';

          meta = with pkgs.lib; {
            description = "Run unit tests for Stock RL";
            license = licenses.mit;
          };
        };
      }
    );
}
