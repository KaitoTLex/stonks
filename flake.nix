{
  description = "Stonks";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [
          (final: prev: {
            pytorch = prev.pytorch.override {
              cudaSupport = true;
              cudatoolkit = prev.cudaPackages.cudatoolkit_11;
            };
          })
        ];

        pkgs = import nixpkgs {
          inherit system;
          overlays = overlays;
          config.allowUnfree = true; # for CUDA
        };

        pythonEnv = pkgs.python310.withPackages (
          ps: with ps; [
            pip
            numpy
            pandas
            matplotlib
            requests
            scipy
            gymnasium
            tensorboard
            wandb
            flake8
            black
            isort
            pytest
            yfinance
            rich

            # Torch + CUDA from override
            pkgs.pytorch
          ]
        );
      in
      {
        devShell = pkgs.mkShell {
          name = "rl-stock-trader-shell";
          buildInputs = [
            pythonEnv
            pkgs.cudaPackages.cudatoolkit_11
            pkgs.cudaPackages.cudnn
            pkgs.cudaPackages.cublas
            pkgs.cudaPackages.cufft
            pkgs.cudaPackages.nccl
          ];

          shellHook = ''
            export PYTHONNOUSERSITE="true"
            export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit_11}
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.cudaPackages.cudatoolkit_11}/lib
            echo "Python with PyTorch (CUDA-enabled) ready."
          '';
        };
      }
    );
}
