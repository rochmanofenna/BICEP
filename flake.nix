{
  description = "BICEP dev shell";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python311             # core interpreter
            (python311.pkgs.buildPythonPackage {
              pname = "bicep-deps";
              version = "0";
              # Pull runtime deps the same way setup.cfg would (numpy, dask…)
              propagatedBuildInputs = with python311.pkgs; [
                numpy
                dask
                cupy        # CUDA users
                pytest
                pytest-benchmark
              ];
              src = null;
            })
            git
            # infra toys for Squarepoint drills:
            ansible
            docker
            slurm
            fio
            prometheus  # node_exporter, etc.
          ];

          # Handy env vars
          shellHook = ''
            export PYTHONBREAKPOINT=ipdb.set_trace
            echo "🐍  Dev shell ready – run 'pytest -q' or 'python -m bicep.demo'"
          '';
        };
      });
}
