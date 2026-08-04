{
  description = "Python ML dev shell with CUDA-enabled PyTorch";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      pythonENV = (
        (pkgs.python3.override {
          packageOverrides = self: super: {
            # torch = super.torch-bin;
            # torchvision = super.torchvision-bin;
            # torchaudio = super.torchaudio-bin;

            transformers = super.transformers.overridePythonAttrs (old: rec {
              version = "5.11.0";
              src = pkgs.fetchFromGitHub {
                owner = "huggingface";
                repo = "transformers";
                rev = "v${version}";
                hash = "sha256-jlDzSA5tBxdx/tESG1m4RPf4HWQSiETivwKEVb8tvGs=";
              };
            });
          };
        }).withPackages
          (
            ps: with ps; [
              pip
              torch
              transformers
              datasets
              evaluate
              scikit-learn
              jinja2
              pandas
              accelerate
              tensorboard
              ipykernel
              jupyter-core
              # executing eval.ipynb headlessly (validate_with_eval_notebook.py)
              nbclient
              nbformat
              matplotlib
            ]
          )
      );
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pythonENV
        ];
        shellHook = ''
          # print information about the development shell
          echo "---------------------------------------------------------------------"
          echo "How to use this Nix development shell:"
          echo "python interpreter: ${pythonENV}/bin/python3"
          echo "python site packages: ${pythonENV}/${pythonENV.sitePackages}"
          echo "---------------------------------------------------------------------"
          echo "In case you need to set the PYTHONPATH environment variable, run:"
          echo "export PYTHONPATH=${pythonENV}/${pythonENV.sitePackages}"
          echo "---------------------------------------------------------------------"
        '';
      };
    };
}
