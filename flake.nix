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
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          (
            (pkgs.python3.override {
              packageOverrides = self: super: {
                # torch = super.torch-bin;
                # torchvision = super.torchvision-bin;
                # torchaudio = super.torchaudio-bin;
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
              ]
            )
          )
        ];
      };
    };
}