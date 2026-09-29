{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  pythonEnv = pkgs.python312.override {
    packageOverrides = self: super: {
      torch = super.torch-bin;
    };
  };
in
pkgs.mkShell {
  buildInputs = [
    (pythonEnv.withPackages (ps: with ps; [
      pip
      torch-bin
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
      ipykernel
    ]))
  ];
}
